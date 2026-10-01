//! Profiler por segmento.
//!
//! Desligado: todos os segmentos de um frame vão num único encoder e uma
//! única submissão (sem custo).
//! Ligado: antes do primeiro segmento espera-se pelo trabalho pendente do
//! frame anterior e mede-se isso à parte ("backlog"); depois cada segmento é
//! submetido sozinho e medido com `submit + poll(Wait)`. Os números incluem a
//! latência de submissão, mas são estáveis e não usam timestamp queries (que
//! davam *device lost* no driver do Filipe no v3).
//!
//! Ligar ao arrancar com `RIBO_PROFILE=1`, ou no painel.

use std::time::{Duration, Instant};

pub const BACKLOG: &str = "backlog";

#[derive(Clone, Debug)]
pub struct SegmentStat {
    pub name: &'static str,
    /// Média exponencial, em ms.
    pub avg_ms: f64,
    pub last_ms: f64,
}

pub struct Profiler {
    pub enabled: bool,
    stats: Vec<SegmentStat>,
    print_every: Option<Duration>,
    last_print: Instant,
    /// Tempo de CPU do frame inteiro (ms, média exponencial).
    pub frame_ms: f64,
    frame_start: Instant,
}

impl Profiler {
    pub fn from_env() -> Self {
        let enabled = std::env::var("RIBO_PROFILE").map(|v| v != "0").unwrap_or(false);
        Self {
            enabled,
            stats: Vec::new(),
            print_every: enabled.then(|| Duration::from_secs(1)),
            last_print: Instant::now(),
            frame_ms: 0.0,
            frame_start: Instant::now(),
        }
    }

    pub fn stats(&self) -> &[SegmentStat] {
        &self.stats
    }

    fn record(&mut self, name: &'static str, d: Duration) {
        let ms = d.as_secs_f64() * 1000.0;
        match self.stats.iter_mut().find(|s| s.name == name) {
            Some(s) => {
                s.avg_ms += (ms - s.avg_ms) * 0.1;
                s.last_ms = ms;
            }
            None => self.stats.push(SegmentStat { name, avg_ms: ms, last_ms: ms }),
        }
    }

    /// Começa um frame. Devolve o gravador de segmentos.
    pub fn begin<'a>(&'a mut self, device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Frame<'a> {
        let now = Instant::now();
        let dt = now - self.frame_start;
        self.frame_ms += (dt.as_secs_f64() * 1000.0 - self.frame_ms) * 0.1;
        self.frame_start = now;
        if !self.enabled {
            self.stats.clear();
        } else {
            let t = Instant::now();
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            self.record(BACKLOG, t.elapsed());
        }
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        Frame { profiler: self, device, queue, encoder: Some(encoder) }
    }

    fn maybe_print(&mut self) {
        let Some(every) = self.print_every else { return };
        if !self.enabled || self.last_print.elapsed() < every {
            return;
        }
        self.last_print = Instant::now();
        let total: f64 = self.stats.iter().map(|s| s.avg_ms).sum();
        let parts: Vec<String> = self.stats.iter().map(|s| format!("{}={:.2}", s.name, s.avg_ms)).collect();
        eprintln!("[perf] frame={:.2}ms  gpu+sync={:.2}ms  {}", self.frame_ms, total, parts.join(" "));
    }
}

pub struct Frame<'a> {
    profiler: &'a mut Profiler,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
}

impl Frame<'_> {
    pub fn device(&self) -> &wgpu::Device {
        self.device
    }

    /// Grava um segmento com nome. Com o profiler ligado é submetido e medido sozinho.
    pub fn segment(&mut self, name: &'static str, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        if !self.profiler.enabled {
            f(self.encoder.as_mut().expect("encoder"));
            return;
        }
        let t = Instant::now();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        f(&mut enc);
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        self.profiler.record(name, t.elapsed());
    }

    /// Submete já command buffers externos (p. ex. os uploads do egui), para
    /// ficarem antes de qualquer segmento gravado a seguir, com ou sem profiler.
    pub fn submit_now(&mut self, bufs: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.queue.submit(bufs);
    }

    /// Submete o que falta e termina o frame.
    pub fn finish(mut self) {
        let enc = self.encoder.take().expect("encoder");
        self.queue.submit([enc.finish()]);
        self.profiler.maybe_print();
    }
}
