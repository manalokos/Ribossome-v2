//! Log automático da sessão: de `every` em `every` frames escreve em
//! `logs/ribossome.log` um bloco com velocidade, população, livro-razão,
//! tempos do profiler e TODOS os parâmetros e definições. Serve para
//! diagnosticar sem ter de copiar valores à mão.
//!
//! - O profiler (submit+wait por segmento) abranda um pouco; por isso só é
//!   ligado durante uma janela de `SAMPLE` frames antes de cada registo, e
//!   volta ao estado em que o utilizador o deixou.
//! - Tamanho limitado: quando o ficheiro passa de `max_bytes`, passa a
//!   `ribossome.log.1` (substituindo o anterior) e começa um novo.
//! - Variáveis: RIBO_LOG=0 desliga; RIBO_LOG_EVERY (frames, 600);
//!   RIBO_LOG_MAX_MB (4).

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::gpu::profiler::Profiler;
use crate::ui::UiState;
use crate::world::World;

/// Frames com o profiler ligado antes de cada registo.
const SAMPLE: u64 = 30;

pub struct RunLog {
    enabled: bool,
    path: PathBuf,
    every: u64,
    max_bytes: u64,
    frame: u64,
    started: Instant,
    /// O profiler foi ligado por nós (desligar depois do registo).
    forced_profiler: bool,
    header: String,
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl RunLog {
    /// `header`: descrição fixa da sessão (modo, tamanho do mundo, GPU).
    pub fn from_env(header: String) -> Self {
        let enabled = env("RIBO_LOG", 1u32) != 0;
        let mut log = Self {
            enabled,
            path: PathBuf::from("logs").join("ribossome.log"),
            every: env("RIBO_LOG_EVERY", 600u64).max(SAMPLE + 1),
            max_bytes: (env("RIBO_LOG_MAX_MB", 4.0f64) * 1024.0 * 1024.0) as u64,
            frame: 0,
            started: Instant::now(),
            forced_profiler: false,
            header,
        };
        if log.enabled {
            if let Err(e) = std::fs::create_dir_all("logs") {
                eprintln!("log: não consegui criar logs/: {e}");
                log.enabled = false;
            } else {
                log.write(&format!(
                    "\n######## sessão nova (unix {}) ########\n{}\nregisto de {} em {} frames; profiler ligado {} frames antes de cada registo\n",
                    unix_now(),
                    log.header,
                    log.every,
                    log.every,
                    SAMPLE
                ));
            }
        }
        log
    }

    /// Chamar no início de cada frame (antes de usar o profiler).
    pub fn before_frame(&mut self, prof: &mut Profiler) {
        if !self.enabled {
            return;
        }
        self.frame += 1;
        if self.frame % self.every == self.every - SAMPLE && !prof.enabled {
            prof.enabled = true;
            self.forced_profiler = true;
        }
    }

    /// Chamar no fim de cada frame.
    pub fn after_frame(&mut self, world: &World, st: &UiState, prof: &mut Profiler) {
        if !self.enabled || !self.frame.is_multiple_of(self.every) {
            return;
        }
        let entry = self.entry(world, st, prof);
        self.write(&entry);
        if self.forced_profiler {
            prof.enabled = false;
            self.forced_profiler = false;
        }
    }

    fn entry(&self, world: &World, st: &UiState, prof: &Profiler) -> String {
        let mut s = String::new();
        let t = self.started.elapsed().as_secs_f64();
        s += &format!("==== frame {}  t={t:.1} s  unix {} ====\n", self.frame, unix_now());
        s += &format!(
            "epoch {}  epochs/s {:.0}  frame {:.2} ms ({:.0} fps)  passos/frame {}  pausa {}  vsync {}  vista {}  cor agentes {}\n",
            world.params.epoch,
            st.stats.epochs_per_sec,
            prof.frame_ms,
            1000.0 / prof.frame_ms.max(1e-3),
            st.steps_per_frame,
            st.paused,
            st.vsync,
            st.view_mode,
            st.signal_view
        );
        match world.last_counters {
            Some(c) => {
                s += &format!(
                    "agentes vivos {}  nasc/s {:.1}  mortes/s {:.1}  | {:?}\n",
                    c.alive(world.cfg.max_agents),
                    st.stats.births_per_sec,
                    st.stats.deaths_per_sec,
                    c
                )
            }
            None => s += "agentes: ainda sem leitura\n",
        }
        if let Some(l) = &st.ledger {
            let base = st.baseline.total() as i64;
            s += &format!(
                "matéria total {} (Δ {:+} desde a semente)  livre {}  em agentes {}  ativ {:?}  gastos {:?}  presos {:?}  (leitura do epoch {})\n",
                l.total(),
                l.total() as i64 - base,
                l.free_total(),
                l.held_total(),
                l.act,
                l.spent,
                l.held,
                st.ledger_epoch
            );
        }
        if prof.enabled {
            s += "profiler (média exponencial):";
            for seg in prof.stats() {
                s += &format!("  {} {:.3} ms", seg.name, seg.avg_ms);
                if seg.name == "world" && !st.paused {
                    s += &format!(" ({:.3} ms/passo)", seg.avg_ms / st.steps_per_frame.max(1) as f64);
                }
            }
            s += "\n";
        }
        s += &format!("params {:?}\n", world.params);
        s += &format!("settings {:?}\n", world.settings);
        s
    }

    fn write(&mut self, text: &str) {
        if std::fs::metadata(&self.path).map(|m| m.len() > self.max_bytes).unwrap_or(false) {
            let old = self.path.with_extension("log.1");
            let _ = std::fs::remove_file(&old);
            let _ = std::fs::rename(&self.path, &old);
            let header = format!("(continua de ribossome.log.1)\n{}\n", self.header);
            self.append(&header);
        }
        self.append(text);
    }

    fn append(&mut self, text: &str) {
        let r = std::fs::OpenOptions::new().create(true).append(true).open(&self.path).and_then(|mut f| f.write_all(text.as_bytes()));
        if let Err(e) = r {
            eprintln!("log: falhou a escrita em {}: {e}; log desligado", self.path.display());
            self.enabled = false;
        }
    }
}
