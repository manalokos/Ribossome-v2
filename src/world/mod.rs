//! O mundo: grelha de monómeros e (nas fases seguintes) fluido, luz e terreno.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::{Gpu, groups};
use crate::params::{SimParams, WorldConfig};
use crate::shaders::{self, CHEM_CELL_CAP};

/// Máximo de passos de simulação por frame (cada um tem a sua cópia dos params).
pub const MAX_STEPS_PER_FRAME: u32 = 64;
/// Alinhamento dos offsets dinâmicos de uniform (o mínimo garantido é 256).
const PARAMS_STRIDE: u64 = 256;

/// Contagem exata da matéria livre, por canal (A U G C) e estado.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    pub act: [u32; 4],
    pub spent: [u32; 4],
}

impl Ledger {
    pub fn total(&self) -> u64 {
        self.act.iter().chain(self.spent.iter()).map(|&v| v as u64).sum()
    }

    pub fn channel(&self, ch: usize) -> u64 {
        self.act[ch] as u64 + self.spent[ch] as u64
    }

    /// Conta no CPU a partir do conteúdo bruto da grelha (u32 por célula×canal).
    pub fn from_cells(cells: &[u32]) -> Self {
        let mut l = Ledger::default();
        for (i, &v) in cells.iter().enumerate() {
            l.act[i % 4] += v & 0xFFFF;
            l.spent[i % 4] += v >> 16;
        }
        l
    }

    fn from_gpu(words: &[u32]) -> Self {
        let mut l = Ledger::default();
        l.act.copy_from_slice(&words[0..4]);
        l.spent.copy_from_slice(&words[4..8]);
        l
    }
}

enum Readback {
    Idle,
    /// Cópia gravada neste frame; falta pedir o map depois do submit.
    Encoded,
    Mapping(Arc<AtomicBool>),
}

pub struct World {
    pub cfg: WorldConfig,
    pub params: SimParams,
    params_buf: wgpu::Buffer,
    pub chem_buf: wgpu::Buffer,
    ledger_buf: wgpu::Buffer,
    ledger_staging: wgpu::Buffer,
    readback: Readback,
    frame_bg: wgpu::BindGroup,
    world_bg: wgpu::BindGroup,
    transport_pl: wgpu::ComputePipeline,
    ledger_pl: wgpu::ComputePipeline,
}

impl World {
    pub fn new(gpu: &Gpu, cfg: WorldConfig, seed: u32) -> Self {
        let device = &gpu.device;
        let params = SimParams { seed, ..Default::default() };

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: PARAMS_STRIDE * MAX_STEPS_PER_FRAME as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chem_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chem grid"),
            size: cfg.cells() * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ledger_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ledger"),
            size: 8 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ledger_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ledger staging"),
            size: 8 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grupo 0 — frame: params com offset dinâmico (uma cópia por passo).
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(size_of::<SimParams>() as u64),
                },
                count: None,
            }],
        });
        // Grupo 1 — mundo.
        let storage_rw = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let world_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world layout"),
            entries: &[storage_rw(0), storage_rw(1)],
        });

        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bg"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<SimParams>() as u64),
                }),
            }],
        });
        let world_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world bg"),
            layout: &world_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: chem_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ledger_buf.as_entire_binding() },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world pipeline layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&world_layout)],
            immediate_size: 0,
        });
        let module = shaders::create(device, &shaders::WORLD, &cfg);
        let compute = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(shaders::entry(&shaders::WORLD, entry)),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let transport_pl = compute("transport_quanta");
        let ledger_pl = compute("ledger_reduce");

        Self {
            cfg,
            params,
            params_buf,
            chem_buf,
            ledger_buf,
            ledger_staging,
            readback: Readback::Idle,
            frame_bg,
            world_bg,
            transport_pl,
            ledger_pl,
        }
    }

    /// Enche o mundo com matéria inicial (determinista para a semente).
    /// Devolve a contagem exata do que foi escrito.
    pub fn seed_matter(&mut self, gpu: &Gpu, seed: u64) -> Ledger {
        let cells = seed_cells(&self.cfg, seed);
        gpu.queue.write_buffer(&self.chem_buf, 0, bytemuck::cast_slice(&cells));
        Ledger::from_cells(&cells)
    }

    /// Grava `steps` passos de simulação. Os params de cada passo são escritos
    /// já (com epoch próprio) e ficam em offsets diferentes do mesmo buffer.
    pub fn encode_steps(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, steps: u32) {
        let steps = steps.min(MAX_STEPS_PER_FRAME);
        if steps == 0 {
            return;
        }
        let mut bytes = vec![0u8; (PARAMS_STRIDE * steps as u64) as usize];
        for i in 0..steps {
            let p = SimParams { epoch: self.params.epoch.wrapping_add(i), ..self.params };
            let at = (PARAMS_STRIDE * i as u64) as usize;
            bytes[at..at + size_of::<SimParams>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        queue.write_buffer(&self.params_buf, 0, &bytes);

        let g = groups(self.cfg.grid_size, 16);
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("chem"), timestamp_writes: None });
        pass.set_bind_group(1, &self.world_bg, &[]);
        pass.set_pipeline(&self.transport_pl);
        for i in 0..steps {
            pass.set_bind_group(0, &self.frame_bg, &[(PARAMS_STRIDE * i as u64) as u32]);
            pass.dispatch_workgroups(g, g, 1);
        }
        drop(pass);
        self.params.epoch = self.params.epoch.wrapping_add(steps);
    }

    /// Grava a redução do livro-razão para `ledger_buf`.
    pub fn encode_ledger(&self, enc: &mut wgpu::CommandEncoder) {
        enc.clear_buffer(&self.ledger_buf, 0, None);
        let g = groups(self.cfg.grid_size, 16);
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ledger"), timestamp_writes: None });
        pass.set_bind_group(0, &self.frame_bg, &[0]);
        pass.set_bind_group(1, &self.world_bg, &[]);
        pass.set_pipeline(&self.ledger_pl);
        pass.dispatch_workgroups(g, g, 1);
    }

    /// Livro-razão assíncrono: se não houver leitura em curso, grava a redução
    /// e a cópia para o staging. Chamar `ledger_after_submit` depois do submit.
    pub fn encode_ledger_readback(&mut self, enc: &mut wgpu::CommandEncoder) {
        if !matches!(self.readback, Readback::Idle) {
            return;
        }
        self.encode_ledger(enc);
        enc.copy_buffer_to_buffer(&self.ledger_buf, 0, &self.ledger_staging, 0, 8 * 4);
        self.readback = Readback::Encoded;
    }

    pub fn ledger_after_submit(&mut self) {
        if let Readback::Encoded = self.readback {
            let ready = Arc::new(AtomicBool::new(false));
            let flag = ready.clone();
            self.ledger_staging.map_async(wgpu::MapMode::Read, .., move |r| {
                if r.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
            self.readback = Readback::Mapping(ready);
        }
    }

    /// Devolve o livro-razão se a leitura já chegou (com um ou mais frames de atraso).
    pub fn poll_ledger(&mut self, device: &wgpu::Device) -> Option<Ledger> {
        let Readback::Mapping(ready) = &self.readback else { return None };
        device.poll(wgpu::PollType::Poll).ok();
        if !ready.load(Ordering::Acquire) {
            return None;
        }
        let ledger = {
            let view = self.ledger_staging.get_mapped_range(..).ok()?;
            Ledger::from_gpu(bytemuck::cast_slice(&view))
        };
        self.ledger_staging.unmap();
        self.readback = Readback::Idle;
        Some(ledger)
    }

    /// Lê a grelha inteira de forma síncrona (testes).
    pub fn read_cells_blocking(&self, gpu: &Gpu) -> Vec<u32> {
        bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.chem_buf)).to_vec()
    }

    /// Lê o livro-razão calculado na GPU de forma síncrona (testes).
    pub fn ledger_blocking(&self, gpu: &Gpu) -> Ledger {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.encode_ledger(&mut enc);
        gpu.queue.submit([enc.finish()]);
        Ledger::from_gpu(bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.ledger_buf)))
    }
}

/// splitmix64: RNG determinista, só para a sementeira no CPU.
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Matéria inicial provisória: manchas suaves e diferentes por canal, metade
/// ativada. Nunca passa da capacidade da célula.
fn seed_cells(cfg: &WorldConfig, seed: u64) -> Vec<u32> {
    let n = cfg.grid_size as usize;
    let mut rng = SplitMix(seed);
    let phase: Vec<f32> = (0..8).map(|_| rng.f32() * std::f32::consts::TAU).collect();
    let max_per_channel = CHEM_CELL_CAP / 4;
    let mut cells = vec![0u32; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let u = x as f32 / n as f32 * std::f32::consts::TAU;
            let v = y as f32 / n as f32 * std::f32::consts::TAU;
            for ch in 0..4 {
                let d = 0.5
                    + 0.25 * (3.0 * u + phase[ch]).sin() * (2.0 * v + phase[ch + 4]).cos()
                    + 0.25 * (5.0 * v + phase[ch + 4] + ch as f32).sin();
                let expect = d.clamp(0.0, 1.0) * (max_per_channel as f32 - 2.0);
                let count = ((expect + rng.f32()) as u32).min(max_per_channel);
                let act = (0..count).filter(|_| rng.f32() < 0.5).count() as u32;
                cells[(y * n + x) * 4 + ch] = act | ((count - act) << 16);
            }
        }
    }
    cells
}
