//! O mundo: grelha de monómeros, fluido com temperatura, luz UV e terreno.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::{Gpu, groups};
use crate::params::{Fumarole, SimParams, WorldConfig};
use crate::shaders::{self, CHEM_CELL_CAP};

/// Máximo de passos de simulação por frame (cada um tem a sua cópia dos params).
pub const MAX_STEPS_PER_FRAME: u32 = 64;
/// Alinhamento dos offsets dinâmicos de uniform (o mínimo garantido é 256).
const PARAMS_STRIDE: u64 = 256;
pub const MAX_FUMAROLES: usize = 64;

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

/// Definições do mundo que não vão para a GPU como parâmetros.
#[derive(Clone, Copy, Debug)]
pub struct WorldSettings {
    pub fluid_enabled: bool,
    /// Resolve o fluido de N em N passos, com dt×N (2 no v3).
    pub fluid_substep: u32,
    /// Iterações de Jacobi (arredondado para par: o resultado tem de cair em pressure_a).
    pub jacobi_iters: u32,
    /// Recalcula a luz UV de N em N passos.
    pub light_interval: u32,
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self { fluid_enabled: true, fluid_substep: 2, jacobi_iters: 10, light_interval: 100 }
    }
}

struct Pipelines {
    transport: wgpu::ComputePipeline,
    thermal_activation: wgpu::ComputePipeline,
    ledger: wgpu::ComputePipeline,
    uv_light: wgpu::ComputePipeline,
    clear_force_vectors: wgpu::ComputePipeline,
    update_temperature: wgpu::ComputePipeline,
    copy_temperature: wgpu::ComputePipeline,
    buoyancy: wgpu::ComputePipeline,
    gather_forces: wgpu::ComputePipeline,
    add_forces: wgpu::ComputePipeline,
    clear_forces: wgpu::ComputePipeline,
    diffuse_velocity: wgpu::ComputePipeline,
    advect_velocity: wgpu::ComputePipeline,
    vorticity: wgpu::ComputePipeline,
    divergence: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    subtract_gradient: wgpu::ComputePipeline,
    boundaries: wgpu::ComputePipeline,
}

pub struct World {
    pub cfg: WorldConfig,
    pub params: SimParams,
    pub settings: WorldSettings,
    pub fumaroles: Vec<Fumarole>,
    light_dirty: bool,
    params_buf: wgpu::Buffer,
    pub chem_buf: wgpu::Buffer,
    pub gamma_buf: wgpu::Buffer,
    pub light_buf: wgpu::Buffer,
    /// Velocidade final do fluido (velocity_a).
    pub velocity_buf: wgpu::Buffer,
    /// Temperatura publicada (temp_in).
    pub temp_buf: wgpu::Buffer,
    fumarole_buf: wgpu::Buffer,
    ledger_buf: wgpu::Buffer,
    ledger_staging: wgpu::Buffer,
    readback: Readback,
    frame_bg: wgpu::BindGroup,
    world_bg: wgpu::BindGroup,
    fluid_ab: wgpu::BindGroup,
    fluid_ba: wgpu::BindGroup,
    pipelines: Pipelines,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl World {
    pub fn new(gpu: &Gpu, cfg: WorldConfig, seed: u32) -> Self {
        let device = &gpu.device;
        let params = SimParams { seed, ..Default::default() };
        let cells = cfg.cells();
        let fcells = cfg.fluid_cells();

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: PARAMS_STRIDE * MAX_STEPS_PER_FRAME as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chem_buf = storage_buffer(device, "chem grid", cells * 16);
        let ledger_buf = storage_buffer(device, "ledger", 8 * 4);
        let gamma_buf = storage_buffer(device, "gamma grid", cells * 4);
        let light_buf = storage_buffer(device, "uv light", cells * 4);
        let ledger_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ledger staging"),
            size: 8 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vel_a = storage_buffer(device, "velocity a", fcells * 8);
        let vel_b = storage_buffer(device, "velocity b", fcells * 8);
        let p_a = storage_buffer(device, "pressure a", fcells * 4);
        let p_b = storage_buffer(device, "pressure b", fcells * 4);
        let div = storage_buffer(device, "divergence", fcells * 4);
        let temp_a = storage_buffer(device, "temperature in", fcells * 4);
        let temp_b = storage_buffer(device, "temperature out", fcells * 4);
        let force_vec = storage_buffer(device, "force vectors", fcells * 8);
        let forces = storage_buffer(device, "fluid forces", fcells * 8);
        let fumarole_buf = storage_buffer(device, "fumaroles", (MAX_FUMAROLES * size_of::<Fumarole>()) as u64);

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
        // Grupo 1 — mundo (resolução do ambiente).
        let world_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        // Grupo 2 — fluido. Bindings 0 e 2 (velocity_in, pressure_in) só de leitura.
        let fluid_entries: Vec<_> = (0..10).map(|b| storage_entry(b, matches!(b, 0 | 2 | 9))).collect();
        let fluid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid layout"),
            entries: &fluid_entries,
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
        let bind_all = |label, layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry> = bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        let world_bg = bind_all("world bg", &world_layout, &[&chem_buf, &ledger_buf, &gamma_buf, &light_buf]);
        // Ping-pong: "ab" lê a e escreve b (velocidade e pressão em simultâneo).
        let fluid_ab = bind_all(
            "fluid ab",
            &fluid_layout,
            &[&vel_a, &vel_b, &p_a, &p_b, &div, &temp_a, &temp_b, &force_vec, &forces, &fumarole_buf],
        );
        let fluid_ba = bind_all(
            "fluid ba",
            &fluid_layout,
            &[&vel_b, &vel_a, &p_b, &p_a, &div, &temp_a, &temp_b, &force_vec, &forces, &fumarole_buf],
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world pipeline layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&world_layout), Some(&fluid_layout)],
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
        let pipelines = Pipelines {
            transport: compute("transport_quanta"),
            thermal_activation: compute("thermal_activation"),
            ledger: compute("ledger_reduce"),
            uv_light: compute("compute_uv_light"),
            clear_force_vectors: compute("clear_force_vectors"),
            update_temperature: compute("update_temperature"),
            copy_temperature: compute("copy_temperature"),
            buoyancy: compute("buoyancy"),
            gather_forces: compute("gather_forces"),
            add_forces: compute("add_forces"),
            clear_forces: compute("clear_forces"),
            diffuse_velocity: compute("diffuse_velocity"),
            advect_velocity: compute("advect_velocity"),
            vorticity: compute("vorticity_confinement"),
            divergence: compute("compute_divergence"),
            jacobi: compute("jacobi_pressure"),
            subtract_gradient: compute("subtract_gradient"),
            boundaries: compute("enforce_boundaries"),
        };

        Self {
            cfg,
            params,
            settings: WorldSettings::default(),
            fumaroles: vec![Fumarole::v3_default()],
            light_dirty: true,
            params_buf,
            chem_buf,
            gamma_buf,
            light_buf,
            velocity_buf: vel_a,
            temp_buf: temp_a,
            fumarole_buf,
            ledger_buf,
            ledger_staging,
            readback: Readback::Idle,
            frame_bg,
            world_bg,
            fluid_ab,
            fluid_ba,
            pipelines,
        }
    }

    /// Enche o mundo com matéria inicial (determinista para a semente).
    /// Devolve a contagem exata do que foi escrito.
    pub fn seed_matter(&mut self, gpu: &Gpu, seed: u64) -> Ledger {
        let cells = seed_cells(&self.cfg, seed);
        gpu.queue.write_buffer(&self.chem_buf, 0, bytemuck::cast_slice(&cells));
        Ledger::from_cells(&cells)
    }

    /// Pede que a luz seja recalculada no próximo passo (p. ex. o terreno mudou).
    pub fn invalidate_light(&mut self) {
        self.light_dirty = true;
    }

    /// Grava `steps` passos de simulação. Os params de cada passo são escritos
    /// já (com epoch próprio) e ficam em offsets diferentes do mesmo buffer.
    pub fn encode_steps(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, steps: u32) {
        let steps = steps.min(MAX_STEPS_PER_FRAME);
        if steps == 0 {
            return;
        }
        let st = self.settings;
        let substep = st.fluid_substep.clamp(1, 4);
        let nfum = self.fumaroles.len().min(MAX_FUMAROLES);
        if nfum > 0 {
            queue.write_buffer(&self.fumarole_buf, 0, bytemuck::cast_slice(&self.fumaroles[..nfum]));
        }
        self.params.fumarole_count = nfum as u32;
        self.params.fluid_dt = self.params.dt * substep as f32;

        let mut bytes = vec![0u8; (PARAMS_STRIDE * steps as u64) as usize];
        for i in 0..steps {
            let p = SimParams { epoch: self.params.epoch.wrapping_add(i), ..self.params };
            let at = (PARAMS_STRIDE * i as u64) as usize;
            bytes[at..at + size_of::<SimParams>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        queue.write_buffer(&self.params_buf, 0, &bytes);

        let g = groups(self.cfg.grid_size, 16);
        let fg = groups(self.cfg.fluid_size, 16);
        let jacobi_iters = (st.jacobi_iters.clamp(2, 128) + 1) & !1;
        let pl = &self.pipelines;
        let (ab, ba) = (&self.fluid_ab, &self.fluid_ba);
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("world"), timestamp_writes: None });
        pass.set_bind_group(1, &self.world_bg, &[]);
        let run = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, n: [u32; 2]| {
            pass.set_bind_group(2, bg, &[]);
            pass.set_pipeline(p);
            pass.dispatch_workgroups(n[0], n[1], 1);
        };
        for i in 0..steps {
            let epoch = self.params.epoch.wrapping_add(i);
            pass.set_bind_group(0, &self.frame_bg, &[(PARAMS_STRIDE * i as u64) as u32]);

            if self.light_dirty || epoch % st.light_interval.max(1) == 0 {
                self.light_dirty = false;
                run(&mut pass, &pl.uv_light, ab, [1, 1]);
            }

            if st.fluid_enabled && epoch % substep == 0 {
                let f = [fg, fg];
                run(&mut pass, &pl.clear_force_vectors, ab, f);
                run(&mut pass, &pl.update_temperature, ab, f);
                run(&mut pass, &pl.copy_temperature, ab, f);
                run(&mut pass, &pl.buoyancy, ab, f);
                run(&mut pass, &pl.gather_forces, ab, f);
                run(&mut pass, &pl.add_forces, ab, f); // a -> b
                run(&mut pass, &pl.clear_forces, ba, f);
                run(&mut pass, &pl.diffuse_velocity, ba, f); // b -> a
                run(&mut pass, &pl.advect_velocity, ab, f); // a -> b
                run(&mut pass, &pl.vorticity, ba, f); // b -> a
                run(&mut pass, &pl.divergence, ab, f); // lê a
                // Pressão em ARRANQUE QUENTE (nunca é limpa): converge ao longo
                // dos frames. Número par de iterações: termina em pressure_a.
                for k in 0..jacobi_iters {
                    run(&mut pass, &pl.jacobi, if k % 2 == 0 { ab } else { ba }, f);
                }
                run(&mut pass, &pl.subtract_gradient, ab, f); // a -> b
                run(&mut pass, &pl.boundaries, ba, f); // b -> a (final em a)
                run(&mut pass, &pl.thermal_activation, ab, [g, g]);
            }

            run(&mut pass, &pl.transport, ab, [g, g]);
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
        pass.set_bind_group(2, &self.fluid_ab, &[]);
        pass.set_pipeline(&self.pipelines.ledger);
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

    /// Lê um buffer f32 inteiro de forma síncrona (testes).
    pub fn read_f32_blocking(&self, gpu: &Gpu, buf: &wgpu::Buffer) -> Vec<f32> {
        bytemuck::cast_slice(&gpu.read_buffer_blocking(buf)).to_vec()
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
