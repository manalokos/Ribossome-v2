//! O mundo: grelha de monómeros, fluido com temperatura, luz UV e terreno.

pub mod terrain;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::{Gpu, groups};
use crate::params::{Agent, Fumarole, MgLevel, SimParams, SpawnRequest, WorldConfig};
use crate::shaders::{self, CHEM_CELL_CAP};

/// Máximo de passos de simulação por frame (cada um tem a sua cópia dos params).
pub const MAX_STEPS_PER_FRAME: u32 = 64;
/// Alinhamento dos offsets dinâmicos de uniform (o mínimo garantido é 256).
const PARAMS_STRIDE: u64 = 256;
pub const MAX_FUMAROLES: usize = 64;
const LEDGER_WORDS: u64 = 12;
/// Máximo de pedidos de sementes por frame.
pub const MAX_SPAWN_REQUESTS: usize = 4096;
/// Palavras por slot nos buffers de genoma e de corpo.
const SLOT_WORDS: u64 = 16;

/// Contagem exata da matéria livre, por canal (A U G C) e estado.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    pub act: [u32; 4],
    pub spent: [u32; 4],
    /// Matéria presa em agentes (genomas + complementos capturados).
    pub held: [u32; 4],
}

impl Ledger {
    pub fn total(&self) -> u64 {
        self.act.iter().chain(self.spent.iter()).chain(self.held.iter()).map(|&v| v as u64).sum()
    }

    pub fn free_total(&self) -> u64 {
        self.act.iter().chain(self.spent.iter()).map(|&v| v as u64).sum()
    }

    pub fn held_total(&self) -> u64 {
        self.held.iter().map(|&v| v as u64).sum()
    }

    pub fn channel(&self, ch: usize) -> u64 {
        self.act[ch] as u64 + self.spent[ch] as u64 + self.held[ch] as u64
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
        l.held.copy_from_slice(&words[8..12]);
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
    /// Pressão: multigrid (por omissão) ou Jacobi (o do v3, para comparar).
    pub multigrid: bool,
    /// Ciclos V do multigrid por resolução.
    pub mg_cycles: u32,
    /// Iterações de Jacobi (arredondado para par: o resultado tem de cair em pressure_a).
    pub jacobi_iters: u32,
    /// Recalcula a luz UV de N em N passos.
    pub light_interval: u32,
    /// Física dos grãos do terreno ligada.
    pub terrain_enabled: bool,
    /// Repulsão estérica entre agentes.
    pub contact_enabled: bool,
}

impl Default for WorldSettings {
    fn default() -> Self {
        // Jacobi 128 (o v3 usava 10): com 10 a pressão não convergia e o fluido
        // criava e destruía água (~50% de fluxo líquido através de uma linha).
        Self {
            fluid_enabled: true,
            fluid_substep: 2,
            multigrid: true,
            mg_cycles: 1,
            jacobi_iters: 128,
            light_interval: 100,
            terrain_enabled: true,
            contact_enabled: true,
        }
    }
}

struct Pipelines {
    transport: wgpu::ComputePipeline,
    commit: wgpu::ComputePipeline,
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
    slope: wgpu::ComputePipeline,
    relax_a: wgpu::ComputePipeline,
    relax_b: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    agents_step: wgpu::ComputePipeline,
    agents_ledger: wgpu::ComputePipeline,
    agents_birth: wgpu::ComputePipeline,
    draw_list: wgpu::ComputePipeline,
    mg_init: wgpu::ComputePipeline,
    mg_red: wgpu::ComputePipeline,
    mg_black: wgpu::ComputePipeline,
    mg_restrict: wgpu::ComputePipeline,
    mg_prolong: wgpu::ComputePipeline,
    mg_finish: wgpu::ComputePipeline,
    contact_clear: wgpu::ComputePipeline,
    contact_insert: wgpu::ComputePipeline,
    contact_resolve: wgpu::ComputePipeline,
    contact_apply: wgpu::ComputePipeline,
}

/// Suavizações por nível do multigrid (antes, depois) e no nível mais grosso.
const MG_PRE: u32 = 2;
const MG_POST: u32 = 2;
const MG_COARSE: u32 = 30;
/// Nível mais grosso do multigrid.
const MG_MIN_SIZE: u32 = 8;
/// Ancoragem do multigrid no nível fino.
const MG_EPS: f32 = 1e-4;

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
    pub slope_buf: wgpu::Buffer,
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
    life_bg: wgpu::BindGroup,
    mg_bg: wgpu::BindGroup,
    /// Tamanho de cada nível do multigrid (o offset dinâmico do nível l é l·256).
    mg_sizes: Vec<u32>,
    pub agents_buf: wgpu::Buffer,
    pub genomes_buf: wgpu::Buffer,
    pub bodies_buf: wgpu::Buffer,
    pub body_pos_buf: wgpu::Buffer,
    pub draw_list_buf: wgpu::Buffer,
    pub joint_state_buf: wgpu::Buffer,
    pub draw_args_buf: wgpu::Buffer,
    pub life_counters_buf: wgpu::Buffer,
    free_buf: wgpu::Buffer,
    spawn_buf: wgpu::Buffer,
    /// Pedidos de sementes à espera do próximo `encode_steps`.
    pending_spawns: Vec<SpawnRequest>,
    pipelines: Pipelines,
}

/// Contadores do ciclo de vida (life_counters na GPU).
#[derive(Clone, Copy, Debug, Default)]
pub struct LifeCounters {
    pub free_top: u32,
    pub next_id: u32,
    pub spawned: u32,
    pub spawn_failed: u32,
    pub deaths: u32,
    pub births: u32,
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
        let chem_next = storage_buffer(device, "chem next", cells * 16);
        let ledger_buf = storage_buffer(device, "ledger", LEDGER_WORDS * 4);
        let gamma_buf = storage_buffer(device, "gamma grid", cells * 4);
        let light_buf = storage_buffer(device, "uv light", cells * 4);
        let slope_buf = storage_buffer(device, "gamma slope", cells * 8);
        let ledger_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ledger staging"),
            size: LEDGER_WORDS * 4,
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
        // Organismos: slots fixos; todos começam livres.
        let max_agents = cfg.max_agents as u64;
        let agents_buf = storage_buffer(device, "agents", max_agents * size_of::<Agent>() as u64);
        let genomes_buf = storage_buffer(device, "genomes", max_agents * SLOT_WORDS * 4);
        let bodies_buf = storage_buffer(device, "bodies", max_agents * SLOT_WORDS * 4);
        let body_pos_buf = storage_buffer(device, "body positions", max_agents * 64 * 8);
        let draw_list_buf = storage_buffer(device, "draw list", max_agents * 4);
        let contact_head = storage_buffer(device, "contact head", cells * 4);
        let contact_next = storage_buffer(device, "contact next", max_agents * 4);
        let contact_disp = storage_buffer(device, "contact disp", max_agents * 16);
        let joint_angle = storage_buffer(device, "joint angle", max_agents * 64 * 4);
        let joint_base = storage_buffer(device, "joint base", max_agents * 64 * 4);
        let joint_state_buf = storage_buffer(device, "joint state", max_agents * 64 * 4);
        let joint_active = storage_buffer(device, "joint active", max_agents * 64 * 4);
        let draw_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let free_slots: Vec<u32> = (0..cfg.max_agents).rev().collect();
        let free_buf = storage_buffer(device, "free slots", max_agents * 4);
        gpu.queue.write_buffer(&free_buf, 0, bytemuck::cast_slice(&free_slots));
        let life_counters_buf = storage_buffer(device, "life counters", 8 * 4);
        gpu.queue.write_buffer(&life_counters_buf, 0, bytemuck::cast_slice(&[cfg.max_agents, 0, 0, 0, 0, 0, 0, 0u32]));
        let spawn_buf =
            storage_buffer(device, "spawn requests", (MAX_SPAWN_REQUESTS * size_of::<SpawnRequest>()) as u64);

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
            entries: &(0..6).map(|b| storage_entry(b, false)).collect::<Vec<_>>(),
        });
        // Grupo 2 — fluido. Bindings 0 e 2 (velocity_in, pressure_in) só de leitura.
        let fluid_entries: Vec<_> = (0..10).map(|b| storage_entry(b, matches!(b, 0 | 2 | 9))).collect();
        let fluid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid layout"),
            entries: &fluid_entries,
        });
        // Grupo 3 — organismos. Binding 4 (pedidos de sementes) só de leitura.
        let life_entries: Vec<_> = (0..16).map(|b| storage_entry(b, b == 4)).collect();
        let life_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("life layout"),
            entries: &life_entries,
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
        let world_bg = bind_all(
            "world bg",
            &world_layout,
            &[&chem_buf, &ledger_buf, &gamma_buf, &light_buf, &slope_buf, &chem_next],
        );
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

        // Grupo 4 — multigrid da pressão: um uniforme por nível (offset
        // dinâmico) e os buffers de todos os níveis seguidos.
        let mut mg_sizes = vec![cfg.fluid_size];
        while *mg_sizes.last().unwrap() > MG_MIN_SIZE && mg_sizes.last().unwrap() % 2 == 0 {
            let n = mg_sizes.last().unwrap() / 2;
            mg_sizes.push(n);
        }
        let mut mg_offsets = vec![0u32];
        for n in &mg_sizes {
            let last = *mg_offsets.last().unwrap();
            mg_offsets.push(last + n * n);
        }
        let mg_total = *mg_offsets.last().unwrap() as u64;
        let mut level_bytes = vec![0u8; PARAMS_STRIDE as usize * mg_sizes.len()];
        for (l, &n) in mg_sizes.iter().enumerate() {
            let lv = MgLevel {
                n,
                off: mg_offsets[l],
                n_c: mg_sizes.get(l + 1).copied().unwrap_or(0),
                off_c: mg_offsets[l + 1],
                // Mata o modo constante (Neumann puro é singular) sem mexer no
                // gradiente; o operador de Galerkin escala ×2 por nível.
                eps: MG_EPS * 2f32.powi(l as i32),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let at = l * PARAMS_STRIDE as usize;
            level_bytes[at..at + size_of::<MgLevel>()].copy_from_slice(bytemuck::bytes_of(&lv));
        }
        let mg_level_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mg levels"),
            size: level_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&mg_level_buf, 0, &level_bytes);
        let mg_p = storage_buffer(device, "mg p", mg_total * 4);
        let mg_rhs = storage_buffer(device, "mg rhs", mg_total * 4);
        let mg_fluid = storage_buffer(device, "mg fluid", mg_total * 4);
        let mg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mg layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<MgLevel>() as u64),
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let mg_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mg bg"),
            layout: &mg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &mg_level_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<MgLevel>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: mg_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mg_rhs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mg_fluid.as_entire_binding() },
            ],
        });

        let life_bg = bind_all(
            "life bg",
            &life_layout,
            &[
                &agents_buf,
                &genomes_buf,
                &free_buf,
                &life_counters_buf,
                &spawn_buf,
                &bodies_buf,
                &body_pos_buf,
                &draw_list_buf,
                &draw_args_buf,
                &contact_head,
                &contact_next,
                &contact_disp,
                &joint_angle,
                &joint_base,
                &joint_state_buf,
                &joint_active,
            ],
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world pipeline layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&world_layout), Some(&fluid_layout), Some(&life_layout)],
            immediate_size: 0,
        });
        let mg_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mg pipeline layout"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&fluid_layout),
                Some(&life_layout),
                Some(&mg_layout),
            ],
            immediate_size: 0,
        });
        let module = shaders::create(device, &shaders::WORLD, &cfg);
        let mg_compute = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&mg_pl_layout),
                module: &module,
                entry_point: Some(shaders::entry(&shaders::WORLD, entry)),
                compilation_options: Default::default(),
                cache: None,
            })
        };
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
            transport: compute("transport_scatter"),
            commit: compute("transport_commit"),
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
            slope: compute("compute_gamma_slope"),
            relax_a: compute("relax_gamma_a"),
            relax_b: compute("relax_gamma_b"),
            spawn: compute("spawn_seeds"),
            agents_step: compute("agents_step"),
            agents_ledger: compute("agents_ledger"),
            agents_birth: compute("agents_birth"),
            draw_list: compute("build_draw_list"),
            mg_init: mg_compute("mg_init"),
            mg_red: mg_compute("mg_smooth_red"),
            mg_black: mg_compute("mg_smooth_black"),
            mg_restrict: mg_compute("mg_restrict"),
            mg_prolong: mg_compute("mg_prolong"),
            mg_finish: mg_compute("mg_finish"),
            contact_clear: compute("contact_clear"),
            contact_insert: compute("contact_insert"),
            contact_resolve: compute("contact_resolve"),
            contact_apply: compute("contact_apply"),
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
            slope_buf,
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
            life_bg,
            mg_bg,
            mg_sizes,
            agents_buf,
            genomes_buf,
            bodies_buf,
            body_pos_buf,
            draw_list_buf,
            joint_state_buf,
            draw_args_buf,
            life_counters_buf,
            free_buf,
            spawn_buf,
            pending_spawns: Vec::new(),
            pipelines,
        }
    }

    /// Pede sementes (geração 0); são processadas no início do próximo `encode_steps`.
    pub fn request_seeds(&mut self, reqs: &[SpawnRequest]) {
        let room = MAX_SPAWN_REQUESTS - self.pending_spawns.len();
        self.pending_spawns.extend_from_slice(&reqs[..reqs.len().min(room)]);
    }

    /// Lê os contadores do ciclo de vida de forma síncrona (testes, depuração).
    pub fn life_counters_blocking(&self, gpu: &Gpu) -> LifeCounters {
        let w: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.life_counters_buf)).to_vec();
        LifeCounters { free_top: w[0], next_id: w[1], spawned: w[2], spawn_failed: w[3], deaths: w[4], births: w[5] }
    }

    /// Lê os agentes (todos os slots) de forma síncrona (testes).
    pub fn read_agents_blocking(&self, gpu: &Gpu) -> Vec<Agent> {
        bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.agents_buf)).to_vec()
    }

    /// Gera o terreno e a matéria inicial (determinista para a semente).
    /// As células com gamma ficam sem monómeros. Devolve a contagem exata
    /// dos monómeros escritos.
    pub fn seed_matter(&mut self, gpu: &Gpu, seed: u64) -> Ledger {
        let gamma = terrain::generate(&self.cfg, seed as u32, &self.fumaroles);
        let mut cells = seed_cells(&self.cfg, seed);
        for (i, &g) in gamma.iter().enumerate() {
            if g > 0 {
                cells[i * 4..i * 4 + 4].fill(0);
            }
        }
        gpu.queue.write_buffer(&self.gamma_buf, 0, bytemuck::cast_slice(&gamma));
        gpu.queue.write_buffer(&self.chem_buf, 0, bytemuck::cast_slice(&cells));
        // Mundo novo: não há agentes (a matéria deles pertencia ao mundo antigo).
        let max = self.cfg.max_agents;
        gpu.queue.write_buffer(&self.agents_buf, 0, &vec![0u8; max as usize * size_of::<Agent>()]);
        let free_slots: Vec<u32> = (0..max).rev().collect();
        gpu.queue.write_buffer(&self.free_buf, 0, bytemuck::cast_slice(&free_slots));
        gpu.queue.write_buffer(&self.life_counters_buf, 0, bytemuck::cast_slice(&[max, 0, 0, 0, 0, 0, 0, 0u32]));
        self.pending_spawns.clear();
        self.light_dirty = true;
        Ledger::from_cells(&cells)
    }

    /// Lê o terreno inteiro de forma síncrona (testes).
    pub fn read_gamma_blocking(&self, gpu: &Gpu) -> Vec<u32> {
        bytemuck::cast_slice(&gpu.read_buffer_blocking(&self.gamma_buf)).to_vec()
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
        self.params.fluid_enabled = st.fluid_enabled as u32;
        self.params.max_agents = self.cfg.max_agents;
        let spawns = std::mem::take(&mut self.pending_spawns);
        if !spawns.is_empty() {
            queue.write_buffer(&self.spawn_buf, 0, bytemuck::cast_slice(&spawns));
        }

        let mut bytes = vec![0u8; (PARAMS_STRIDE * steps as u64) as usize];
        for i in 0..steps {
            // As sementes só entram no primeiro passo do lote.
            let spawn_count = if i == 0 { spawns.len() as u32 } else { 0 };
            let p = SimParams { epoch: self.params.epoch.wrapping_add(i), spawn_count, ..self.params };
            let at = (PARAMS_STRIDE * i as u64) as usize;
            bytes[at..at + size_of::<SimParams>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        queue.write_buffer(&self.params_buf, 0, &bytes);

        let g = groups(self.cfg.grid_size, 16);
        let fg = groups(self.cfg.fluid_size, 16);
        let jacobi_iters = (st.jacobi_iters.clamp(2, 128) + 1) & !1;
        // transport_commit: uma thread por u32 da grelha; o shader assume
        // linhas de 65535 workgroups quando é preciso uma segunda dimensão.
        let commit_wg = (self.cfg.cells() * 4).div_ceil(256) as u32;
        let commit_groups = [commit_wg.min(65535), commit_wg.div_ceil(65535)];
        // Grelha de contacto (células de 120 unidades; igual a CONTACT_N no shader).
        let contact_n = (self.cfg.sim_size() / 120.0) as u32 + 1;
        let contact_cells = contact_n * contact_n;
        let pl = &self.pipelines;
        let (ab, ba) = (&self.fluid_ab, &self.fluid_ba);
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("world"), timestamp_writes: None });
        pass.set_bind_group(1, &self.world_bg, &[]);
        pass.set_bind_group(3, &self.life_bg, &[]);
        let ag = groups(self.cfg.max_agents, 64);
        let run = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, n: [u32; 2]| {
            pass.set_bind_group(2, bg, &[]);
            pass.set_pipeline(p);
            pass.dispatch_workgroups(n[0], n[1], 1);
        };
        for i in 0..steps {
            let epoch = self.params.epoch.wrapping_add(i);
            pass.set_bind_group(0, &self.frame_bg, &[(PARAMS_STRIDE * i as u64) as u32]);

            // SEMENTES (só no primeiro passo, e nunca entre scatter e commit).
            if i == 0 && !spawns.is_empty() {
                run(&mut pass, &pl.spawn, ab, [groups(spawns.len() as u32, 64), 1]);
            }

            // TERRENO: duas passagens de relaxação dos grãos e o declive.
            if st.terrain_enabled {
                run(&mut pass, &pl.relax_a, ab, [g, g]);
                run(&mut pass, &pl.relax_b, ab, [g, g]);
            }
            run(&mut pass, &pl.slope, ab, [g, g]);

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
                // Pressão em ARRANQUE QUENTE (nunca é limpa); o resultado fica
                // sempre em pressure_a.
                if st.multigrid {
                    let mg = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, bg, l: usize, n: u32| {
                        pass.set_bind_group(2, bg, &[]);
                        pass.set_bind_group(4, &self.mg_bg, &[(l as u64 * PARAMS_STRIDE) as u32]);
                        pass.set_pipeline(p);
                        let w = groups(n, 16);
                        pass.dispatch_workgroups(w, w, 1);
                    };
                    let levels = self.mg_sizes.len();
                    mg(&mut pass, &pl.mg_init, ab, 0, self.mg_sizes[0]);
                    for _ in 0..st.mg_cycles.max(1) {
                        for l in 0..levels - 1 {
                            for _ in 0..MG_PRE {
                                mg(&mut pass, &pl.mg_red, ab, l, self.mg_sizes[l]);
                                mg(&mut pass, &pl.mg_black, ab, l, self.mg_sizes[l]);
                            }
                            mg(&mut pass, &pl.mg_restrict, ab, l, self.mg_sizes[l + 1]);
                        }
                        for _ in 0..MG_COARSE {
                            mg(&mut pass, &pl.mg_red, ab, levels - 1, self.mg_sizes[levels - 1]);
                            mg(&mut pass, &pl.mg_black, ab, levels - 1, self.mg_sizes[levels - 1]);
                        }
                        for l in (0..levels - 1).rev() {
                            mg(&mut pass, &pl.mg_prolong, ab, l, self.mg_sizes[l]);
                            for _ in 0..MG_POST {
                                mg(&mut pass, &pl.mg_red, ab, l, self.mg_sizes[l]);
                                mg(&mut pass, &pl.mg_black, ab, l, self.mg_sizes[l]);
                            }
                        }
                    }
                    mg(&mut pass, &pl.mg_finish, ba, 0, self.mg_sizes[0]);
                } else {
                    for k in 0..jacobi_iters {
                        run(&mut pass, &pl.jacobi, if k % 2 == 0 { ab } else { ba }, f);
                    }
                }
                run(&mut pass, &pl.subtract_gradient, ab, f); // a -> b
                run(&mut pass, &pl.boundaries, ba, f); // b -> a (final em a)
                run(&mut pass, &pl.thermal_activation, ab, [g, g]);
            }

            // TRANSPORTE em duas fases (reprodutível): espalhar para chem_next
            // a partir do estado antes do passo, depois copiar de volta.
            run(&mut pass, &pl.transport, ab, [g, g]);
            run(&mut pass, &pl.commit, ab, commit_groups);

            // ORGANISMOS: depois do commit (os depósitos da morte vão para chem_grid).
            run(&mut pass, &pl.agents_step, ab, [ag, 1]);
            // CONTACTO: grelha de agentes, empurrões, aplicação.
            if st.contact_enabled {
                run(&mut pass, &pl.contact_clear, ab, [contact_cells.div_ceil(256), 1]);
                run(&mut pass, &pl.contact_insert, ab, [ag, 1]);
                run(&mut pass, &pl.contact_resolve, ab, [ag, 1]);
                run(&mut pass, &pl.contact_apply, ab, [ag, 1]);
            }
            // Nascimentos num passe à parte: a morte devolve slots (push) e o
            // nascimento tira-os (pop); nunca no mesmo despacho.
            run(&mut pass, &pl.agents_birth, ab, [ag, 1]);
        }
        drop(pass);
        self.params.epoch = self.params.epoch.wrapping_add(steps);
    }

    /// Grava a lista dos agentes vivos e os argumentos do draw indireto.
    pub fn encode_draw_list(&self, enc: &mut wgpu::CommandEncoder) {
        enc.clear_buffer(&self.draw_args_buf, 0, None);
        let mut pass =
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("draw list"), timestamp_writes: None });
        pass.set_bind_group(0, &self.frame_bg, &[0]);
        pass.set_bind_group(1, &self.world_bg, &[]);
        pass.set_bind_group(2, &self.fluid_ab, &[]);
        pass.set_bind_group(3, &self.life_bg, &[]);
        pass.set_pipeline(&self.pipelines.draw_list);
        pass.dispatch_workgroups(groups(self.cfg.max_agents, 64), 1, 1);
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
        pass.set_bind_group(3, &self.life_bg, &[]);
        pass.set_pipeline(&self.pipelines.ledger);
        pass.dispatch_workgroups(g, g, 1);
        pass.set_pipeline(&self.pipelines.agents_ledger);
        pass.dispatch_workgroups(groups(self.cfg.max_agents, 64), 1, 1);
    }

    /// Livro-razão assíncrono: se não houver leitura em curso, grava a redução
    /// e a cópia para o staging. Chamar `ledger_after_submit` depois do submit.
    pub fn encode_ledger_readback(&mut self, enc: &mut wgpu::CommandEncoder) {
        if !matches!(self.readback, Readback::Idle) {
            return;
        }
        self.encode_ledger(enc);
        enc.copy_buffer_to_buffer(&self.ledger_buf, 0, &self.ledger_staging, 0, LEDGER_WORDS * 4);
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
