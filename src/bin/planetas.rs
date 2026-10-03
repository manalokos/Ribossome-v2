//! PLANETAS: um pequeno simulador de acreção num disco 2D, sobre a mesma
//! ideia de "matéria em quanta numa grelha" do Ribossome, mas COM inércia.
//!
//! Cada célula da grelha guarda um PACOTE: massa (quanta inteiros), velocidade
//! e centro de massa dentro da célula (0..1). Em cada passo:
//!   1. kick: a velocidade de cada pacote muda com a gravidade da estrela
//!      central (exata, 1/r²) e dos pacotes vizinhos num raio (1/r²,
//!      suavizada): a estrela domina; a formação de planetas vem dos
//!      encontros próximos;
//!   2. drift + recolha: cada pacote anda v·dt e cai INTEIRO na célula onde
//!      fica o seu centro de massa (sem se partir: não há difusão numérica);
//!      cada célula recolhe o que lhe cai das 9 vizinhas e junta: massa
//!      somada (exata), momento somado (conservado), centro de massa pela
//!      média pesada. Dois pacotes na mesma célula = colisão inelástica =
//!      ACREÇÃO.
//!
//! Nada anda mais de uma célula por passo (CFL), por isso basta olhar para as
//! 9 vizinhas. Unidades: células e passos (dt = 1).

use std::sync::Arc;

use ribossome::gpu::Gpu;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const N: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    gm_star: f32,
    g_quantum: f32,
    soft: f32,
    star_soft: f32,
    center_x: f32,
    center_y: f32,
    radius: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewParams {
    center_x: f32,
    center_y: f32,
    scale: f32,
    aspect: f32,
    max_log: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

// VMAX: velocidade máxima (células por passo): o pacote nunca salta mais de uma.
const COMMON: &str = r#"
struct SimParams { gm_star: f32, g_quantum: f32, soft: f32, star_soft: f32, center_x: f32, center_y: f32, radius: u32, n: u32 }
const VMAX: f32 = 0.95;
"#;

const SIM_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> mass_in: array<u32>;
@group(0) @binding(2) var<storage, read> dyn_in: array<vec4<f32>>;   // (vx, vy, cx, cy)
@group(0) @binding(3) var<storage, read_write> dyn_tmp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mass_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> dyn_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>, 8>;

// 1. KICK: gravidade da estrela + dos vizinhos no raio.
@compute @workgroup_size(16, 16)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let i = y * P.n + x;
    let m = mass_in[i];
    var d = dyn_in[i];
    if (m == 0u) { dyn_tmp[i] = vec4<f32>(0.0); return; }
    let pos = vec2<f32>(f32(x), f32(y)) + d.zw;
    let r = pos - vec2<f32>(P.center_x, P.center_y);
    let r2 = dot(r, r) + P.star_soft * P.star_soft;
    var a = -P.gm_star * r / (r2 * sqrt(r2));
    let R = i32(P.radius);
    for (var dy = -R; dy <= R; dy++) {
        for (var dx = -R; dx <= R; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(P.n) || ny >= i32(P.n)) { continue; }
            let j = u32(ny) * P.n + u32(nx);
            let mj = mass_in[j];
            if (mj == 0u) { continue; }
            let pj = vec2<f32>(f32(nx), f32(ny)) + dyn_in[j].zw;
            let dd = pj - pos;
            let s2 = dot(dd, dd) + P.soft * P.soft;
            a += P.g_quantum * f32(mj) * dd / (s2 * sqrt(s2));
        }
    }
    var v = d.xy + a;
    let sp = length(v);
    if (sp > VMAX) { v *= VMAX / sp; }
    dyn_tmp[i] = vec4<f32>(v, d.zw);
}

// Onde cai o pacote da célula s (e com que velocidade): contra a parede do
// mundo fica e a velocidade reflete-se.
fn landing(s: vec2<i32>, d: vec4<f32>) -> vec4<f32> {
    var v = d.xy;
    var p = vec2<f32>(s) + d.zw + v;
    if (p.x < 0.0 || p.x >= f32(P.n)) { v.x = -v.x; p.x = f32(s.x) + d.z; }
    if (p.y < 0.0 || p.y >= f32(P.n)) { v.y = -v.y; p.y = f32(s.y) + d.w; }
    return vec4<f32>(p, v);
}

// 2. DRIFT + RECOLHA: cada célula junta os pacotes que lhe caem.
@compute @workgroup_size(16, 16)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let here = vec2<i32>(i32(x), i32(y));
    var mt = 0u;
    var mom = vec2<f32>(0.0);
    var com = vec2<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = here + vec2<i32>(dx, dy);
            if (s.x < 0 || s.y < 0 || s.x >= i32(P.n) || s.y >= i32(P.n)) { continue; }
            let j = u32(s.y) * P.n + u32(s.x);
            let ms = mass_in[j];
            if (ms == 0u) { continue; }
            let l = landing(s, dyn_tmp[j]);
            if (any(vec2<i32>(floor(l.xy)) != here)) { continue; }
            mt += ms;
            mom += f32(ms) * l.zw;
            com += f32(ms) * (l.xy - vec2<f32>(here));
        }
    }
    let i = y * P.n + x;
    mass_out[i] = mt;
    if (mt == 0u) {
        dyn_out[i] = vec4<f32>(0.0);
    } else {
        let c = clamp(com / f32(mt), vec2<f32>(0.0), vec2<f32>(0.99999));
        dyn_out[i] = vec4<f32>(mom / f32(mt), c);
    }
}

// Estatísticas: massa total, corpos, maior massa, momento (ponto fixo).
@compute @workgroup_size(16, 16)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let i = y * P.n + x;
    let m = mass_in[i];
    if (m == 0u) { return; }
    atomicAdd(&stats[0], m);
    atomicAdd(&stats[1], 1u);
    atomicMax(&stats[2], m);
    // Momento angular em torno da estrela (L = r × m v), ponto fixo com sinal.
    let d = dyn_in[i];
    let r = vec2<f32>(f32(x), f32(y)) + d.zw - vec2<f32>(P.center_x, P.center_y);
    let l = f32(m) * (r.x * d.y - r.y * d.x);
    atomicAdd(&stats[3], bitcast<u32>(i32(round(l))));
    if (m >= 30u) { atomicAdd(&stats[4], 1u); }
}
"#;

const VIEW_WGSL: &str = r#"
struct ViewParams { center_x: f32, center_y: f32, scale: f32, aspect: f32, max_log: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> V: ViewParams;
@group(0) @binding(1) var<storage, read> mass_v: array<u32>;
@group(0) @binding(2) var<uniform> P: SimParams;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VsOut;
    o.pos = vec4<f32>(c[vi], 0.0, 1.0);
    o.uv = c[vi];
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Pixel -> células (y para cima).
    let w = vec2<f32>(V.center_x + in.uv.x * V.scale * V.aspect, V.center_y + in.uv.y * V.scale);
    let star = length(w - vec2<f32>(P.center_x, P.center_y));
    var col = vec3<f32>(0.0);
    if (w.x >= 0.0 && w.y >= 0.0 && w.x < f32(P.n) && w.y < f32(P.n)) {
        let m = f32(mass_v[u32(w.y) * P.n + u32(w.x)]);
        if (m > 0.0) {
            let t = clamp(log2(1.0 + m) / V.max_log, 0.0, 1.0);
            // Poeira acastanhada -> planetesimais -> planetas brancos.
            col = mix(vec3<f32>(0.35, 0.22, 0.12), vec3<f32>(1.0, 0.85, 0.55), sqrt(t));
            col = mix(col, vec3<f32>(1.0), smoothstep(0.7, 1.0, t));
        }
    } else {
        col = vec3<f32>(0.03);
    }
    // A estrela.
    col += vec3<f32>(1.0, 0.8, 0.4) * exp(-star * star / 30.0);
    return vec4<f32>(col, 1.0);
}
"#;

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Disco inicial: anel de planetesimais em órbita quase circular.
struct DiscSettings {
    r_in: f32,
    r_out: f32,
    fill: f32,
    dispersion: f32,
    seed: u64,
}

fn make_disc(s: &DiscSettings, gm: f32) -> (Vec<u32>, Vec<[f32; 4]>) {
    let n = N as usize;
    let c = N as f32 * 0.5;
    let mut rng = SplitMix(s.seed);
    let mut mass = vec![0u32; n * n];
    let mut dynv = vec![[0f32; 4]; n * n];
    for y in 0..n {
        for x in 0..n {
            let (px, py) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
            let r = (px * px + py * py).sqrt();
            if r < s.r_in || r > s.r_out || rng.f32() > s.fill {
                continue;
            }
            let vc = (gm / r).sqrt();
            let (tx, ty) = (-py / r, px / r);
            let jx = (rng.f32() - 0.5) * 2.0 * s.dispersion * vc;
            let jy = (rng.f32() - 0.5) * 2.0 * s.dispersion * vc;
            let i = y * n + x;
            mass[i] = 1 + (rng.f32() * 3.0) as u32;
            dynv[i] = [tx * vc + jx, ty * vc + jy, rng.f32() * 0.99, rng.f32() * 0.99];
        }
    }
    (mass, dynv)
}

struct Sim {
    params: SimParams,
    params_buf: wgpu::Buffer,
    mass: [wgpu::Buffer; 2],
    dynb: [wgpu::Buffer; 2],
    stats: wgpu::Buffer,
    bg: [wgpu::BindGroup; 2],
    kick: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    cur: usize,
    step: u64,
}

impl Sim {
    fn new(gpu: &Gpu, params: SimParams) -> Self {
        let device = &gpu.device;
        let cells = (N * N) as u64;
        let mass = [storage(device, "mass a", cells * 4), storage(device, "mass b", cells * 4)];
        let dynb = [storage(device, "dyn a", cells * 16), storage(device, "dyn b", cells * 16)];
        let tmp = storage(device, "dyn tmp", cells * 16);
        let stats = storage(device, "stats", 32);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planetas sim"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{SIM_WGSL}").into()),
        });
        let st = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planetas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                st(1, true),
                st(2, true),
                st(3, false),
                st(4, false),
                st(5, false),
                st(6, false),
            ],
        });
        let mk = |a: usize, b: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("planetas"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mass[a].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dynb[a].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: tmp.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: mass[b].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: dynb[b].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: stats.as_entire_binding() },
                ],
            })
        };
        let bg = [mk(0, 1), mk(1, 0)];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planetas"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            params,
            params_buf,
            kick: compute("kick"),
            gather: compute("gather"),
            reduce: compute("reduce"),
            mass,
            dynb,
            stats,
            bg,
            cur: 0,
            step: 0,
        }
    }

    fn upload(&mut self, gpu: &Gpu, mass: &[u32], dynv: &[[f32; 4]]) {
        gpu.queue.write_buffer(&self.mass[0], 0, bytemuck::cast_slice(mass));
        gpu.queue.write_buffer(&self.dynb[0], 0, bytemuck::cast_slice(dynv));
        self.cur = 0;
        self.step = 0;
    }

    fn encode(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, steps: u32) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let g = N.div_ceil(16);
        let mut pass = enc.begin_compute_pass(&Default::default());
        for _ in 0..steps {
            pass.set_bind_group(0, &self.bg[self.cur], &[]);
            pass.set_pipeline(&self.kick);
            pass.dispatch_workgroups(g, g, 1);
            pass.set_pipeline(&self.gather);
            pass.dispatch_workgroups(g, g, 1);
            self.cur ^= 1;
            self.step += 1;
        }
    }

    /// (massa total, corpos, maior massa, momento angular, corpos >= 30).
    fn read_stats(&self, gpu: &Gpu) -> [u32; 8] {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.stats, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.bg[self.cur], &[]);
            pass.set_pipeline(&self.reduce);
            let g = N.div_ceil(16);
            pass.dispatch_workgroups(g, g, 1);
        }
        gpu.queue.submit([enc.finish()]);
        let raw = gpu.read_buffer_blocking(&self.stats);
        let w: &[u32] = bytemuck::cast_slice(&raw);
        let mut out = [0u32; 8];
        out.copy_from_slice(&w[..8]);
        out
    }
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    sim: Sim,
    disc: DiscSettings,
    view_buf: wgpu::Buffer,
    view_bg: [wgpu::BindGroup; 2],
    view_pipeline: wgpu::RenderPipeline,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    center: [f32; 2],
    scale: f32,
    cursor: [f32; 2],
    dragging: bool,
    paused: bool,
    steps_per_frame: u32,
    stats: [u32; 8],
    stats_l0: Option<i32>,
    frame: u64,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Planetas")
                        .with_inner_size(winit::dpi::LogicalSize::new(1100, 1000)),
                )
                .expect("criar janela"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface))).expect("GPU");
        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&gpu.device, &surface_cfg);

        // Unidades: com GM = 25, a velocidade circular a 100 células é 0,5
        // células/passo (dentro da CFL) e a 450 é 0,24.
        let params = SimParams {
            gm_star: 25.0,
            g_quantum: 3.0e-6,
            soft: 0.7,
            star_soft: 20.0,
            center_x: N as f32 * 0.5,
            center_y: N as f32 * 0.5,
            radius: 6,
            n: N,
        };
        let disc = DiscSettings { r_in: 110.0, r_out: 460.0, fill: 0.35, dispersion: 0.02, seed: 1 };
        let mut sim = Sim::new(&gpu, params);
        let (m, d) = make_disc(&disc, params.gm_star);
        sim.upload(&gpu, &m, &d);

        let device = &gpu.device;
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<ViewParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planetas view"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{VIEW_WGSL}").into()),
        });
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                uni(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                uni(2),
            ],
        });
        let mk = |i: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("view"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sim.mass[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sim.params_buf.as_entire_binding() },
                ],
            })
        };
        let view_bg = [mk(0), mk(1)];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let view_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("view"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        Self {
            window,
            surface,
            surface_cfg,
            gpu,
            sim,
            disc,
            view_buf,
            view_bg,
            view_pipeline,
            egui_state,
            egui_renderer,
            center: [N as f32 * 0.5, N as f32 * 0.5],
            scale: N as f32 * 0.52,
            cursor: [0.0; 2],
            dragging: false,
            paused: false,
            steps_per_frame: 8,
            stats: [0; 8],
            stats_l0: None,
            frame: 0,
        }
    }

    fn reset(&mut self) {
        let (m, d) = make_disc(&self.disc, self.sim.params.gm_star);
        self.sim.upload(&self.gpu, &m, &d);
        self.stats_l0 = None;
    }

    fn redraw(&mut self) {
        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.gpu.device, &self.surface_cfg);
                return;
            }
        };
        let target = tex.texture.create_view(&Default::default());
        if self.frame.is_multiple_of(20) {
            self.stats = self.sim.read_stats(&self.gpu);
            if self.stats_l0.is_none() {
                self.stats_l0 = Some(self.stats[3] as i32);
            }
        }
        self.frame += 1;

        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let mut reset = false;
        let st = self.stats;
        let l0 = self.stats_l0.unwrap_or(0);
        let step = self.sim.step;
        let out = ctx.run_ui(raw, |root| {
            egui::Window::new("Planetas").default_pos([10.0, 10.0]).show(root.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button(if self.paused { "▶ continuar" } else { "⏸ pausa" }).clicked() {
                        self.paused = !self.paused;
                    }
                    if ui.button("disco novo").clicked() {
                        reset = true;
                    }
                });
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=64).text("passos/frame"));
                ui.add(
                    egui::Slider::new(&mut self.sim.params.g_quantum, 0.0..=0.01)
                        .logarithmic(true)
                        .smallest_positive(1e-6)
                        .text("gravidade entre corpos"),
                );
                ui.add(egui::Slider::new(&mut self.sim.params.radius, 1..=12).text("raio da gravidade (células)"));
                ui.separator();
                ui.label("disco novo:");
                ui.add(egui::Slider::new(&mut self.disc.fill, 0.02..=1.0).text("densidade"));
                ui.add(egui::Slider::new(&mut self.disc.dispersion, 0.0..=0.2).text("agitação"));
                ui.add(egui::Slider::new(&mut self.disc.r_in, 40.0..=400.0).text("raio interior"));
                ui.add(egui::Slider::new(&mut self.disc.r_out, 60.0..=500.0).text("raio exterior"));
                ui.separator();
                ui.label(format!("passo {step}"));
                ui.label(format!("massa total {} quanta (conserva-se exatamente)", st[0]));
                ui.label(format!("corpos {}  maior {} quanta  com ≥ 30: {}", st[1], st[2], st[4]));
                let l = st[3] as i32;
                ui.label(format!(
                    "momento angular {} ({:+.3}% desde o início)",
                    l,
                    if l0 != 0 { 100.0 * (l - l0) as f64 / l0 as f64 } else { 0.0 }
                ));
                ui.label("roda do rato: zoom · arrastar: mover");
            });
        });
        if reset {
            self.disc.seed += 1;
            self.reset();
        }
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height],
            pixels_per_point: out.pixels_per_point,
        };
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
        }

        let aspect = self.surface_cfg.width as f32 / self.surface_cfg.height.max(1) as f32;
        let view = ViewParams {
            center_x: self.center[0],
            center_y: self.center[1],
            scale: self.scale,
            aspect,
            max_log: (1.0 + st[2].max(4) as f32).log2(),
            _p0: 0.0,
            _p1: 0.0,
            _p2: 0.0,
        };
        self.gpu.queue.write_buffer(&self.view_buf, 0, bytemuck::bytes_of(&view));
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if !self.paused {
            self.sim.encode(&self.gpu.queue, &mut enc, self.steps_per_frame);
        }
        let mut extra = self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &jobs, &sd);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            pass.set_pipeline(&self.view_pipeline);
            pass.set_bind_group(0, &self.view_bg[self.sim.cur], &[]);
            pass.draw(0..3, 0..1);
            self.egui_renderer.render(&mut pass, &jobs, &sd);
        }
        extra.push(enc.finish());
        self.gpu.queue.submit(extra);
        self.window.pre_present_notify();
        self.gpu.queue.present(tex);
        for id in &out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    fn screen_to_world(&self, p: [f32; 2]) -> [f32; 2] {
        let (w, h) = (self.surface_cfg.width as f32, self.surface_cfg.height as f32);
        let ux = p[0] / w * 2.0 - 1.0;
        let uy = 1.0 - p[1] / h * 2.0;
        [self.center[0] + ux * self.scale * (w / h), self.center[1] + uy * self.scale]
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let resp = self.egui_state.on_window_event(&self.window, &event);
        let ctx = self.egui_state.egui_ctx().clone();
        let over_ui = ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                self.surface_cfg.width = s.width.max(1);
                self.surface_cfg.height = s.height.max(1);
                self.surface.configure(&self.gpu.device, &self.surface_cfg);
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                if self.dragging {
                    let h = self.surface_cfg.height as f32;
                    let k = 2.0 * self.scale / h;
                    self.center[0] -= (p[0] - self.cursor[0]) * k;
                    self.center[1] += (p[1] - self.cursor[1]) * k;
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left | MouseButton::Right, .. } => {
                self.dragging = state == ElementState::Pressed && !over_ui;
            }
            WindowEvent::MouseWheel { delta, .. } if !over_ui && !resp.consumed => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let before = self.screen_to_world(self.cursor);
                self.scale = (self.scale * 0.87f32.powf(lines)).clamp(8.0, N as f32);
                let after = self.screen_to_world(self.cursor);
                self.center[0] += before[0] - after[0];
                self.center[1] += before[1] - after[1];
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct App {
    run: Option<Running>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.run.is_none() {
            self.run = Some(Running::new(event_loop));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(r) = self.run.as_mut() {
            r.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = self.run.as_ref() {
            r.window.request_redraw();
        }
    }
}

/// Sem janela: `planetas --teste [passos]` corre o disco e mostra a evolução.
fn headless(steps: u64) {
    let gpu = Gpu::new_headless().expect("GPU");
    let params = SimParams {
        gm_star: 25.0,
        g_quantum: 3.0e-6,
        soft: 0.7,
        star_soft: 20.0,
        center_x: N as f32 * 0.5,
        center_y: N as f32 * 0.5,
        radius: 6,
        n: N,
    };
    let disc = DiscSettings { r_in: 110.0, r_out: 460.0, fill: 0.35, dispersion: 0.02, seed: 1 };
    let mut sim = Sim::new(&gpu, params);
    let (m, d) = make_disc(&disc, params.gm_star);
    sim.upload(&gpu, &m, &d);
    let l0 = sim.read_stats(&gpu)[3] as i32;
    let t = std::time::Instant::now();
    let report = 10;
    for k in 0..=report {
        let target = steps * k / report;
        while sim.step < target {
            let n = (target - sim.step).min(64) as u32;
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            sim.encode(&gpu.queue, &mut enc, n);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
        }
        let st = sim.read_stats(&gpu);
        let l = st[3] as i32;
        println!(
            "passo {:7}: massa {} corpos {:7} maior {:6} (≥30: {:5})  L {:+.3}%",
            sim.step,
            st[0],
            st[1],
            st[2],
            st[4],
            100.0 * (l - l0) as f64 / l0 as f64
        );
    }
    println!("{:.1} ms por passo", t.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,planetas=info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|a| a == "--teste").unwrap_or(false) {
        headless(args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000));
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app");
}
