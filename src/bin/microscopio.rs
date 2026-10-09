//! MICROSCÓPIO 3D (protótipo, modo fotográfico): uma zona de uma cena vista
//! como num microscópio eletrónico de varrimento, com câmara em perspetiva
//! que se pode rodar e imagem ACUMULADA no tempo (cada frame lança raios um
//! pouco diferentes e a média limpa o ruído; o que se mexe fica arrastado).
//!
//! A simulação é 2D: a profundidade é só desenho. Em cada frame a zona é
//! desenhada de cima, com o desenho normal do programa, para duas texturas:
//! a COR (as peças como as vemos, com as suas sombras) e a ALTURA de cada
//! peça (um troço é um cilindro deitado, um órgão uma cúpula sobre o seu
//! contorno, a rocha um planalto, o entulho seixos). Depois cada píxel do
//! ecrã lança um raio contra esse relevo. O brilho é o do microscópio: sem
//! luzes, mais claro nas arestas (superfícies de lado para o observador) e
//! mais escuro nas zonas encaixadas entre vizinhos.
//!
//! Uso: `cargo run --release --bin microscopio` (ou microscopio.bat).
//!   SCENE   cena a abrir (por omissão o autosave)
//!   REGION  meio lado da zona, em unidades do mundo (por omissão 420)
//!   STEPS   passos de simulação por frame (por omissão 2; 0 = parada)
//!   --foto ficheiro.png [amostras]   sem janela: acumula e grava a imagem
//! Rato: arrastar com o botão esquerdo roda a câmara, com o direito desloca a
//! zona, a roda aproxima. Espaço pausa a simulação (a imagem converge);
//! F/G fecham e abrem o diafragma (profundidade de campo); Esc sai.

use std::sync::Arc;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::capture::Capture;
use ribossome::render::{Camera, WorldView, depth_attachment, depth_texture, msaa_texture};
use ribossome::world::{Scene, World};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Lado das texturas de cor e de altura da zona (píxeis).
const TEX: u32 = 2048;
const HEIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Altura máxima do relevo (unidades do mundo): o raio começa a marchar aqui.
const HMAX: f32 = 80.0;

const MARCH_WGSL: &str = r#"
struct U {
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    // centro da zona (x, y), meio lado, altura máxima
    region: vec4<f32>,
    // largura, altura do ecrã, número do frame, peso desta amostra
    screen: vec4<f32>,
    // distância de focagem, abertura (raio da lente), tan(meio campo), exagero da altura
    lens: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var height_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn hash3(p: vec3<u32>) -> vec3<f32> {
    var v = p * vec3<u32>(1664525u, 1013904223u, 2891336453u) + p.yzx * 747796405u;
    v = (v ^ (v >> vec3<u32>(16u))) * 2246822519u;
    v = (v ^ (v >> vec3<u32>(13u))) * 3266489917u;
    v = v ^ (v >> vec3<u32>(16u));
    return vec3<f32>(v) / 4294967295.0;
}

// Mundo (x, y) -> coordenadas da textura da zona (o mundo cresce para cima).
fn region_uv(xy: vec2<f32>) -> vec2<f32> {
    let d = (xy - u.region.xy) / (2.0 * u.region.z);
    return vec2<f32>(0.5 + d.x, 0.5 - d.y);
}

fn height_at(xy: vec2<f32>) -> f32 {
    let uv = region_uv(xy);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 0.0; }
    return min(textureSampleLevel(height_tex, samp, uv, 0.0).r * u.lens.w, u.region.w);
}

@fragment
fn fs_march(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let frame = u32(u.screen.z);
    let rnd = hash3(vec3<u32>(u32(pos.x), u32(pos.y), frame));
    let rnd2 = hash3(vec3<u32>(u32(pos.y) + 977u, u32(pos.x) + 131u, frame * 7u + 3u));
    // Raio do píxel, com um desvio dentro do píxel (suaviza as arestas)...
    let px = pos.xy + rnd.xy - 0.5;
    let ndc = vec2<f32>(px.x / u.screen.x * 2.0 - 1.0, 1.0 - px.y / u.screen.y * 2.0);
    let aspect = u.screen.x / u.screen.y;
    var dir = normalize(u.fwd.xyz + u.right.xyz * (ndc.x * u.lens.z * aspect) + u.up.xyz * (ndc.y * u.lens.z));
    var org = u.eye.xyz;
    // ...e a partir de um ponto ao acaso da lente (profundidade de campo: só
    // o plano de focagem fica nítido).
    if (u.lens.y > 0.0) {
        let focus = org + dir * (u.lens.x / max(dot(dir, u.fwd.xyz), 1e-3));
        let a = 6.2831853 * rnd2.x;
        let l = u.lens.y * sqrt(rnd2.y);
        org += (u.right.xyz * cos(a) + u.up.xyz * sin(a)) * l;
        dir = normalize(focus - org);
    }
    var col = vec3<f32>(0.0);
    if (dir.z < -1e-4) {
        // Do topo do relevo até ao fundo, em passos; ao passar para baixo da
        // superfície, afina por bisseção.
        let t0 = max((u.region.w - org.z) / dir.z, 0.0);
        let t1 = (0.0 - org.z) / dir.z;
        let steps = 200.0;
        let dt = (t1 - t0) / steps;
        var t = t0 + dt * rnd.z;
        var prev_t = t0;
        var hit = false;
        for (var i = 0; i < 200; i++) {
            let p = org + dir * t;
            if (p.z <= height_at(p.xy)) {
                hit = true;
                break;
            }
            prev_t = t;
            t += dt;
        }
        if (!hit) {
            // O fundo (altura zero).
            t = t1;
            prev_t = t1;
        }
        var lo = prev_t;
        var hi = t;
        for (var i = 0; i < 8; i++) {
            let m = 0.5 * (lo + hi);
            let p = org + dir * m;
            if (p.z <= height_at(p.xy)) { hi = m; } else { lo = m; }
        }
        let p = org + dir * hi;
        let uv = region_uv(p.xy);
        if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
            // Normal pelo declive do relevo.
            let e = 2.0 * u.region.z / f32(textureDimensions(height_tex).x) * 1.5;
            let hx = height_at(p.xy + vec2<f32>(e, 0.0)) - height_at(p.xy - vec2<f32>(e, 0.0));
            let hy = height_at(p.xy + vec2<f32>(0.0, e)) - height_at(p.xy - vec2<f32>(0.0, e));
            let n = normalize(vec3<f32>(-hx, -hy, 2.0 * e));
            // Numa parede (a borda de uma peça) a cor é a da peça, não a do
            // que está por baixo: lê-se um pouco para dentro, para o lado alto.
            let wall = clamp(1.0 - n.z, 0.0, 1.0);
            let inward = -n.xy / max(length(n.xy), 1e-4) * (2.5 * e * wall);
            let albedo = textureSampleLevel(color_tex, samp, region_uv(p.xy + inward), 0.0).rgb;
            let ndv = clamp(dot(n, -dir), 0.0, 1.0);
            // MICROSCÓPIO ELETRÓNICO: as superfícies de lado para o
            // observador soltam mais eletrões (arestas claras)...
            let edge = pow(1.0 - ndv, 2.0);
            // ...e os sítios encaixados entre vizinhos mais altos soltam
            // menos (oclusão): olha-se à volta, a duas distâncias.
            var occ = 0.0;
            let h0 = height_at(p.xy);
            for (var k = 0; k < 8; k++) {
                let a = 0.785398 * f32(k) + 6.2831853 * rnd2.z;
                let rr = select(9.0, 22.0, (k & 1) == 1);
                occ += clamp((height_at(p.xy + vec2<f32>(cos(a), sin(a)) * rr) - h0) / rr, 0.0, 1.5);
            }
            let ao = 1.0 / (1.0 + 1.6 * occ / 8.0 * 4.0);
            // (O fundo fica um cinzento muito escuro em vez de preto puro, como
            // o suporte de uma amostra.)
            let base = max(albedo, vec3<f32>(0.035, 0.036, 0.04));
            col = base * (0.95 + 1.5 * edge) * ao + vec3<f32>(0.25) * edge * edge * step(0.5, h0);
            // Esbate para o preto na borda da zona.
            let b = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            col *= smoothstep(0.0, 0.04, b);
        }
    }
    // Grão do detetor (desaparece com a acumulação).
    col += (rnd.z - 0.5) * 0.03;
    let prev = textureLoad(prev_tex, vec2<i32>(pos.xy), 0).rgb;
    return vec4<f32>(mix(prev, max(col, vec3<f32>(0.0)), u.screen.w), 1.0);
}

@fragment
fn fs_present(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(prev_tex, vec2<i32>(pos.xy), 0).rgb;
    // Curva suave nos claros, para as arestas não queimarem.
    return vec4<f32>(c / (1.0 + 0.25 * c), 1.0);
}
"#;

/// Câmara em órbita à volta do centro da zona.
#[derive(Clone, Copy, PartialEq)]
struct Orbit {
    centre: [f32; 2],
    yaw: f32,
    pitch: f32,
    dist: f32,
    aperture: f32,
}

struct Scope {
    world: World,
    cap: Capture,
    height_view: WorldView,
    height_msaa: wgpu::Texture,
    height_depth: wgpu::Texture,
    height_tex: wgpu::Texture,
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    march: wgpu::RenderPipeline,
    present: wgpu::RenderPipeline,
    accum: [wgpu::Texture; 2],
    size: [u32; 2],
    region: f32,
    orbit: Orbit,
    last_orbit: Option<Orbit>,
    samples: u32,
    frame: u32,
    steps: u32,
    paused: bool,
}

fn accum_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("accum"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ACCUM_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

impl Scope {
    fn new(gpu: &Gpu, target_format: wgpu::TextureFormat, size: [u32; 2]) -> Self {
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
        let scene = Scene::read(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let cfg = WorldConfig {
            grid_size: g("grid_size", 2048),
            fluid_size: g("fluid_size", 1024),
            world_units_per_cell: g("world_units_per_cell", 30),
            max_agents: g("max_agents", 400_000),
        };
        let mut world = World::new(gpu, cfg, 1);
        world.load_scene(gpu, &scene).unwrap_or_else(|e| panic!("{path}: {e}"));
        // Centro: CENTER=x,y (unidades do mundo) ou, por omissão, o agente com
        // mais tipos de órgãos diferentes (corpo de 14 a 48 resíduos); PICK=n
        // escolhe o n-ésimo dessa lista.
        let agents = world.read_agents_blocking(gpu);
        let alive: Vec<_> = agents.iter().filter(|a| a.alive != 0 && a.body_len >= 10).collect();
        let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&world.organs_buf));
        let mut best: Vec<(usize, usize)> = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.alive != 0 && (14..=48).contains(&a.body_len))
            .map(|(slot, a)| {
                let kinds: std::collections::HashSet<u16> = organs[slot * 64..slot * 64 + a.body_len as usize].iter().filter(|&&c| c != 0).map(|&c| c & 0x1F).collect();
                (kinds.len(), slot)
            })
            .collect();
        best.sort_by(|a, b| b.cmp(a));
        let pick = env("PICK", 0.0) as usize;
        let centre = std::env::var("CENTER")
            .ok()
            .and_then(|v| v.split_once(',').and_then(|(x, y)| Some([x.trim().parse().ok()?, y.trim().parse().ok()?])))
            .or_else(|| best.get(pick.min(best.len().saturating_sub(1))).map(|&(_, s)| [agents[s].pos_x, agents[s].pos_y]))
            .unwrap_or([cfg.sim_size() * 0.5; 2]);
        let region = env("REGION", 420.0);
        log::info!("{path}: {} agentes; zona de {} unidades à volta de ({:.0}, {:.0})", alive.len(), 2.0 * region, centre[0], centre[1]);

        let device = &gpu.device;
        let cap = Capture::new(gpu, &world, TEX);
        let height_view = WorldView::new(device, &gpu.queue, &world, HEIGHT_FORMAT);
        height_view.height_pass.set(true);
        let height_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("height"),
            size: wgpu::Extent3d { width: TEX, height: TEX, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEIGHT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scope"),
            size: 7 * 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scope layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scope"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("microscopio"), source: wgpu::ShaderSource::Wgsl(MARCH_WGSL.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scope"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |entry: &'static str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(entry), compilation_options: Default::default(), targets: &[Some(format.into())] }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            cap,
            height_view,
            height_msaa: msaa_texture(device, HEIGHT_FORMAT, TEX, TEX),
            height_depth: depth_texture(device, TEX, TEX),
            height_tex,
            uniform,
            march: pipeline("fs_march", ACCUM_FORMAT),
            present: pipeline("fs_present", target_format),
            layout,
            sampler,
            accum: [accum_texture(device, size[0], size[1]), accum_texture(device, size[0], size[1])],
            size,
            region,
            orbit: Orbit { centre, yaw: 0.6, pitch: 0.75, dist: region * 1.9, aperture: 0.0 },
            last_orbit: None,
            samples: 0,
            frame: 0,
            steps: env("STEPS", 2.0) as u32,
            paused: false,
            world,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        self.size = size;
        self.accum = [accum_texture(device, size[0], size[1]), accum_texture(device, size[0], size[1])];
        self.last_orbit = None;
    }

    /// Um frame: passos da simulação, a zona vista de cima (cor e altura),
    /// uma amostra do traçado de raios acumulada, e a imagem para `target`.
    fn frame(&mut self, gpu: &Gpu, target: &wgpu::TextureView) {
        let o = self.orbit;
        let moving = !self.paused && self.steps > 0;
        if self.last_orbit != Some(o) {
            self.samples = 0;
            self.last_orbit = Some(o);
        }
        // Parada, a média vai convergindo; a correr, pesa mais o presente (o
        // que se mexe deixa um rasto curto).
        let weight = if moving { (1.0 / (self.samples + 1) as f32).max(0.12) } else { 1.0 / (self.samples + 1) as f32 };
        self.samples += 1;
        self.frame += 1;

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        if moving {
            self.world.encode_steps(&gpu.queue, &mut enc, self.steps.min(ribossome::world::MAX_STEPS_PER_FRAME));
        }
        let r = self.region;
        self.world.set_draw_rect(&gpu.queue, Some(([o.centre[0] - r, o.centre[1] - r], [o.centre[0] + r, o.centre[1] + r])));
        self.world.encode_draw_list(&mut enc);
        let cam = Camera { center: o.centre, zoom: TEX as f32 / (2.0 * r) };
        self.cap.view.epoch.set(self.world.params.epoch);
        self.cap.view.ghost_steps.set(60.0);
        self.cap.encode(&gpu.queue, &mut enc, &cam, 0, 0.7);
        self.height_view.epoch.set(self.world.params.epoch);
        self.height_view.ghost_steps.set(60.0);
        self.height_view.update(&gpu.queue, &cam, [TEX as f32; 2], 0, 0.0, 0);
        {
            let many = self.height_msaa.create_view(&Default::default());
            let depth = self.height_depth.create_view(&Default::default());
            let resolve = self.height_tex.create_view(&Default::default());
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("height"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &many,
                    depth_slice: None,
                    resolve_target: Some(&resolve),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(depth_attachment(&depth)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.height_view.draw(&mut pass);
        }
        // Câmara: olha para o centro da zona, a meia altura do relevo.
        let target_pt = [o.centre[0], o.centre[1], 0.3 * HMAX];
        let (sp, cp) = o.pitch.sin_cos();
        let (sy, cy) = o.yaw.sin_cos();
        let eye = [target_pt[0] + o.dist * cp * sy, target_pt[1] - o.dist * cp * cy, target_pt[2] + o.dist * sp];
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let fwd = norm([target_pt[0] - eye[0], target_pt[1] - eye[1], target_pt[2] - eye[2]]);
        let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        let v4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        let data: [[f32; 4]; 7] = [
            v4(eye),
            v4(right),
            v4(up),
            v4(fwd),
            [o.centre[0], o.centre[1], r, HMAX],
            [self.size[0] as f32, self.size[1] as f32, self.frame as f32, weight],
            [o.dist, o.aperture, 0.36, 1.0],
        ];
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&data));
        let (src, dst) = ((self.frame % 2) as usize, ((self.frame + 1) % 2) as usize);
        let color = self.cap.texture_view();
        let height = self.height_tex.create_view(&Default::default());
        let bind = |prev: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&color) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&height) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(prev) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            })
        };
        let prev_view = self.accum[src].create_view(&Default::default());
        let next_view = self.accum[dst].create_view(&Default::default());
        let pass_to = |enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bg: &wgpu::BindGroup| {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scope"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };
        pass_to(&mut enc, &next_view, &self.march, &bind(&prev_view));
        pass_to(&mut enc, target, &self.present, &bind(&next_view));
        gpu.queue.submit([enc.finish()]);
    }
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    scope: Scope,
    cursor: [f32; 2],
    orbiting: bool,
    panning: bool,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Ribossome: microscope").with_inner_size(winit::dpi::LogicalSize::new(1280, 800)))
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
        let scope = Scope::new(&gpu, format, [surface_cfg.width, surface_cfg.height]);
        Self { window, surface, surface_cfg, gpu, scope, cursor: [0.0; 2], orbiting: false, panning: false }
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
        self.scope.frame(&self.gpu, &target);
        if self.scope.frame % 30 == 0 {
            let s = &self.scope;
            self.window.set_title(&format!(
                "Ribossome: microscope   {} samples   {}   aperture {:.1}   (drag: orbit, right drag: move, wheel: zoom, space: pause, F/G: aperture)",
                s.samples,
                if s.paused || s.steps == 0 { "paused" } else { "running" },
                s.orbit.aperture
            ));
        }
        self.window.pre_present_notify();
        self.gpu.queue.present(tex);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                self.surface_cfg.width = s.width.max(1);
                self.surface_cfg.height = s.height.max(1);
                self.surface.configure(&self.gpu.device, &self.surface_cfg);
                self.scope.resize(&self.gpu.device, [self.surface_cfg.width, self.surface_cfg.height]);
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                let (dx, dy) = (p[0] - self.cursor[0], p[1] - self.cursor[1]);
                let o = &mut self.scope.orbit;
                if self.orbiting {
                    o.yaw -= dx * 0.006;
                    o.pitch = (o.pitch + dy * 0.006).clamp(0.12, 1.55);
                }
                if self.panning {
                    // Desloca a zona no plano do fundo, no referencial da câmara.
                    let k = o.dist / self.surface_cfg.height as f32;
                    let (sy, cy) = o.yaw.sin_cos();
                    o.centre[0] += (-dx * cy + dy * sy) * k;
                    o.centre[1] += (-dx * sy - dy * cy) * k;
                }
                self.cursor = p;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.orbiting = down,
                    MouseButton::Right => self.panning = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let o = &mut self.scope.orbit;
                o.dist = (o.dist * 0.9f32.powf(lines)).clamp(60.0, 6.0 * self.scope.region);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => {
                    self.scope.paused = !self.scope.paused;
                    self.scope.last_orbit = None;
                }
                Key::Character("f") => self.scope.orbit.aperture = (self.scope.orbit.aperture - 1.0).max(0.0),
                Key::Character("g") => self.scope.orbit.aperture = (self.scope.orbit.aperture + 1.0).min(30.0),
                _ => {}
            },
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

/// Sem janela: acumula `samples` amostras com a simulação parada e grava a
/// imagem. YAW, PITCH (radianos), DIST e APERTURE mudam a câmara.
fn photo(path: &str, samples: u32) {
    let gpu = Gpu::new_headless().expect("GPU");
    let (w, h) = (1280u32, 768u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut scope = Scope::new(&gpu, format, [w, h]);
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    scope.orbit.yaw = env("YAW", scope.orbit.yaw);
    scope.orbit.pitch = env("PITCH", scope.orbit.pitch);
    scope.orbit.dist = env("DIST", scope.orbit.dist);
    scope.orbit.aperture = env("APERTURE", scope.orbit.aperture);
    // Uns passos para a grelha de desenho e as poses assentarem, depois parada.
    scope.frame(&gpu, &gpu.device.create_texture(&target_desc(w, h, format)).create_view(&Default::default()));
    scope.paused = true;
    scope.last_orbit = None;
    let target = gpu.device.create_texture(&target_desc(w, h, format));
    let view = target.create_view(&Default::default());
    for _ in 0..samples.max(1) {
        scope.frame(&gpu, &view);
    }
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("photo"),
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    gpu.wait_idle();
    let data = readback.get_mapped_range(..).expect("mapped").to_vec();
    readback.unmap();
    let file = std::io::BufWriter::new(std::fs::File::create(path).expect("criar o PNG"));
    let mut e = png::Encoder::new(file, w, h);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&data).unwrap();
    println!("{path}: {samples} amostras, {w} × {h}");
}

fn target_desc(w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("photo"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,microscopio=info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--foto") {
        photo(args.get(2).map(String::as_str).unwrap_or("microscopio.png"), args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64));
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app");
}
