//! PLANETAS: uma nuvem de matéria em quanta numa grelha 2D, com inércia,
//! gravidade e pressão. Não há estrela pré-feita: tudo é matéria; o que se
//! forma (estrelas, discos, planetas) vem do colapso.
//!
//! Cada célula guarda um PACOTE: massa (quanta inteiros), velocidade e centro
//! de massa dentro da célula (0..1). Em cada passo:
//!   1. GRAVIDADE 1/r² (P³M): uma grelha grossa (blocos de B×B células) com a
//!      massa e o centro de massa de cada bloco; cada bloco soma a atração de
//!      TODOS os outros fora dos 3×3 à volta (campo distante); cada célula
//!      soma, célula a célula, a das que estão nesses 3×3 blocos (campo
//!      próximo). Cada massa conta uma vez.
//!   2. PRESSÃO: P = K·mᵞ por célula; empurra as velocidades para longe das
//!      zonas densas (−∇P/m).
//!   3. MOVIMENTO + RECOLHA: cada pacote anda v·dt e cai INTEIRO na célula do
//!      seu centro de massa; cada célula junta o que lhe cai (massa e momento
//!      somados, centro de massa pela média pesada): colisão inelástica.
//!   4. EXPANSÃO: uma célula com mais pressão do que uma vizinha passa-lhe
//!      quanta inteiros (levam a sua velocidade). Sem isto um pacote nunca se
//!      partia. Mais compressão (gravidade) = mais quanta por célula.
//!
//! Massa e momento linear conservam-se exatamente. Nada anda mais de uma
//! célula por passo (CFL). Unidades: células e passos.

use std::sync::Arc;

use ribossome::gpu::Gpu;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const N: u32 = 1024;
/// Lado do bloco da grelha grossa (células) e número de blocos por lado.
const B: u32 = 8;
const C: u32 = N / B;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    g: f32,
    k_press: f32,
    gamma: f32,
    expand: f32,
    soft: f32,
    epoch: u32,
    n: u32,
    c: u32,
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

const COMMON: &str = r#"
struct SimParams { g: f32, k_press: f32, gamma: f32, expand: f32, soft: f32, epoch: u32, n: u32, c: u32 }
// Velocidade máxima (células por passo): um pacote nunca salta mais de uma.
const VMAX: f32 = 0.95;
const BLK: u32 = 8u;
"#;

const SIM_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read_write> mass_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> dyn_a: array<vec4<f32>>;   // (vx, vy, cx, cy)
@group(0) @binding(3) var<storage, read_write> dyn_tmp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mass_b: array<u32>;
@group(0) @binding(5) var<storage, read_write> dyn_b: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> coarse: array<vec4<f32>>;   // (m, x, y, _) centro de massa absoluto
@group(0) @binding(7) var<storage, read_write> coarse_acc: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>, 8>;
// Contador de passos (para os sorteios), incrementado na GPU no fim de cada passo.
@group(0) @binding(9) var<storage, read_write> step_counter: array<atomic<u32>, 1>;

@compute @workgroup_size(1)
fn tick() {
    atomicAdd(&step_counter[0], 1u);
}

fn hash(a: u32, b: u32, c: u32) -> f32 {
    var x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ c * 0xC2B2AE3Du;
    x ^= x >> 15u; x *= 0x2C1B3C6Du; x ^= x >> 12u; x *= 0x297A2D39u; x ^= x >> 15u;
    return f32(x >> 8u) / 16777216.0;
}

fn pressure(m: u32) -> f32 {
    return P.k_press * pow(f32(m), P.gamma);
}

fn mass_at(x: i32, y: i32) -> u32 {
    if (x < 0 || y < 0 || x >= i32(P.n) || y >= i32(P.n)) { return 0u; }
    return mass_a[u32(y) * P.n + u32(x)];
}

// 1a. Grelha grossa: massa e centro de massa de cada bloco.
@compute @workgroup_size(8, 8)
fn coarse_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.c || gid.y >= P.c) { return; }
    var m = 0.0;
    var s = vec2<f32>(0.0);
    for (var dy = 0u; dy < BLK; dy++) {
        for (var dx = 0u; dx < BLK; dx++) {
            let x = gid.x * BLK + dx;
            let y = gid.y * BLK + dy;
            let i = y * P.n + x;
            let mi = f32(mass_a[i]);
            if (mi == 0.0) { continue; }
            m += mi;
            s += mi * (vec2<f32>(f32(x), f32(y)) + dyn_a[i].zw);
        }
    }
    let c = select(vec2<f32>(f32(gid.x * BLK), f32(gid.y * BLK)) + 4.0, s / max(m, 1e-6), m > 0.0);
    coarse[gid.y * P.c + gid.x] = vec4<f32>(m, c, 0.0);
}

// 1b. Campo distante: cada bloco soma a atração dos blocos fora dos 3×3 à volta.
@compute @workgroup_size(8, 8)
fn coarse_far(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.c || gid.y >= P.c) { return; }
    let here = coarse[gid.y * P.c + gid.x];
    let p = select(vec2<f32>(f32(gid.x * BLK), f32(gid.y * BLK)) + 4.0, here.yz, here.x > 0.0);
    var a = vec2<f32>(0.0);
    for (var jy = 0u; jy < P.c; jy++) {
        let near_row = abs(i32(jy) - i32(gid.y)) <= 1;
        for (var jx = 0u; jx < P.c; jx++) {
            if (near_row && abs(i32(jx) - i32(gid.x)) <= 1) { continue; }
            let o = coarse[jy * P.c + jx];
            if (o.x == 0.0) { continue; }
            let d = o.yz - p;
            let r2 = dot(d, d) + P.soft * P.soft;
            a += o.x * d / (r2 * sqrt(r2));
        }
    }
    coarse_acc[gid.y * P.c + gid.x] = P.g * a;
}

// 2. KICK: campo distante + próximo (células dos 3×3 blocos) + pressão.
@compute @workgroup_size(16, 16)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let i = y * P.n + x;
    let m = mass_a[i];
    let d = dyn_a[i];
    if (m == 0u) { dyn_tmp[i] = vec4<f32>(0.0); return; }
    let pos = vec2<f32>(f32(x), f32(y)) + d.zw;
    let cx = i32(x / BLK);
    let cy = i32(y / BLK);
    var a = coarse_acc[u32(cy) * P.c + u32(cx)];
    var near = vec2<f32>(0.0);
    let x0 = max((cx - 1) * i32(BLK), 0);
    let y0 = max((cy - 1) * i32(BLK), 0);
    let x1 = min((cx + 2) * i32(BLK), i32(P.n));
    let y1 = min((cy + 2) * i32(BLK), i32(P.n));
    for (var yy = y0; yy < y1; yy++) {
        for (var xx = x0; xx < x1; xx++) {
            if (xx == i32(x) && yy == i32(y)) { continue; }
            let j = u32(yy) * P.n + u32(xx);
            let mj = mass_a[j];
            if (mj == 0u) { continue; }
            let dd = vec2<f32>(f32(xx), f32(yy)) + dyn_a[j].zw - pos;
            let r2 = dot(dd, dd) + P.soft * P.soft;
            near += f32(mj) * dd / (r2 * sqrt(r2));
        }
    }
    a += P.g * near;
    // Pressão: −∇P / m (diferenças centradas entre vizinhas).
    let ix = i32(x);
    let iy = i32(y);
    let gp = vec2<f32>(pressure(mass_at(ix + 1, iy)) - pressure(mass_at(ix - 1, iy)),
                       pressure(mass_at(ix, iy + 1)) - pressure(mass_at(ix, iy - 1))) * 0.5;
    a -= gp / f32(m);
    var v = d.xy + a;
    let sp = length(v);
    if (sp > VMAX) { v *= VMAX / sp; }
    dyn_tmp[i] = vec4<f32>(v, d.zw);
}

// Onde cai o pacote da célula s: contra a parede do mundo fica e reflete.
fn landing(s: vec2<i32>, d: vec4<f32>) -> vec4<f32> {
    var v = d.xy;
    var p = vec2<f32>(s) + d.zw + v;
    if (p.x < 0.0 || p.x >= f32(P.n)) { v.x = -v.x; p.x = f32(s.x) + d.z; }
    if (p.y < 0.0 || p.y >= f32(P.n)) { v.y = -v.y; p.y = f32(s.y) + d.w; }
    return vec4<f32>(p, v);
}

// 3. MOVIMENTO + RECOLHA (a -> b).
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
            let ms = mass_a[j];
            if (ms == 0u) { continue; }
            let l = landing(s, dyn_tmp[j]);
            if (any(vec2<i32>(floor(l.xy)) != here)) { continue; }
            mt += ms;
            mom += f32(ms) * l.zw;
            com += f32(ms) * (l.xy - vec2<f32>(here));
        }
    }
    let i = y * P.n + x;
    mass_b[i] = mt;
    if (mt == 0u) {
        dyn_b[i] = vec4<f32>(0.0);
    } else {
        dyn_b[i] = vec4<f32>(mom / f32(mt), clamp(com / f32(mt), vec2<f32>(0.0), vec2<f32>(0.99999)));
    }
}

// 4. EXPANSÃO (b -> a): quanta da célula s para a vizinha na direção k,
// ∝ à diferença RELATIVA de pressão (no máximo metade da massa no total).
// Calcula-se igual no emissor e no recetor (o mesmo sorteio): sem atómicas.
fn mass_b_at(x: i32, y: i32) -> u32 {
    if (x < 0 || y < 0 || x >= i32(P.n) || y >= i32(P.n)) { return 0xFFFFFFFFu; }
    return mass_b[u32(y) * P.n + u32(x)];
}

fn outflow(s: vec2<i32>) -> vec4<u32> {
    let ms = mass_b_at(s.x, s.y);
    var q = vec4<u32>(0u);
    if (ms == 0u || ms == 0xFFFFFFFFu || P.expand <= 0.0) { return q; }
    let ps = pressure(ms);
    var dirs = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
    var raw = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let t = s + dirs[k];
        let mt = mass_b_at(t.x, t.y);
        if (mt == 0xFFFFFFFFu) { continue; } // parede
        let pt = pressure(mt);
        if (ps > pt) { raw[k] = P.expand * (ps - pt) / ps * f32(ms) * 0.25; }
    }
    var tot = 0u;
    for (var k = 0u; k < 4u; k++) {
        let r = hash(u32(s.y) * P.n + u32(s.x), k, atomicLoad(&step_counter[0]));
        q[k] = u32(floor(raw[k] + r));
        tot += q[k];
    }
    let cap = ms / 2u;
    if (tot > cap) {
        for (var k = 0u; k < 4u; k++) { q[k] = q[k] * cap / tot; }
    }
    return q;
}

@compute @workgroup_size(16, 16)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let i = y * P.n + x;
    let here = vec2<i32>(i32(x), i32(y));
    let own = dyn_b[i];
    let out = outflow(here);
    let m0 = mass_b[i];
    let keep = m0 - (out.x + out.y + out.z + out.w);
    var mt = keep;
    var mom = f32(keep) * own.xy;
    var com = f32(keep) * own.zw;
    // Entradas: a vizinha na direção k manda-me o que sai dela na direção oposta.
    var dirs = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
    var opp = array<u32, 4>(1u, 0u, 3u, 2u);
    // Onde entram (perto do lado de onde vêm).
    var entry = array<vec2<f32>, 4>(vec2<f32>(0.95, 0.5), vec2<f32>(0.05, 0.5), vec2<f32>(0.5, 0.95), vec2<f32>(0.5, 0.05));
    for (var k = 0u; k < 4u; k++) {
        let s = here + dirs[k];
        if (s.x < 0 || s.y < 0 || s.x >= i32(P.n) || s.y >= i32(P.n)) { continue; }
        let qin = outflow(s)[opp[k]];
        if (qin == 0u) { continue; }
        let ds = dyn_b[u32(s.y) * P.n + u32(s.x)];
        mt += qin;
        mom += f32(qin) * ds.xy;
        com += f32(qin) * entry[k];
    }
    mass_a[i] = mt;
    if (mt == 0u) {
        dyn_a[i] = vec4<f32>(0.0);
    } else {
        dyn_a[i] = vec4<f32>(mom / f32(mt), clamp(com / f32(mt), vec2<f32>(0.0), vec2<f32>(0.99999)));
    }
}

// Estatísticas: massa total, células com massa, maior massa, momento angular
// em torno do centro da grelha, células com >= 50 quanta.
@compute @workgroup_size(16, 16)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.n || y >= P.n) { return; }
    let i = y * P.n + x;
    let m = mass_a[i];
    if (m == 0u) { return; }
    atomicAdd(&stats[0], m);
    atomicAdd(&stats[1], 1u);
    atomicMax(&stats[2], m);
    let d = dyn_a[i];
    let r = vec2<f32>(f32(x), f32(y)) + d.zw - vec2<f32>(f32(P.n) * 0.5);
    atomicAdd(&stats[3], bitcast<u32>(i32(round(f32(m) * (r.x * d.y - r.y * d.x)))));
    if (m >= 50u) { atomicAdd(&stats[4], 1u); }
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
    let w = vec2<f32>(V.center_x + in.uv.x * V.scale * V.aspect, V.center_y + in.uv.y * V.scale);
    var col = vec3<f32>(0.03);
    if (w.x >= 0.0 && w.y >= 0.0 && w.x < f32(P.n) && w.y < f32(P.n)) {
        col = vec3<f32>(0.0);
        let m = f32(mass_v[u32(w.y) * P.n + u32(w.x)]);
        if (m > 0.0) {
            // Gás ténue avermelhado -> denso dourado -> núcleos brancos.
            let t = clamp(log2(1.0 + m) / V.max_log, 0.0, 1.0);
            col = mix(vec3<f32>(0.25, 0.08, 0.06), vec3<f32>(1.0, 0.75, 0.35), sqrt(t));
            col = mix(col, vec3<f32>(1.0, 1.0, 0.95), smoothstep(0.75, 1.0, t));
        }
    }
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

/// Nuvem inicial: disco de raio `radius`, com rotação e agitação ao acaso.
struct Cloud {
    radius: f32,
    fill: f32,
    spin: f32,
    dispersion: f32,
    seed: u64,
}

fn make_cloud(s: &Cloud) -> (Vec<u32>, Vec<[f32; 4]>) {
    let n = N as usize;
    let c = N as f32 * 0.5;
    let mut rng = SplitMix(s.seed);
    let mut mass = vec![0u32; n * n];
    let mut dynv = vec![[0f32; 4]; n * n];
    for y in 0..n {
        for x in 0..n {
            let (px, py) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
            let r = (px * px + py * py).sqrt();
            if r > s.radius || rng.f32() > s.fill {
                continue;
            }
            // Rotação de corpo rígido (spin = velocidade na borda) + agitação.
            let w = s.spin / s.radius;
            let vx = -py * w + (rng.f32() - 0.5) * 2.0 * s.dispersion;
            let vy = px * w + (rng.f32() - 0.5) * 2.0 * s.dispersion;
            let i = y * n + x;
            mass[i] = 1 + (rng.f32() * 3.0) as u32;
            dynv[i] = [vx, vy, rng.f32() * 0.99, rng.f32() * 0.99];
        }
    }
    (mass, dynv)
}

struct Sim {
    params: SimParams,
    params_buf: wgpu::Buffer,
    mass_a: wgpu::Buffer,
    dyn_a: wgpu::Buffer,
    stats: wgpu::Buffer,
    bg: wgpu::BindGroup,
    coarse_build: wgpu::ComputePipeline,
    coarse_far: wgpu::ComputePipeline,
    kick: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    expand: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    tick: wgpu::ComputePipeline,
    step: u64,
}

impl Sim {
    fn new(gpu: &Gpu, params: SimParams) -> Self {
        let device = &gpu.device;
        let cells = (N * N) as u64;
        let mass_a = storage(device, "mass a", cells * 4);
        let dyn_a = storage(device, "dyn a", cells * 16);
        let dyn_tmp = storage(device, "dyn tmp", cells * 16);
        let mass_b = storage(device, "mass b", cells * 4);
        let dyn_b = storage(device, "dyn b", cells * 16);
        let coarse = storage(device, "coarse", (C * C) as u64 * 16);
        let coarse_acc = storage(device, "coarse acc", (C * C) as u64 * 8);
        let stats = storage(device, "stats", 32);
        let step_counter = storage(device, "step counter", 4);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planetas sim"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}{SIM_WGSL}").into()),
        });
        let st = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend((1..=9).map(st));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("planetas"), entries: &entries });
        let bufs = [&mass_a, &dyn_a, &dyn_tmp, &mass_b, &dyn_b, &coarse, &coarse_acc, &stats, &step_counter];
        let mut bge = vec![wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() }];
        bge.extend(bufs.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry { binding: k as u32 + 1, resource: b.as_entire_binding() }));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("planetas"), layout: &layout, entries: &bge });
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
            coarse_build: compute("coarse_build"),
            coarse_far: compute("coarse_far"),
            kick: compute("kick"),
            gather: compute("gather"),
            expand: compute("expand"),
            reduce: compute("reduce"),
            tick: compute("tick"),
            mass_a,
            dyn_a,
            stats,
            bg,
            step: 0,
        }
    }

    fn upload(&mut self, gpu: &Gpu, mass: &[u32], dynv: &[[f32; 4]]) {
        gpu.queue.write_buffer(&self.mass_a, 0, bytemuck::cast_slice(mass));
        gpu.queue.write_buffer(&self.dyn_a, 0, bytemuck::cast_slice(dynv));
        self.step = 0;
    }

    fn encode(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, steps: u32) {
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let g = N.div_ceil(16);
        let gc = C.div_ceil(8);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &self.bg, &[]);
        for _ in 0..steps {
            pass.set_pipeline(&self.coarse_build);
            pass.dispatch_workgroups(gc, gc, 1);
            pass.set_pipeline(&self.coarse_far);
            pass.dispatch_workgroups(gc, gc, 1);
            pass.set_pipeline(&self.kick);
            pass.dispatch_workgroups(g, g, 1);
            pass.set_pipeline(&self.gather);
            pass.dispatch_workgroups(g, g, 1);
            pass.set_pipeline(&self.expand);
            pass.dispatch_workgroups(g, g, 1);
            pass.set_pipeline(&self.tick);
            pass.dispatch_workgroups(1, 1, 1);
            self.step += 1;
        }
    }

    /// (massa total, células com massa, maior massa, momento angular, células >= 50).
    fn read_stats(&self, gpu: &Gpu) -> [u32; 8] {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.stats, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.bg, &[]);
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

fn default_params() -> SimParams {
    SimParams { g: 3.0e-5, k_press: 2.0e-4, gamma: 2.0, expand: 0.3, soft: 1.0, epoch: 0, n: N, c: C }
}

fn default_cloud() -> Cloud {
    Cloud { radius: 380.0, fill: 0.5, spin: 0.15, dispersion: 0.03, seed: 1 }
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    sim: Sim,
    cloud: Cloud,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
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

        let cloud = default_cloud();
        let mut sim = Sim::new(&gpu, default_params());
        let (m, d) = make_cloud(&cloud);
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
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sim.mass_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sim.params_buf.as_entire_binding() },
            ],
        });
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
            cloud,
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
            steps_per_frame: 4,
            stats: [0; 8],
            stats_l0: None,
            frame: 0,
        }
    }

    fn reset(&mut self) {
        let (m, d) = make_cloud(&self.cloud);
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
                    if ui.button("nuvem nova").clicked() {
                        reset = true;
                    }
                });
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=32).text("passos/frame"));
                let p = &mut self.sim.params;
                ui.add(egui::Slider::new(&mut p.g, 0.0..=1e-3).logarithmic(true).smallest_positive(1e-7).text("gravidade G"));
                ui.add(egui::Slider::new(&mut p.k_press, 0.0..=0.1).logarithmic(true).smallest_positive(1e-6).text("pressão K"))
                    .on_hover_text("P = K·mᵞ: mais pressão, mais difícil comprimir");
                ui.add(egui::Slider::new(&mut p.gamma, 1.0..=3.0).text("γ (rigidez)"));
                ui.add(egui::Slider::new(&mut p.expand, 0.0..=1.0).text("expansão (fluxo de quanta)"))
                    .on_hover_text("quanta passam para vizinhas com menos pressão");
                ui.separator();
                ui.label("nuvem nova:");
                let c = &mut self.cloud;
                ui.add(egui::Slider::new(&mut c.radius, 50.0..=500.0).text("raio"));
                ui.add(egui::Slider::new(&mut c.fill, 0.02..=1.0).text("densidade"));
                ui.add(egui::Slider::new(&mut c.spin, 0.0..=0.6).text("rotação (vel. na borda)"));
                ui.add(egui::Slider::new(&mut c.dispersion, 0.0..=0.3).text("agitação"));
                ui.separator();
                ui.label(format!("passo {step}"));
                ui.label(format!("massa total {} quanta (exata)", st[0]));
                ui.label(format!("células com massa {}  maior {} quanta  com ≥ 50: {}", st[1], st[2], st[4]));
                let l = st[3] as i32;
                ui.label(format!(
                    "momento angular {} ({:+.2}%)",
                    l,
                    if l0 != 0 { 100.0 * (l - l0) as f64 / l0 as f64 } else { 0.0 }
                ));
                ui.label("roda do rato: zoom · arrastar: mover");
            });
        });
        if reset {
            self.cloud.seed += 1;
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
            self.sim.encode(&self.gpu, &mut enc, self.steps_per_frame);
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
            pass.set_bind_group(0, &self.view_bg, &[]);
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

/// Sem janela: `planetas --teste [passos]` corre a nuvem e mostra a evolução.
fn headless(steps: u64) {
    let gpu = Gpu::new_headless().expect("GPU");
    let mut sim = Sim::new(&gpu, default_params());
    let (m, d) = make_cloud(&default_cloud());
    sim.upload(&gpu, &m, &d);
    let l0 = sim.read_stats(&gpu)[3] as i32;
    let t = std::time::Instant::now();
    let report = 10;
    for k in 0..=report {
        let target = steps * k / report;
        while sim.step < target {
            let n = (target - sim.step).min(32) as u32;
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            sim.encode(&gpu, &mut enc, n);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
        }
        let st = sim.read_stats(&gpu);
        let l = st[3] as i32;
        println!(
            "passo {:6}: massa {} células {:7} maior {:6} (≥50: {:5})  L {:+.2}%",
            sim.step,
            st[0],
            st[1],
            st[2],
            st[4],
            if l0 != 0 { 100.0 * (l - l0) as f64 / l0 as f64 } else { 0.0 }
        );
    }
    println!("{:.2} ms por passo", t.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,planetas=info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|a| a == "--teste").unwrap_or(false) {
        headless(args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5_000));
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app");
}
