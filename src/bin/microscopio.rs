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
//! zona, a roda aproxima. Começa PARADO (a imagem converge e fica nítida);
//! Espaço põe a simulação a correr e volta a parar. F/G fecham e abrem o
//! diafragma (profundidade de campo); C liga a cor (por omissão é a preto e
//! branco); M mostra os monómeros; Esc sai.
//!   TERRAIN=1 centra numa zona com rocha, entulho e água (em vez de num agente)
//!   COLOR=1, MONOMERS=0.7  arrancam com cor / com monómeros

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
const HMAX: f32 = 95.0;
/// Lado da textura de "quanto terreno há" e da do chão desfocado.
const PRES: u32 = 512;
const GROUND: u32 = 256;
/// Raio do desfoque do chão, em unidades do mundo.
const GROUND_BLUR: f32 = 55.0;

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
    // quanto da cor das peças se mantém (0 = preto e branco), lado da textura
    // do chão, raio do desfoque do chão (em fração da zona), livre
    opts: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
// Volume das peças: vermelho = cimo, verde = fundo (acima do chão; 0 = nada).
// No passo do chão liga-se aqui a textura de "quanto terreno há".
@group(0) @binding(2) var height_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
// Relevo suave do chão (0..1), feito no passo fs_ground.
@group(0) @binding(5) var ground_tex: texture_2d<f32>;
// O volume das peças lê-se da textura de VÁRIAS AMOSTRAS, uma amostra só: a
// versão resolvida faz a média das bordas com o vazio, e essa média (um cimo
// e um fundo a meio caminho de zero) aparecia como cortinas por baixo das peças.
@group(0) @binding(6) var vol_tex: texture_multisampled_2d<f32>;
// DUAS CAMADAS: a de cima (6) são os agentes; esta é o mundo por baixo deles
// (pedras e monómeros), com a sua cor à parte. Assim o que está debaixo de um
// bicho continua a existir.
@group(0) @binding(7) var world_vol: texture_multisampled_2d<f32>;
@group(0) @binding(8) var world_color: texture_2d<f32>;

// Altura do chão onde há rocha maciça (unidades do mundo).
const GROUND_H: f32 = 24.0;

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

// Ruído suave (0..1) para o grão do chão.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = hash3(vec3<u32>(vec2<u32>(i + vec2<i32>(32768)), 5u)).x;
    let b = hash3(vec3<u32>(vec2<u32>(i + vec2<i32>(32769, 32768)), 5u)).x;
    let c = hash3(vec3<u32>(vec2<u32>(i + vec2<i32>(32768, 32769)), 5u)).x;
    let d = hash3(vec3<u32>(vec2<u32>(i + vec2<i32>(32769, 32769)), 5u)).x;
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// Mundo (x, y) -> coordenadas da textura da zona (o mundo cresce para cima).
fn region_uv(xy: vec2<f32>) -> vec2<f32> {
    let d = (xy - u.region.xy) / (2.0 * u.region.z);
    return vec2<f32>(0.5 + d.x, 0.5 - d.y);
}

// CHÃO: o terreno desfocado. Onde há rocha ou entulho o chão sobe, num monte
// suave onde as pedras e os bichos assentam (em vez de um degrau a pique).
@fragment
fn fs_ground(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / u.opts.y;
    var sum = 0.0;
    var wsum = 0.0;
    for (var j = -4; j <= 4; j++) {
        for (var i = -4; i <= 4; i++) {
            let o = vec2<f32>(f32(i), f32(j)) * 0.25;
            let w = exp(-2.2 * dot(o, o));
            sum += w * textureSampleLevel(height_tex, samp, uv + o * u.opts.z, 0.0).r;
            wsum += w;
        }
    }
    return vec4<f32>(sum / wsum, 0.0, 0.0, 1.0);
}

fn ground_at(xy: vec2<f32>) -> f32 {
    let uv = clamp(region_uv(xy), vec2<f32>(0.0), vec2<f32>(1.0));
    let g = textureSampleLevel(ground_tex, samp, uv, 0.0).r;
    return GROUND_H * g * g * (3.0 - 2.0 * g);
}

// O que há no ponto xy, em alturas absolutas: o volume dos agentes e o do
// mundo (cimo, fundo; sem peça, um intervalo vazio) e o chão.
struct Surf {
    a: vec2<f32>,
    w: vec2<f32>,
    g: f32,
}
const NONE: vec2<f32> = vec2<f32>(-1000.0, 1000.0);

fn surf(xy: vec2<f32>) -> Surf {
    var s: Surf;
    s.g = ground_at(xy);
    s.a = NONE;
    s.w = NONE;
    let uv = region_uv(xy);
    if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
        let c = vec2<i32>(uv * vec2<f32>(textureDimensions(vol_tex)));
        let ta = textureLoad(vol_tex, c, 0).rg * u.lens.w;
        let tw = textureLoad(world_vol, c, 0).rg * u.lens.w;
        if (ta.x > 0.25) { s.a = vec2<f32>(s.g + ta.x, s.g + ta.y); }
        if (tw.x > 0.25) { s.w = vec2<f32>(s.g + tw.x, s.g + tw.y); }
    }
    return s;
}

// O mesmo, mas INTERPOLADO entre os 4 texels à volta (só entre os que têm
// peça, para não misturar com o vazio). Texel a texel cada peça era uma
// escadinha de caixas, e de lado viam-se os degraus como linhas de
// varrimento; o passo largo dos raios usa surf (barato) e a afinação final,
// a normal e a oclusão usam este.
fn lerp_volume(t: texture_multisampled_2d<f32>, uv: vec2<f32>, g: f32) -> vec2<f32> {
    let dim = vec2<f32>(textureDimensions(t));
    let x = uv * dim - 0.5;
    let c = vec2<i32>(floor(x));
    let f = x - floor(x);
    let hi = vec2<i32>(dim) - 1;
    var sum = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let v = textureLoad(t, clamp(c + o, vec2<i32>(0), hi), 0).rg;
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * step(0.25, v.x);
        sum += v * w;
        wsum += w;
    }
    if (wsum < 0.5) { return NONE; }
    return g + sum / wsum * u.lens.w;
}

fn surf_smooth(xy: vec2<f32>) -> Surf {
    var s: Surf;
    s.g = ground_at(xy);
    s.a = NONE;
    s.w = NONE;
    let uv = region_uv(xy);
    if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
        s.a = lerp_volume(vol_tex, uv, s.g);
        s.w = lerp_volume(world_vol, uv, s.g);
    }
    return s;
}

fn within(z: f32, v: vec2<f32>, eps: f32) -> bool {
    return z <= v.x + eps && z >= v.y - eps;
}

fn solid(p: vec3<f32>, s: Surf) -> bool {
    return p.z <= s.g || within(p.z, s.a, 0.0) || within(p.z, s.w, 0.0);
}

// A superfície mais alta no ponto: para a oclusão.
fn top_at(xy: vec2<f32>) -> f32 {
    let s = surf_smooth(xy);
    return max(s.g, max(s.a.x, s.w.x));
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
        // Do topo do relevo até ao fundo, em passos; ao entrar numa peça ou
        // no chão, afina por bisseção.
        let t0 = max((u.region.w - org.z) / dir.z, 0.0);
        let t1 = (0.0 - org.z) / dir.z;
        let dt = (t1 - t0) / 240.0;
        var t = t0 + dt * rnd.z;
        var prev_t = t0;
        for (var i = 0; i < 240; i++) {
            let p = org + dir * t;
            if (solid(p, surf(p.xy))) { break; }
            prev_t = t;
            t += dt;
        }
        t = min(t, t1);
        var lo = prev_t;
        var hi = t;
        for (var i = 0; i < 8; i++) {
            let m = 0.5 * (lo + hi);
            let p = org + dir * m;
            if (solid(p, surf_smooth(p.xy))) { hi = m; } else { lo = m; }
        }
        let p = org + dir * hi;
        let s = surf_smooth(p.xy);
        let e = 2.0 * u.region.z / f32(textureDimensions(vol_tex).x) * 1.5;
        // Onde bateu: num agente, numa peça do mundo, ou no chão?
        let on_agent = within(p.z, s.a, 0.06);
        let on_world = !on_agent && within(p.z, s.w, 0.06);
        let on_ground = !on_agent && !on_world;
        var n = vec3<f32>(0.0, 0.0, 1.0);
        var albedo = vec3<f32>(0.0);
        if (on_ground) {
            let eg = 5.0;
            let gx = ground_at(p.xy + vec2<f32>(eg, 0.0)) - ground_at(p.xy - vec2<f32>(eg, 0.0));
            let gy = ground_at(p.xy + vec2<f32>(0.0, eg)) - ground_at(p.xy - vec2<f32>(0.0, eg));
            n = normalize(vec3<f32>(-gx, -gy, 2.0 * eg));
            // O chão tem a cor do mundo sem agentes (a água, o que lá houver);
            // o monte de terreno, onde não há pedra à vista, é pó de rocha.
            albedo = textureSampleLevel(world_color, samp, region_uv(p.xy), 0.0).rgb * 0.5;
            // SUPORTE: onde não há nada, o chão é uma lâmina lisa (como vidro)
            // com um grão muito fino e umas partículas minúsculas pousadas.
            let grain = 0.5 * vnoise(p.xy * 0.33) + 0.3 * vnoise(p.xy * 1.7) + 0.2 * vnoise(p.xy * 6.1);
            let cell = floor(p.xy / 13.0);
            let hs = hash3(vec3<u32>(vec2<u32>(vec2<i32>(cell) + vec2<i32>(32768)), 17u));
            let speck = step(0.8, hs.z) * (1.0 - smoothstep(0.2, 1.0, length(p.xy / 13.0 - cell - 0.1 - 0.8 * hs.xy) * (5.0 + 6.0 * hs.x)));
            albedo = max(albedo, vec3<f32>(0.075 + 0.035 * grain + 0.16 * speck));
            albedo = max(albedo, vec3<f32>(0.11, 0.105, 0.1) * smoothstep(0.5, 6.0, s.g));
        } else {
            // A peça é a mesma forma para cima e para baixo do seu meio: onde
            // um vizinho já não tem peça, a superfície fecha no meio.
            let v = select(s.w, s.a, on_agent);
            let mid = 0.5 * (v.x + v.y);
            let upper = p.z >= mid;
            var d = vec4<f32>(0.0);
            var offs = array<vec2<f32>, 4>(vec2<f32>(e, 0.0), vec2<f32>(-e, 0.0), vec2<f32>(0.0, e), vec2<f32>(0.0, -e));
            for (var k = 0; k < 4; k++) {
                let q = surf_smooth(p.xy + offs[k]);
                let qv = select(q.w, q.a, on_agent);
                d[k] = select(mid, select(qv.y, qv.x, upper), qv.x > -500.0);
            }
            n = normalize(vec3<f32>(-(d[0] - d[1]), -(d[2] - d[3]), 2.0 * e));
            if (!upper) { n = vec3<f32>(-n.x, -n.y, -n.z); }
            // Numa parede (a borda de uma peça) lê-se a cor um pouco para dentro.
            let wall = clamp(1.0 - abs(n.z), 0.0, 1.0);
            let inward = -n.xy / max(length(n.xy), 1e-4) * (2.5 * e * wall);
            let uv = region_uv(p.xy + inward);
            albedo = textureSampleLevel(color_tex, samp, uv, 0.0).rgb;
            if (on_world) { albedo = textureSampleLevel(world_color, samp, uv, 0.0).rgb; }
        }
        let ndv = clamp(dot(n, -dir), 0.0, 1.0);
        // MICROSCÓPIO ELETRÓNICO: as superfícies de lado para o observador
        // soltam mais eletrões (arestas claras)...
        let edge = pow(1.0 - ndv, 2.0);
        // ...e os sítios encaixados entre vizinhos mais altos soltam menos
        // (oclusão): olha-se à volta, a duas distâncias.
        var occ = 0.0;
        for (var k = 0; k < 8; k++) {
            let a = 0.785398 * f32(k) + 6.2831853 * rnd2.z;
            let rr = select(9.0, 22.0, (k & 1) == 1);
            occ += clamp((top_at(p.xy + vec2<f32>(cos(a), sin(a)) * rr) - p.z) / rr, 0.0, 1.5);
        }
        let ao = 1.0 / (1.0 + 0.8 * occ);
        // PRETO E BRANCO, como uma micrografia: fica só o claro-escuro das
        // peças (opts.x repõe a cor).
        let gray = vec3<f32>(dot(albedo, vec3<f32>(0.33, 0.45, 0.22)) * 1.35);
        let base = max(mix(gray, albedo, u.opts.x), vec3<f32>(0.03, 0.031, 0.034));
        col = base * (0.95 + 1.5 * edge) * ao + vec3<f32>(0.25) * edge * edge * select(0.0, 1.0, !on_ground);
        // O campo de visão não acaba num quadrado: esbate-se com a distância
        // ao centro da zona e com a distância para lá do plano de focagem.
        let rd = length(p.xy - u.region.xy) / u.region.z;
        col *= 1.0 - smoothstep(0.8, 1.0, rd);
        col *= 1.0 - 0.6 * clamp((hi - u.lens.x) / (4.0 * u.region.z), 0.0, 1.0);
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
    /// A camada do mundo por baixo dos agentes: volumes e cor.
    world_vol_msaa: wgpu::Texture,
    world_col_msaa: wgpu::Texture,
    world_col_tex: wgpu::Texture,
    pres_msaa: wgpu::Texture,
    pres_depth: wgpu::Texture,
    pres_tex: wgpu::Texture,
    ground_tex: wgpu::Texture,
    ground: wgpu::RenderPipeline,
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
    /// Manter a cor das peças (por omissão é a preto e branco).
    colour: bool,
    /// Brilho dos monómeros na zona (0 = não se desenham).
    monomers: f32,
}

/// Alvo de várias amostras de onde também se lê (uma amostra de cada vez).
fn readable_msaa(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume msaa"),
        size: wgpu::Extent3d { width: TEX, height: TEX, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: ribossome::render::MSAA,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Textura quadrada de vírgula flutuante onde se desenha e de onde se lê.
fn float_target(device: &wgpu::Device, side: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("float target"),
        size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HEIGHT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
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
        // TERRAIN=1: em vez disso, o bloco do mundo com a melhor mistura de
        // rocha, entulho e água (para ver o terreno).
        let centre = if env("TERRAIN", 0.0) != 0.0 {
            let gamma = world.read_gamma_blocking(gpu);
            let n = cfg.grid_size as usize;
            let b = 24usize;
            let mut best_at = (0.0f32, centre);
            for by in (0..n - b).step_by(b) {
                for bx in (0..n - b).step_by(b) {
                    let (mut rock, mut rubble) = (0u32, 0u32);
                    for y in by..by + b {
                        for x in bx..bx + b {
                            let v = gamma[y * n + x];
                            rock += (v >= 3) as u32;
                            rubble += (v > 0 && v < 3) as u32;
                        }
                    }
                    let t = (b * b) as f32;
                    let score = (rock as f32 / t).min(rubble as f32 / t).min(1.0 - (rock + rubble) as f32 / t);
                    if score > best_at.0 {
                        let u = cfg.world_units_per_cell as f32;
                        best_at = (score, [(bx + b / 2) as f32 * u, (by + b / 2) as f32 * u]);
                    }
                }
            }
            best_at.1
        } else {
            centre
        };
        let region = env("REGION", 420.0);
        log::info!("{path}: {} agentes; zona de {} unidades à volta de ({:.0}, {:.0})", alive.len(), 2.0 * region, centre[0], centre[1]);

        let device = &gpu.device;
        let cap = Capture::new(gpu, &world, TEX);
        let height_view = WorldView::new(device, &gpu.queue, &world, HEIGHT_FORMAT);
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
            size: 8 * 16,
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
                tex_entry(5),
                tex_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
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
            height_msaa: readable_msaa(device, HEIGHT_FORMAT),
            world_vol_msaa: readable_msaa(device, HEIGHT_FORMAT),
            world_col_msaa: msaa_texture(device, wgpu::TextureFormat::Rgba8Unorm, TEX, TEX),
            world_col_tex: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("world colour"),
                size: wgpu::Extent3d { width: TEX, height: TEX, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
            height_depth: depth_texture(device, TEX, TEX),
            height_tex,
            pres_msaa: msaa_texture(device, HEIGHT_FORMAT, PRES, PRES),
            pres_depth: depth_texture(device, PRES, PRES),
            pres_tex: float_target(device, PRES),
            ground_tex: float_target(device, GROUND),
            ground: pipeline("fs_ground", HEIGHT_FORMAT),
            uniform,
            march: pipeline("fs_march", ACCUM_FORMAT),
            present: pipeline("fs_present", target_format),
            layout,
            sampler,
            accum: [accum_texture(device, size[0], size[1]), accum_texture(device, size[0], size[1])],
            size,
            region,
            orbit: Orbit { centre, yaw: 0.6, pitch: 0.75, dist: region * 1.3, aperture: env("APERTURE", 1.5) },
            last_orbit: None,
            samples: 0,
            frame: 0,
            steps: env("STEPS", 2.0) as u32,
            // Começa PARADO: é assim que a imagem converge e se vê bem.
            paused: true,
            colour: env("COLOR", 0.0) != 0.0,
            monomers: env("MONOMERS", 0.7),
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

        let r = self.region;
        // QUANTO TERRENO HÁ na zona (só o fundo, sem agentes), num envio à
        // parte: usa a mesma vista do passo dos volumes com outro modo, e os
        // parâmetros da vista são um só bloco, escrito antes de os comandos correrem.
        {
            let cam = Camera { center: o.centre, zoom: PRES as f32 / (2.0 * r) };
            self.height_view.height_pass.set(2);
            self.height_view.update(&gpu.queue, &cam, [PRES as f32; 2], 0, 0.0, 0);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let many = self.pres_msaa.create_view(&Default::default());
            let depth = self.pres_depth.create_view(&Default::default());
            let resolve = self.pres_tex.create_view(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain presence"),
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
                self.height_view.draw_world_only(&mut pass);
            }
            gpu.queue.submit([enc.finish()]);
            self.height_view.height_pass.set(1);
        }
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        if moving {
            self.world.encode_steps(&gpu.queue, &mut enc, self.steps.min(ribossome::world::MAX_STEPS_PER_FRAME));
        }
        self.world.set_draw_rect(&gpu.queue, Some(([o.centre[0] - r, o.centre[1] - r], [o.centre[0] + r, o.centre[1] + r])));
        self.world.encode_draw_list(&mut enc);
        let cam = Camera { center: o.centre, zoom: TEX as f32 / (2.0 * r) };
        self.cap.view.epoch.set(self.world.params.epoch);
        self.cap.view.ghost_steps.set(60.0);
        self.cap.encode(&gpu.queue, &mut enc, &cam, 0, self.monomers);
        self.height_view.epoch.set(self.world.params.epoch);
        self.height_view.ghost_steps.set(60.0);
        // (Com monómeros: no passo dos volumes cada molécula é um grãozinho.)
        self.height_view.update(&gpu.queue, &cam, [TEX as f32; 2], 0, self.monomers, 0);
        // TRÊS desenhos da zona vista de cima: o volume dos agentes, o volume
        // do mundo por baixo deles (pedras, monómeros) e a cor desse mundo
        // sem agentes (a cor com agentes já está em self.cap).
        {
            let depth = self.height_depth.create_view(&Default::default());
            let agents_vol = self.height_msaa.create_view(&Default::default());
            let world_vol = self.world_vol_msaa.create_view(&Default::default());
            let world_col = self.world_col_msaa.create_view(&Default::default());
            let world_col_resolve = self.world_col_tex.create_view(&Default::default());
            let mut layer = |label: &'static str, view: &wgpu::TextureView, resolve: Option<&wgpu::TextureView>, draw: &dyn Fn(&mut wgpu::RenderPass<'_>)| {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: resolve,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(depth_attachment(&depth)),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw(&mut pass);
            };
            layer("agent volumes", &agents_vol, None, &|pass| self.height_view.draw_agents_only(pass));
            layer("world volumes", &world_vol, None, &|pass| self.height_view.draw_world_only(pass));
            layer("world colour", &world_col, Some(&world_col_resolve), &|pass| self.cap.view.draw_world_only(pass));
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
        let data: [[f32; 4]; 8] = [
            v4(eye),
            v4(right),
            v4(up),
            v4(fwd),
            [o.centre[0], o.centre[1], r, HMAX],
            [self.size[0] as f32, self.size[1] as f32, self.frame as f32, weight],
            [o.dist, o.aperture, 0.36, 1.0],
            [if self.colour { 1.0 } else { 0.0 }, GROUND as f32, GROUND_BLUR / (2.0 * r), 0.0],
        ];
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&data));
        let (src, dst) = ((self.frame % 2) as usize, ((self.frame + 1) % 2) as usize);
        let color = self.cap.texture_view();
        let height = self.height_tex.create_view(&Default::default());
        let pres = self.pres_tex.create_view(&Default::default());
        let volume_msaa = self.height_msaa.create_view(&Default::default());
        let world_volume = self.world_vol_msaa.create_view(&Default::default());
        let world_colour = self.world_col_tex.create_view(&Default::default());
        let ground = self.ground_tex.create_view(&Default::default());
        let bind = |volume: &wgpu::TextureView, prev: &wgpu::TextureView, floor: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&color) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(volume) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(prev) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(floor) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&volume_msaa) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&world_volume) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&world_colour) },
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
        // O chão desfocado (lê "quanto terreno há"), a amostra do traçado de
        // raios e a imagem final.
        pass_to(&mut enc, &ground, &self.ground, &bind(&pres, &prev_view, &pres));
        pass_to(&mut enc, &next_view, &self.march, &bind(&height, &prev_view, &ground));
        pass_to(&mut enc, target, &self.present, &bind(&height, &next_view, &ground));
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
                "Ribossome: microscope   {} samples   {}   aperture {:.1}   (drag: orbit, right drag: move, wheel: zoom in/out, space: run/pause, F/G: aperture, C: colour, M: monomers)",
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
                    // (Arrastar para baixo traz a cena para o observador.)
                    o.centre[0] += (-dx * cy - dy * sy) * k;
                    o.centre[1] += (-dx * sy + dy * cy) * k;
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
                // ZOOM: a câmara afasta-se e a zona desenhada cresce com ela (a
                // de perto é pequena e detalhada, a de longe apanha mais mundo).
                let o = &mut self.scope.orbit;
                o.dist = (o.dist * 0.9f32.powf(lines)).clamp(90.0, 5000.0);
                self.scope.region = (o.dist / 1.3).clamp(50.0, 2600.0);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => {
                    self.scope.paused = !self.scope.paused;
                    self.scope.last_orbit = None;
                }
                Key::Character("f") => self.scope.orbit.aperture = (self.scope.orbit.aperture - 1.0).max(0.0),
                Key::Character("g") => self.scope.orbit.aperture = (self.scope.orbit.aperture + 1.0).min(30.0),
                Key::Character("c") => {
                    self.scope.colour = !self.scope.colour;
                    self.scope.last_orbit = None;
                }
                Key::Character("m") => {
                    self.scope.monomers = if self.scope.monomers > 0.0 { 0.0 } else { 0.7 };
                    self.scope.last_orbit = None;
                }
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
    if std::env::var("REGION").is_err() {
        scope.region = (scope.orbit.dist / 1.3).clamp(50.0, 2600.0);
    }
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
