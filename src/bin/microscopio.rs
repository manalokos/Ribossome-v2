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
//! diafragma (profundidade de campo); Z/X alongam e encurtam a lente (longa =
//! quase axonométrica; por omissão 135 mm); E/R clareiam e escurecem a
//! exposição; C liga a cor (por omissão é a preto e branco); M esconde os
//! monómeros; S liga a superamostragem; Esc sai.
//!   LENS=mm, APERTURE, EXPOSURE  arrancam com esses valores
//! Um CLIQUE (sem arrastar) foca no ponto clicado; Tab mostra e esconde o
//! painel de controlos; por baixo da imagem fica a barra de dados com a
//! escala em nanómetros (ver NM_PER_UNIT: é uma convenção).
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
// A média acumulada guarda-se com 32 bits por canal: com 16, ao fim de umas
// centenas de amostras cada amostra nova já pesava menos do que a precisão
// do número e a imagem deixava de melhorar.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Altura máxima do relevo (unidades do mundo): o raio começa a marchar aqui.
// (Tem de ficar ACIMA de tudo: chão 24 + pedras + corpo + órgão grande passa
// dos 150. Com 95 os raios nasciam já dentro das peças mais altas e elas
// apareciam cortadas por um plano, com a "tampa" deslocada para o lado.)
const HMAX: f32 = 200.0;
/// Lado da textura de "quanto terreno há" e da do chão desfocado.
const PRES: u32 = 512;
const GROUND: u32 = 256;
/// Tangente de meio campo de uma lente de 33 mm (a referência de `Orbit::dist`).
const REF_TAN: f32 = 0.36;
/// Meio lado da zona desenhada, em distâncias da câmara ao ponto que ela olha:
/// larga, para a cena se apagar com a distância sem acabar num corte.
const REGION_PER_DIST: f32 = 1.7;
/// Raio de uma molécula de monómero, em células (o desenho normal usa mais).
const MOLECULE_R: f32 = 0.2;
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
    // do chão, raio do desfoque do chão (em fração da zona), superamostragem
    opts: vec4<f32>,
    // exposição (multiplica a imagem final), livre × 3
    photo: vec4<f32>,
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

fn pcg(v: u32) -> u32 {
    let x = v * 747796405u + 2891336453u;
    let w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}

// Três números ao acaso (0..1) a partir de três inteiros. CADA saída depende
// das TRÊS entradas: na versão anterior cada uma só misturava duas, e a que
// escolhia o ponto da lente dependia do píxel mas não do frame, por isso
// repetia-se sempre e o desfoque nunca limpava.
fn hash3(p: vec3<u32>) -> vec3<f32> {
    let a = pcg(pcg(pcg(p.x) ^ p.y) ^ p.z);
    let b = pcg(a ^ 0x9E3779B9u);
    let c = pcg(b ^ 0x7F4A7C15u);
    return vec3<f32>(vec3<u32>(a, b, c)) / 4294967295.0;
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
    var sum = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var j = -4; j <= 4; j++) {
        for (var i = -4; i <= 4; i++) {
            let o = vec2<f32>(f32(i), f32(j)) * 0.25;
            let w = exp(-2.2 * dot(o, o));
            // A presença é uma média; o cimo das pedras é uma média de
            // potência 4 (puxa para as MAIS ALTAS da vizinhança): com a média
            // simples um bicho ficava abaixo das pedras maiores, metido num
            // buraco escuro entre elas.
            let v = textureSampleLevel(height_tex, samp, uv + o * u.opts.z, 0.0).rg;
            let v2 = v.y * v.y;
            sum += w * vec2<f32>(v.x, v2 * v2);
            wsum += w;
        }
    }
    return vec4<f32>(sum.x / wsum, sqrt(sqrt(sum.y / wsum)), 0.0, 1.0);
}

// APOIO dos agentes: o cimo das pedras, desfocado (verde da textura do chão).
// Um bicho em cima de entulho assenta nesse relevo suave em vez de ficar
// metido entre as pedras.
fn support_at(xy: vec2<f32>) -> f32 {
    let uv = clamp(region_uv(xy), vec2<f32>(0.0), vec2<f32>(1.0));
    return 1.15 * textureSampleLevel(ground_tex, samp, uv, 0.0).g;
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
        if (ta.x > 0.25) { s.a = s.g + support_at(xy) + ta; }
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
        s.a = lerp_volume(vol_tex, uv, s.g + support_at(xy));
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
    // Distância (ao longo do eixo da câmara) do ponto visto: fica no alfa da
    // imagem acumulada, para o autofoco a ler onde se clica.
    var depth = u.lens.x;
    if (dir.z < -1e-4) {
        // Do topo do relevo até ao fundo, em passos; ao entrar numa peça ou
        // no chão, afina por bisseção.
        let t0 = max((u.region.w - org.z) / dir.z, 0.0);
        let t1 = (0.0 - org.z) / dir.z;
        let dt = (t1 - t0) / 420.0;
        var t = t0 + dt * rnd.z;
        var prev_t = t0;
        for (var i = 0; i < 420; i++) {
            let p = org + dir * t;
            // O teste barato (texel a texel) só conta se o interpolado, que é
            // o que dá a superfície final, concordar: senão ficavam pontinhos
            // fixos nas bordas das peças, que a acumulação não limpava.
            if (solid(p, surf(p.xy)) && solid(p, surf_smooth(p.xy))) { break; }
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
        depth = dot(p - u.eye.xyz, u.fwd.xyz);
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
        // (A parte de BAIXO de uma peça só se vê de raspão, e por isso ficava
        // toda com o brilho de aresta, mais clara do que a de cima: aí quase
        // não há brilho de aresta, e fica na sombra da própria peça.)
        let under = n.z < 0.0;
        let edge = pow(1.0 - ndv, 2.0) * select(1.0, 0.15, under);
        // ...e os sítios encaixados entre vizinhos mais altos soltam menos
        // (oclusão): olha-se à volta, a duas distâncias.
        var occ = 0.0;
        for (var k = 0; k < 8; k++) {
            let a = 0.785398 * f32(k) + 6.2831853 * rnd2.z;
            let rr = select(9.0, 22.0, (k & 1) == 1);
            occ += clamp((top_at(p.xy + vec2<f32>(cos(a), sin(a)) * rr) - p.z) / rr, 0.0, 1.5);
        }
        var ao = 1.0 / (1.0 + 0.8 * occ);
        // Na metade de BAIXO de uma peça isto contava o cimo da própria peça
        // como coisa por cima, e ficava uma faixa preta logo abaixo do
        // equador de cada bola. Aí a oclusão é só a proximidade do chão.
        if (under) { ao = mix(0.18, 0.45, clamp((p.z - s.g) / 12.0, 0.0, 1.0)); }
        // PRETO E BRANCO, como uma micrografia: fica só o claro-escuro das
        // peças (opts.x repõe a cor).
        let gray = vec3<f32>(dot(albedo, vec3<f32>(0.33, 0.45, 0.22)) * 1.35);
        // COR FALSA (opts.x = 1): a micrografia continua a ser o cinzento, e
        // só os agentes levam por cima o TOM da sua cor (saturado), como nas
        // imagens de microscópio eletrónico coloridas depois.
        let lum = max(max(albedo.r, albedo.g), albedo.b);
        let hue = pow(albedo / max(lum, 1e-3), vec3<f32>(1.6));
        var tinted = gray;
        if (u.opts.x > 1.5) { tinted = albedo; } else if (u.opts.x > 0.5 && on_agent) { tinted = gray * 1.25 * hue; }
        let base = max(tinted, vec3<f32>(0.03, 0.031, 0.034));
        col = base * (0.95 + 1.5 * edge) * ao + vec3<f32>(0.25) * edge * edge * select(0.0, 1.0, !on_ground);
        // O campo de visão não acaba num quadrado: esbate-se com a distância
        // ao centro da zona e com a distância para lá do plano de focagem.
        // O brilho cai com a DISTÂNCIA ao ponto para onde a câmara olha (uma
        // divisão, sem horizonte marcado), e chega a zero num CÍRCULO antes
        // da borda da zona desenhada, para nunca se ver o quadrado dela.
        let away = length(p.xy - u.region.xy) / u.region.z;
        col /= 1.0 + 7.0 * away * away;
        col *= 1.0 - smoothstep(0.7, 0.98, away);
    }
    // Grão do detetor (desaparece com a acumulação).
    col += (rnd.z - 0.5) * 0.03;
    let prev = textureLoad(prev_tex, vec2<i32>(pos.xy), 0);
    return mix(prev, vec4<f32>(max(col, vec3<f32>(0.0)), depth), u.screen.w);
}

@fragment
fn fs_present(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // SUPERAMOSTRAGEM: a imagem acumulada tem opts.w vezes o lado do ecrã;
    // cada píxel é a média do seu bloco.
    let ss = max(i32(u.opts.w), 1);
    var c = vec3<f32>(0.0);
    for (var j = 0; j < ss; j++) {
        for (var i = 0; i < ss; i++) {
            c += textureLoad(prev_tex, vec2<i32>(pos.xy) * ss + vec2<i32>(i, j), 0).rgb;
        }
    }
    c = c / f32(ss * ss) * u.photo.x;
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
    /// Tamanho do enquadramento (a distância a que uma lente de 33 mm o daria).
    dist: f32,
    aperture: f32,
    /// Tangente de meio campo de visão vertical: pequena = lente longa
    /// (quase axonométrica), grande = grande angular.
    focal: f32,
    /// Plano de focagem: quanto fica para lá (+) ou para cá (−) do ponto para
    /// onde a câmara olha, ao longo do eixo dela (unidades do mundo).
    focus_shift: f32,
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
    /// 0 = preto e branco; 1 = cor falsa (só os agentes, tingidos por cima
    /// do cinzento, como nas micrografias coloridas à mão); 2 = cor inteira.
    colour: u32,
    /// Brilho dos monómeros na zona (0 = não se desenham).
    monomers: f32,
    /// Superamostragem: lado da imagem acumulada em múltiplos do do ecrã.
    ss: u32,
    /// Exposição: multiplica a imagem final.
    exposure: f32,
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
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
        // ORGAN=n: só agentes com este tipo de órgão (11 = protease).
        let need: Option<u16> = std::env::var("ORGAN").ok().and_then(|v| v.parse().ok());
        let mut best: Vec<(usize, usize)> = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.alive != 0 && (14..=48).contains(&a.body_len))
            .map(|(slot, a)| {
                let kinds: std::collections::HashSet<u16> = organs[slot * 64..slot * 64 + a.body_len as usize].iter().filter(|&&c| c != 0).map(|&c| c & 0x1F).collect();
                (kinds.len(), slot, need.is_none_or(|n| kinds.contains(&(n + 1))))
            })
            .filter(|b| b.2)
            .map(|b| (b.0, b.1))
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
            size: 9 * 16,
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
                // (A acumulada é de 32 bits, que não se filtra: lê-se texel a texel.)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
            orbit: Orbit { centre, yaw: 0.6, pitch: 0.75, dist: region / REGION_PER_DIST, aperture: env("APERTURE", 1.5), focal: 12.0 / env("LENS", 135.0), focus_shift: 0.0 },
            last_orbit: None,
            samples: 0,
            frame: 0,
            steps: env("STEPS", 2.0) as u32,
            // Começa PARADO: é assim que a imagem converge e se vê bem.
            paused: true,
            colour: env("COLOR", 0.0) as u32,
            monomers: env("MONOMERS", 0.7),
            ss: 1,
            exposure: env("EXPOSURE", 1.0),
            world,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        self.size = size;
        let (w, h) = (size[0] * self.ss, size[1] * self.ss);
        self.accum = [accum_texture(device, w, h), accum_texture(device, w, h)];
        self.last_orbit = None;
    }

    /// AUTOFOCO: põe o plano de focagem no que se vê no píxel `px` do ecrã
    /// (lê a distância acumulada no alfa da imagem). Devolve se conseguiu.
    fn focus_at(&mut self, gpu: &Gpu, px: [f32; 2]) -> bool {
        let tex = &self.accum[((self.frame + 1) % 2) as usize];
        let x = ((px[0].max(0.0) as u32) * self.ss).min(tex.width() - 1);
        let y = ((px[1].max(0.0) as u32) * self.ss).min(tex.height() - 1);
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("focus"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: None, rows_per_image: None } },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([enc.finish()]);
        buf.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        gpu.wait_idle();
        let depth = bytemuck::pod_read_unaligned::<[f32; 4]>(&buf.get_mapped_range(..).expect("mapped")[..16])[3];
        buf.unmap();
        if !(depth.is_finite() && depth > 1.0) {
            return false;
        }
        let o = &mut self.orbit;
        o.focus_shift = depth - o.dist * REF_TAN / o.focal;
        true
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
        // (Moléculas pequenas: no microscópio são partículas, não manchas.)
        self.cap.view.coc_radius.set(MOLECULE_R);
        self.height_view.coc_radius.set(MOLECULE_R);
        self.cap.view.relief_order.set(true);
        self.height_view.relief_order.set(true);
        self.cap.view.epoch.set(self.world.params.epoch);
        self.cap.view.ghost_steps.set(std::env::var("GHOSTS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0));
        self.cap.encode(&gpu.queue, &mut enc, &cam, 0, self.monomers);
        self.height_view.epoch.set(self.world.params.epoch);
        self.height_view.ghost_steps.set(std::env::var("GHOSTS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0));
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
        let target_pt = [o.centre[0], o.centre[1], 28.0];
        let (sp, cp) = o.pitch.sin_cos();
        let (sy, cy) = o.yaw.sin_cos();
        // Com uma lente mais longa a câmara recua na mesma proporção: o
        // enquadramento fica igual e a perspetiva achata.
        let eye_dist = o.dist * REF_TAN / o.focal;
        let eye = [target_pt[0] + eye_dist * cp * sy, target_pt[1] - eye_dist * cp * cy, target_pt[2] + eye_dist * sp];
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let fwd = norm([target_pt[0] - eye[0], target_pt[1] - eye[1], target_pt[2] - eye[2]]);
        let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        let v4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        let data: [[f32; 4]; 9] = [
            v4(eye),
            v4(right),
            v4(up),
            v4(fwd),
            [o.centre[0], o.centre[1], r, HMAX],
            [(self.size[0] * self.ss) as f32, (self.size[1] * self.ss) as f32, self.frame as f32, weight],
            [(eye_dist + o.focus_shift).max(1.0), o.aperture * REF_TAN / o.focal, o.focal, 1.0],
            [self.colour as f32, GROUND as f32, GROUND_BLUR / (2.0 * r), self.ss as f32],
            [self.exposure, 0.0, 0.0, 0.0],
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

/// Nanómetros por unidade do mundo. CONVENÇÃO (não sai da simulação): um
/// resíduo do corpo (11 unidades) vale 0,5 nm, mais ou menos uma volta de
/// hélice; um órgão fica com 2 a 3 nm, o tamanho de um domínio de proteína.
const NM_PER_UNIT: f32 = 0.5 / 11.0;

fn nm_text(nm: f32) -> String {
    if nm >= 1000.0 {
        format!("{:.2} µm", nm / 1000.0)
    } else if nm >= 10.0 {
        format!("{nm:.0} nm")
    } else {
        format!("{nm:.1} nm")
    }
}

/// O que a interface pede que só se pode fazer fora do desenho dela.
#[derive(Default)]
struct Asked {
    supersampling: bool,
}

/// INTERFACE de microscópio eletrónico: a barra de dados por baixo da imagem
/// (campo, ampliação, distância de trabalho, lente, inclinação, amostras e a
/// escala em nanómetros), a marca do último ponto focado e o painel de
/// controlos (Tab esconde-o).
fn interface(root: &mut egui::Ui, s: &mut Scope, panel: &mut bool, marker: Option<[f32; 2]>) -> Asked {
    let mut asked = Asked::default();
    let ctx = root.ctx().clone();
    let ppp = ctx.pixels_per_point();
    let screen = root.max_rect();
    let o = s.orbit;
    let eye_dist = o.dist * REF_TAN / o.focal;
    let wd = (eye_dist + o.focus_shift).max(1.0);
    let aspect = s.size[0] as f32 / s.size[1].max(1) as f32;
    // Largura do campo no plano de focagem, em nanómetros.
    let hfw = 2.0 * wd * o.focal * aspect * NM_PER_UNIT;
    // Ampliação como nos microscópios: tamanho no ecrã (a 96 pontos por
    // polegada, 0,2646 mm cada) sobre o tamanho real.
    let mag = screen.width() * 0.2646e6 / hfw.max(1e-6);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("barra de dados")));
    let bar_h = 44.0;
    let bar = egui::Rect::from_min_max(egui::pos2(screen.left(), screen.bottom() - bar_h), screen.right_bottom());
    painter.rect_filled(bar, 0.0, egui::Color32::from_black_alpha(235));
    painter.line_segment([bar.left_top(), bar.right_top()], egui::Stroke::new(1.0, egui::Color32::from_gray(150)));
    let dim = egui::Color32::from_gray(140);
    let bright = egui::Color32::from_gray(235);
    let mag_text = if mag >= 1e6 { format!("{:.2} M×", mag / 1e6) } else { format!("{:.0} k×", mag / 1e3) };
    let fields: [(&str, String); 9] = [
        ("HFW", nm_text(hfw)),
        ("Mag", mag_text),
        ("WD", nm_text(wd * NM_PER_UNIT)),
        ("Lens", format!("{:.0} mm", 12.0 / o.focal)),
        ("Aperture", format!("{:.1}", o.aperture)),
        ("Tilt", format!("{:.0}°", 90.0 - o.pitch.to_degrees())),
        ("Exposure", format!("{:.2}", s.exposure)),
        ("Samples", format!("{}{}", s.samples, if s.ss > 1 { " ×4" } else { "" })),
        ("Det", (["SE", "SE false colour", "colour"][s.colour.min(2) as usize]).to_string()),
    ];
    let mut x = bar.left() + 14.0;
    for (label, value) in fields {
        painter.text(egui::pos2(x, bar.top() + 6.0), egui::Align2::LEFT_TOP, label, egui::FontId::monospace(10.0), dim);
        let r = painter.text(egui::pos2(x, bar.top() + 20.0), egui::Align2::LEFT_TOP, value, egui::FontId::monospace(14.0), bright);
        x += r.width().max(48.0) + 22.0;
    }
    // ESCALA: um comprimento redondo (1, 2 ou 5 × 10ⁿ nm) perto de 1/6 do campo.
    let want = hfw / 6.0;
    let pow = 10f32.powf(want.max(1e-6).log10().floor());
    let nice = [1.0, 2.0, 5.0, 10.0].into_iter().map(|k| k * pow).rfind(|&v| v <= want * 1.2).unwrap_or(pow);
    let len = nice / hfw * screen.width();
    let right = bar.right() - 16.0;
    let y = bar.top() + 30.0;
    let white = egui::Stroke::new(2.0, egui::Color32::WHITE);
    painter.line_segment([egui::pos2(right - len, y), egui::pos2(right, y)], white);
    painter.line_segment([egui::pos2(right - len, y - 5.0), egui::pos2(right - len, y + 5.0)], white);
    painter.line_segment([egui::pos2(right, y - 5.0), egui::pos2(right, y + 5.0)], white);
    painter.text(egui::pos2(right - 0.5 * len, y - 7.0), egui::Align2::CENTER_BOTTOM, nm_text(nice), egui::FontId::monospace(13.0), egui::Color32::WHITE);
    // Marca do ponto focado.
    if let Some(m) = marker {
        let c = egui::pos2(m[0] / ppp, m[1] / ppp);
        let st = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 235, 120));
        painter.circle_stroke(c, 9.0, st);
        for d in [egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0), egui::vec2(0.0, 1.0), egui::vec2(0.0, -1.0)] {
            painter.line_segment([c + d * 13.0, c + d * 20.0], st);
        }
    }
    egui::Window::new("Microscope").open(panel).anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0]).resizable(false).show(&ctx, |ui| {
        let mut mm = 12.0 / s.orbit.focal;
        if ui.add(egui::Slider::new(&mut mm, 14.0..=800.0).logarithmic(true).suffix(" mm").text("Lens")).changed() {
            s.orbit.focal = 12.0 / mm;
        }
        ui.add(egui::Slider::new(&mut s.orbit.aperture, 0.0..=30.0).text("Aperture"));
        let mut focus = wd * NM_PER_UNIT;
        let far = (2.5 * eye_dist * NM_PER_UNIT).max(1.0);
        if ui.add(egui::Slider::new(&mut focus, 0.3 * eye_dist * NM_PER_UNIT..=far).suffix(" nm").text("Focus")).changed() {
            s.orbit.focus_shift = focus / NM_PER_UNIT - eye_dist;
        }
        ui.add(egui::Slider::new(&mut s.exposure, 0.05..=16.0).logarithmic(true).text("Exposure"));
        ui.horizontal(|ui| {
            for (k, name) in ["Grey", "False colour", "Colour"].into_iter().enumerate() {
                if ui.selectable_label(s.colour == k as u32, name).clicked() {
                    s.colour = k as u32;
                    s.last_orbit = None;
                }
            }
        });
        ui.horizontal(|ui| {
            let mut mono = s.monomers > 0.0;
            if ui.checkbox(&mut mono, "Monomers").changed() {
                s.monomers = if mono { 0.7 } else { 0.0 };
                s.last_orbit = None;
            }
            let mut ss = s.ss > 1;
            if ui.checkbox(&mut ss, "Supersampling").changed() {
                asked.supersampling = true;
            }
        });
        ui.horizontal(|ui| {
            if ui.button(if s.paused { "▶ Run" } else { "⏸ Pause" }).clicked() {
                s.paused = !s.paused;
                s.last_orbit = None;
            }
            if ui.button("Focus on centre").clicked() {
                s.orbit.focus_shift = 0.0;
            }
        });
        ui.label(egui::RichText::new("Click: focus there · drag: orbit · right drag: move\nwheel: zoom · Tab: hide this panel").small().weak());
    });
    asked
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
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// Painel de controlos à vista (Tab).
    panel: bool,
    /// Onde o botão esquerdo desceu (um clique sem arrastar foca ali).
    pressed_at: Option<[f32; 2]>,
    /// Último ponto focado e quando, para a marca.
    focused: Option<([f32; 2], std::time::Instant)>,
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
        let egui_state = egui_winit::State::new(egui::Context::default(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(gpu.device.limits().max_texture_dimension_2d as usize));
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        Self { window, surface, surface_cfg, gpu, scope, cursor: [0.0; 2], orbiting: false, panning: false, egui_state, egui_renderer, panel: true, pressed_at: None, focused: None }
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
                "Ribossome: microscope   {} samples{}   {}   lens {:.0} mm   aperture {:.1}   exposure {:.2}   (click: focus, drag: orbit, right drag: move, wheel: zoom, space: run/pause, Tab: panel, Z/X: lens, F/G: aperture, E/R: exposure, C: colour, M: monomers, S: supersampling)",
                s.samples,
                if s.ss > 1 { " ×4 (supersampled)" } else { "" },
                if s.paused || s.steps == 0 { "paused" } else { "running" },
                12.0 / s.orbit.focal,
                s.orbit.aperture,
                s.exposure
            ));
        }
        // INTERFACE por cima da imagem.
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let marker = self.focused.filter(|(_, t)| t.elapsed().as_secs_f32() < 1.2).map(|(p, _)| p);
        let mut asked = Asked::default();
        let mut out = ctx.run_ui(raw, |root| asked = interface(root, &mut self.scope, &mut self.panel, marker));
        if asked.supersampling {
            self.scope.ss = if self.scope.ss >= 2 { 1 } else { 2 };
            let size = self.scope.size;
            self.scope.resize(&self.gpu.device, size);
        }
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height], pixels_per_point: out.pixels_per_point };
        for (id, deltas) in out.textures_delta.set.drain() {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, id, &delta);
            }
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut cmds = self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut enc, &jobs, &sd);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("interface"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &jobs, &sd);
        }
        cmds.push(enc.finish());
        self.gpu.queue.submit(cmds);
        self.window.pre_present_notify();
        self.gpu.queue.present(tex);
        for id in out.textures_delta.free.drain() {
            self.egui_renderer.free_texture(&id);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        // A interface vê primeiro: o que for dela (rato sobre o painel,
        // teclas num campo) não mexe na câmara.
        let _ = self.egui_state.on_window_event(&self.window, &event);
        let ctx = self.egui_state.egui_ctx().clone();
        let over_ui = ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui();
        let pressing = matches!(event, WindowEvent::MouseInput { state: ElementState::Pressed, .. }) || matches!(event, WindowEvent::MouseWheel { .. });
        let typing = matches!(event, WindowEvent::KeyboardInput { .. }) && ctx.egui_wants_keyboard_input();
        if (over_ui && pressing) || typing {
            return;
        }
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
                    MouseButton::Left => {
                        self.orbiting = down;
                        if down {
                            self.pressed_at = Some(self.cursor);
                        } else if let Some(p) = self.pressed_at.take() {
                            // Clique sem arrastar: AUTOFOCO nesse ponto.
                            let moved = (self.cursor[0] - p[0]).hypot(self.cursor[1] - p[1]);
                            if moved < 4.0 && self.scope.focus_at(&self.gpu, self.cursor) {
                                self.focused = Some((self.cursor, std::time::Instant::now()));
                            }
                        }
                    }
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
                o.dist = (o.dist * 0.9f32.powf(lines)).clamp(25.0, 5000.0);
                self.scope.region = (o.dist * REGION_PER_DIST).clamp(30.0, 4000.0);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Tab) => self.panel = !self.panel,
                Key::Named(NamedKey::Space) => {
                    self.scope.paused = !self.scope.paused;
                    self.scope.last_orbit = None;
                }
                Key::Character("f") => self.scope.orbit.aperture = (self.scope.orbit.aperture - 1.0).max(0.0),
                Key::Character("g") => self.scope.orbit.aperture = (self.scope.orbit.aperture + 1.0).min(30.0),
                Key::Character("c") => {
                    self.scope.colour = (self.scope.colour + 1) % 3;
                    self.scope.last_orbit = None;
                }
                // LENTE: Z alonga (mais axonométrica), X encurta (grande angular).
                Key::Character("z") => self.scope.orbit.focal = (self.scope.orbit.focal / 1.15).max(12.0 / 800.0),
                Key::Character("x") => self.scope.orbit.focal = (self.scope.orbit.focal * 1.15).min(12.0 / 14.0),
                // EXPOSIÇÃO: E clareia, R escurece (não reinicia a acumulação).
                Key::Character("e") => self.scope.exposure = (self.scope.exposure * 1.12).min(16.0),
                Key::Character("r") => self.scope.exposure = (self.scope.exposure / 1.12).max(0.05),
                Key::Character("s") => {
                    self.scope.ss = if self.scope.ss >= 2 { 1 } else { 2 };
                    let size = self.scope.size;
                    self.scope.resize(&self.gpu.device, size);
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
    scope.ss = (std::env::var("SS").ok().and_then(|v| v.parse().ok()).unwrap_or(2u32)).clamp(1, 3);
    scope.resize(&gpu.device, [w, h]);
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    scope.orbit.yaw = env("YAW", scope.orbit.yaw);
    scope.orbit.pitch = env("PITCH", scope.orbit.pitch);
    scope.orbit.dist = env("DIST", scope.orbit.dist);
    if std::env::var("REGION").is_err() {
        scope.region = (scope.orbit.dist * REGION_PER_DIST).clamp(30.0, 4000.0);
    }
    scope.orbit.aperture = env("APERTURE", scope.orbit.aperture);
    scope.orbit.focal = 12.0 / env("LENS", 12.0 / scope.orbit.focal);
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
