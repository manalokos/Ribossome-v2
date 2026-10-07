// Vista do mundo: um triângulo de ecrã inteiro; cada píxel lê a célula da
// grelha por baixo dele. Cima no ecrã = +y no mundo.
// Paleta provisória: a paleta do v3 (composite.wgsl) é portada no fim da fase 2.
// Vistas: 1–4 ativados por canal, 5 gastos, 6 terreno, 7 temperatura, 8 UV, 9 fluido.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> chem_view: array<u32>;
@group(0) @binding(2) var<storage, read> gamma_view: array<u32>;
@group(0) @binding(3) var<storage, read> light_view: array<f32>;
@group(0) @binding(4) var<storage, read> temp_view: array<f32>;
@group(0) @binding(5) var<storage, read> velocity_view: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> redox_view: array<f32>;

// Célula do fluido debaixo de uma posição do mundo.
fn fluid_index_at_world(w: vec2<f32>) -> u32 {
    let f = clamp(floor(w / SIM_SIZE * f32(FLUID_SIZE)), vec2<f32>(0.0), vec2<f32>(f32(FLUID_SIZE - 1u)));
    return u32(f.y) * FLUID_SIZE + u32(f.x);
}

// Rampa térmica: preto -> vermelho -> amarelo -> branco.
fn heat_ramp(t: f32) -> vec3<f32> {
    let u = clamp(t, 0.0, 1.0);
    return vec3<f32>(clamp(u * 3.0, 0.0, 1.0), clamp(u * 3.0 - 1.0, 0.0, 1.0), clamp(u * 3.0 - 2.0, 0.0, 1.0));
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// PALETA DO v3 (composite.wgsl). Duas cores de energia: ATIVADOS a dourado
// (carregados, vivos), GASTOS no azul complementar, com diferença de
// luminância para se lerem também com daltonismo. O tom vem da FRAÇÃO de
// ativados (não da soma: dourado + azul somados davam cinzento) e o brilho
// da quantidade total.
// Gastos: cinzento visível (a matéria está lá; só não é comida).
const MONOMER_SPENT_COLOR: vec3<f32> = vec3<f32>(0.30, 0.30, 0.32);
const MONOMER_GAMMA: f32 = 0.5;     // alpha_gamma_adjust do v3
const DYE_VIS_GAIN: f32 = 2.0;
const WATER: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
// Brilho dourado da luz UV (somado na vista normal).
const LIGHT_GOLD: vec3<f32> = vec3<f32>(0.30, 0.22, 0.06);

// Teclas 1–4 (só ativados): A vermelho, U amarelo, G verde, C azul.
fn channel_color(ch: u32) -> vec3<f32> {
    switch ch {
        case 0u: { return vec3<f32>(1.0, 0.15, 0.1); }
        case 1u: { return vec3<f32>(1.0, 0.85, 0.1); }
        case 2u: { return vec3<f32>(0.15, 0.9, 0.25); }
        default: { return vec3<f32>(0.2, 0.45, 1.0); }
    }
}

// MONÓMEROS COMO PONTOS (de perto). A simulação só sabe quantos monómeros
// há em cada célula; para o desenho, cada um ganha uma posição própria
// dentro da célula (um hash da célula, do canal, do estado e do seu índice:
// fica no mesmo sítio enquanto lá estiver) e desenha-se como uma mancha gaussiana
// de raio view.coc_radius (o círculo de confusão). A soma das manchas é uma
// DENSIDADE em monómeros por célula, que entra na mesma paleta das contagens:
// com o raio grande dá a névoa de sempre, sem quadrados; com o raio pequeno
// veem-se as moléculas.
// Discos desenhados por (célula, canal, estado); acima disto cada disco
// representa vários monómeros (o total conserva-se).
const DOTS_PER_KIND: u32 = 10u;
// Tamanho de um píxel (em células) abaixo do qual se desenham pontos; entre
// os dois valores faz-se a transição para a cor por célula.
// Densidade (monómeros por célula) no centro de um ponto isolado.
const DOT_PEAK: f32 = 2.0;
// O raio do círculo de confusão em desvios-padrão da gaussiana.
const DOT_SIGMAS: f32 = 2.5;
const DOTS_PIXEL_FULL: f32 = 0.25;
const DOTS_PIXEL_NONE: f32 = 0.6;

fn dot_hash(n: u32) -> u32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

// Posição (0..1)² do monómero k do tipo `kind` (canal·2 + estado) na célula.
fn dot_pos(cell: u32, kind: u32, k: u32) -> vec2<f32> {
    let h = dot_hash(cell * 97u + kind * 8191u + k * 131071u + 1u);
    return vec2<f32>(f32(h & 0xFFFFu), f32(h >> 16u)) / 65536.0;
}

struct Soup {
    act: vec4<f32>,
    spent: vec4<f32>,
}

// Densidade de monómeros (por célula de área) no ponto pc (em células),
// somando os discos das células à volta. A rocha não tem monómeros.
fn soup_at(pc: vec2<f32>, radius: f32) -> Soup {
    var s: Soup;
    s.act = vec4<f32>(0.0);
    s.spent = vec4<f32>(0.0);
    let r = clamp(radius, 0.05, 1.0);
    // Núcleo GAUSSIANO, σ = r / DOT_SIGMAS, cortado em r e descido para
    // acabar em zero aí (sem degrau na borda). O integral sobre o disco é
    // 0,8611·r², por isso 1,1613/r² dá integral 1 (um monómero). Com o raio
    // pequeno isso daria um pico enorme e o ponto saturava (um confete de
    // borda dura): limita-se o pico a DOT_PEAK, abaixo da saturação da
    // paleta, para se ver o perfil da gaussiana.
    let norm = min(1.1613 / (r * r), DOT_PEAK);
    let inv_2s2 = 0.5 * DOT_SIGMAS * DOT_SIGMAS / (r * r);
    let floor_g = exp(-0.5 * DOT_SIGMAS * DOT_SIGMAS);
    let lo = vec2<i32>(floor(pc - vec2<f32>(r)));
    let hi = vec2<i32>(floor(pc + vec2<f32>(r)));
    for (var cy = lo.y; cy <= hi.y; cy++) {
        for (var cx = lo.x; cx <= hi.x; cx++) {
            if (cx >= 0 && cy >= 0 && cx < i32(GRID_SIZE) && cy < i32(GRID_SIZE)) {
                let cell = u32(cy) * GRID_SIZE + u32(cx);
                let origin = vec2<f32>(f32(cx), f32(cy));
                for (var ch = 0u; ch < 4u; ch++) {
                    let v = chem_view[cell * 4u + ch];
                    for (var st = 0u; st < 2u; st++) {
                        let count = select(v >> 16u, v & 0xFFFFu, st == 0u);
                        let shown = min(count, DOTS_PER_KIND);
                        var sum = 0.0;
                        for (var k = 0u; k < shown; k++) {
                            let d = pc - (origin + dot_pos(cell, ch * 2u + st, k));
                            sum += max(exp(-dot(d, d) * inv_2s2) - floor_g, 0.0);
                        }
                        let dens = sum / (1.0 - floor_g) * norm * f32(count) / f32(max(shown, 1u));
                        if (st == 0u) { s.act[ch] += dens; } else { s.spent[ch] += dens; }
                    }
                }
            }
        }
    }
    return s;
}

// TERRENO EM GRÃOS (como os monómeros, para não se verem as células). A
// simulação só sabe os quanta de terreno de cada célula; para o desenho cada
// célula ganha 4 grãos, um por quadrante, desviados por um hash fixo.
//   rocha (>= 3): manchas gaussianas largas que somam ~1 no interior; a
//     borda é a curva de nível 0,5 dessa soma (arredondada, sem degraus);
//   entulho (1–2): 2 ou 4 seixos pequenos e separados (é poroso).
// Só o DESENHO: as colisões e a luz continuam a ser por célula.
const GRAIN_PIXEL_FULL: f32 = 0.5;
const GRAIN_PIXEL_NONE: f32 = 1.0;
const ROCK_SIGMA: f32 = 0.5;
const PEBBLE_SIGMA: f32 = 0.17;

struct Ground {
    rock: f32,
    pebble: f32,
    // Tom (0..1) da rocha e do entulho ali, média dos grãos vizinhos.
    tone: f32,
}

fn ground_at(pc: vec2<f32>) -> Ground {
    var rock = 0.0;
    var pebble = 0.0;
    var tone = 0.0;
    var weight = 1e-5;
    let base = vec2<i32>(floor(pc));
    let rock_k = 0.5 / (ROCK_SIGMA * ROCK_SIGMA);
    let rock_norm = 0.25 / (6.2831853 * ROCK_SIGMA * ROCK_SIGMA);
    let peb_k = 0.5 / (PEBBLE_SIGMA * PEBBLE_SIGMA);
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let c = base + vec2<i32>(ox, oy);
            if (c.x >= 0 && c.y >= 0 && c.x < i32(GRID_SIZE) && c.y < i32(GRID_SIZE)) {
                let cell = u32(c.y) * GRID_SIZE + u32(c.x);
                let g = gamma_view[cell];
                if (g > 0u) {
                    let grains = select(min(g * 2u, 4u), 4u, g >= 3u);
                    for (var k = 0u; k < grains; k++) {
                        let h = dot_hash(cell * 4u + k + 77u);
                        let jit = vec2<f32>(f32(h & 0xFFu), f32((h >> 8u) & 0xFFu)) / 255.0 - 0.5;
                        // Quadrante k (a ordem roda com a célula para os
                        // seixos do entulho não caírem sempre no mesmo canto).
                        let q = (k + (h >> 20u)) % 4u;
                        let home = vec2<f32>(0.25 + 0.5 * f32(q & 1u), 0.25 + 0.5 * f32(q >> 1u));
                        let d = pc - (vec2<f32>(c) + home + jit * select(0.3, 0.2, g >= 3u));
                        let d2 = dot(d, d);
                        let t = f32((h >> 16u) & 0xFu) / 15.0;
                        if (g >= 3u) {
                            let w = exp(-d2 * rock_k) * rock_norm;
                            rock += w;
                            tone += w * t;
                            weight += w;
                        } else {
                            let size = 0.7 + 0.6 * t;
                            let w = exp(-d2 * peb_k / (size * size));
                            pebble = max(pebble, w);
                            tone += w * 0.05 * t;
                            weight += w * 0.05;
                        }
                    }
                }
            }
        }
    }
    var out: Ground;
    out.rock = rock;
    out.pebble = pebble;
    out.tone = tone / weight;
    return out;
}

@fragment
fn fs_world(in: VsOut) -> @location(0) vec4<f32> {
    // Píxel -> mundo (y invertido: o ecrã cresce para baixo, o mundo para cima).
    let px = in.pos.xy - vec2<f32>(view.origin_x, view.origin_y) - 0.5 * vec2<f32>(view.screen_w, view.screen_h);
    let world = vec2<f32>(view.center_x + px.x / view.zoom, view.center_y - px.y / view.zoom);
    // Mundo -> célula, explicitamente.
    let cell_f = floor(world / f32(WORLD_UNITS_PER_CELL));
    if (cell_f.x < 0.0 || cell_f.y < 0.0 || cell_f.x >= f32(GRID_SIZE) || cell_f.y >= f32(GRID_SIZE)) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }
    let idx = u32(cell_f.y) * GRID_SIZE + u32(cell_f.x);

    var act = vec4<f32>(0.0);
    var spent = vec4<f32>(0.0);
    for (var ch = 0u; ch < 4u; ch++) {
        let v = chem_view[idx * 4u + ch];
        act[ch] = f32(v & 0xFFFFu);
        spent[ch] = f32(v >> 16u);
    }
    // De perto: monómeros como pontos (ver soup_at). pixel_cells = tamanho
    // de um píxel em células.
    let pixel_cells = 1.0 / (view.zoom * f32(WORLD_UNITS_PER_CELL));
    let dots = (1.0 - smoothstep(DOTS_PIXEL_FULL, DOTS_PIXEL_NONE, pixel_cells)) * step(1e-4, view.coc_radius);
    if (dots > 0.0 && view.view_mode <= 5u) {
        let soup = soup_at(world / f32(WORLD_UNITS_PER_CELL), view.coc_radius);
        act = mix(act, soup.act, dots);
        spent = mix(spent, soup.spent, dots);
    }
    // LUZ como SOMA DOURADA: onde chega luz soma-se um brilho dourado; a
    // sombra (do terreno e dos agentes) é a falta dele. Não multiplica nada,
    // por isso os monómeros guardam a cor. √luz para se ver mais fundo do que
    // a luz física (que cai exp(-uv_depth) do topo ao fundo).
    // A luz está a 1/LIGHT_DIV da resolução.
    let ly = u32(cell_f.y) / LIGHT_DIV;
    let lidx = ly * LIGHT_SIZE + u32(cell_f.x) / LIGHT_DIV;
    let light_here = light_view[lidx];
    let glow = LIGHT_GOLD * sqrt(clamp(light_here, 0.0, 1.0));
    let water = WATER;
    // Contagens em unidades de 3 quanta, com tone map de Reinhard (como no v3).
    let act_lin = act / 3.0 * DYE_VIS_GAIN;
    let act_tm = act_lin / (vec4<f32>(1.0) + act_lin);

    if (view.view_mode >= 1u && view.view_mode <= 4u) {
        let ch = view.view_mode - 1u;
        return vec4<f32>(clamp(water + channel_color(ch) * act_tm[ch], vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    // 6: terreno (quanta de gamma; rocha >= 3 a branco).
    if (view.view_mode == 6u) {
        let g = f32(gamma_view[idx]);
        return vec4<f32>(mix(vec3<f32>(0.02, 0.03, 0.05), vec3<f32>(0.85, 0.8, 0.7), clamp(g / 3.0, 0.0, 1.0)), 1.0);
    }
    // 7: temperatura (v3): azul frio -> vermelho quente, com a isotérmica de
    // ativação (T = 2) a branco: lá dentro a água reativa monómeros gastos.
    if (view.view_mode == 7u) {
        let t = temp_view[fluid_index_at_world(world)];
        var c = mix(vec3<f32>(0.02, 0.08, 0.35), vec3<f32>(0.95, 0.12, 0.05), clamp(sqrt(t / 12.0), 0.0, 1.0));
        if (abs(t - 2.0) < 0.12) { c = vec3<f32>(0.95); }
        return vec4<f32>(c, 1.0);
    }
    // 8: luz UV (raiz quadrada, para ver o fundo).
    if (view.view_mode == 8u) {
        let l = sqrt(clamp(light_here, 0.0, 1.0));
        return vec4<f32>(vec3<f32>(0.75, 0.6, 1.0) * l, 1.0);
    }
    // 9: velocidade do fluido (|v| em células/s; escala 0..50) e direção no tom.
    if (view.view_mode == 9u) {
        let v = velocity_view[fluid_index_at_world(world)];
        let sp = clamp(length(v) / 50.0, 0.0, 1.0);
        let dir = select(vec2<f32>(0.0), normalize(v), length(v) > 1e-6);
        let hue = vec3<f32>(0.5 + 0.5 * dir.x, 0.5 + 0.5 * dir.y, 0.6);
        return vec4<f32>(hue * sqrt(sp), 1.0);
    }
    // 10: redutor das fumarolas (raiz quadrada; amarelo-enxofre).
    if (view.view_mode == 10u) {
        let r = redox_view[fluid_index_at_world(world)];
        return vec4<f32>(vec3<f32>(0.95, 0.85, 0.2) * clamp(sqrt(r / 5.0), 0.0, 1.0) + water * 0.3, 1.0);
    }
    if (view.view_mode == 5u) {
        let t = clamp(dot(spent, vec4<f32>(1.0)) / 24.0, 0.0, 1.0);
        return vec4<f32>(mix(water, vec3<f32>(0.75), t), 1.0);
    }

    // Terreno: rocha (>= 3) opaca; o entulho (1–2) é poroso e é o FUNDO por
    // trás dos monómeros que lá estão (desenhados por cima, mais abaixo).
    let g = gamma_view[idx];
    var rock_m = select(0.0, 1.0, g >= 3u);
    var rubble_m = select(0.0, 1.0, g > 0u && g < 3u);
    var tone = 0.5 * f32(g % 3u);
    // De perto, o terreno em grãos (ver ground_at) em vez de células.
    let grains = 1.0 - smoothstep(GRAIN_PIXEL_FULL, GRAIN_PIXEL_NONE, pixel_cells);
    if (grains > 0.0) {
        let gr = ground_at(world / f32(WORLD_UNITS_PER_CELL));
        rock_m = mix(rock_m, smoothstep(0.40, 0.56, gr.rock), grains);
        rubble_m = mix(rubble_m, smoothstep(0.3, 0.55, gr.pebble), grains);
        tone = mix(tone, gr.tone, grains);
    }
    let rock = vec3<f32>(0.30, 0.27, 0.24) + 0.08 * tone;
    // A rocha soma a luz que lhe CHEGA (a da célula de cima): a
    // superfície fica dourada, o interior não.
    let ly_up = min(ly + 1u, LIGHT_SIZE - 1u);
    let light_up = light_view[ly_up * LIGHT_SIZE + u32(cell_f.x) / LIGHT_DIV];
    let rock_col = rock + LIGHT_GOLD * sqrt(clamp(light_up, 0.0, 1.0));
    let rubble = vec3<f32>(0.22, 0.20, 0.18) * (0.75 + 0.3 * tone);
    let back = mix(water, rubble, rubble_m);

    // Normal: os ATIVADOS têm a média das cores dos seus canais (A vermelho,
    // U amarelo, G verde, C azul, pesada pelas contagens); os GASTOS são
    // cinzento escuro. Tom pela fração de ativados, brilho pela quantidade.
    let act_amt = clamp(dot(act_tm, vec4<f32>(1.0)), 0.0, 1.5);
    let spent_amt = clamp(dot(spent, vec4<f32>(1.0)) / 3.0, 0.0, 1.5);
    let total_amt = act_amt + spent_amt;
    var act_col = vec3<f32>(0.0);
    for (var ch = 0u; ch < 4u; ch++) { act_col += channel_color(ch) * act[ch]; }
    act_col /= max(dot(act, vec4<f32>(1.0)), 1e-5);
    // Satura: a média de 4 cores tende para bege; normalizar pelo canal mais
    // forte dá a cor dominante com brilho total (contraste com os gastos).
    act_col /= max(max(act_col.r, max(act_col.g, act_col.b)), 1e-5);
    let act_frac = act_amt / max(total_amt, 1e-5);
    let hue = mix(MONOMER_SPENT_COLOR, act_col, act_frac);
    let inten = pow(clamp(total_amt, 0.0, 1.0), MONOMER_GAMMA);
    // A noite vê-se só na camada da luz (glow), que já vem escura do topo.
    var c = mix(back, hue, clamp(inten * view.monomer_brightness, 0.0, 1.0)) + glow;
    c = mix(c, rock_col, rock_m);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
