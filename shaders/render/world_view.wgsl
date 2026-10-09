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
// Atlas de sprites (o mesmo dos agentes, ver agents_view.wgsl): aqui usam-se
// as linhas dos grãos de entulho e dos blocos de rocha.
// MAPA DE PARENTESCO (cor dos agentes = parentesco com o selecionado): em
// vez de uma bola por agente, cada ponto do mundo fica com a cor do agente
// MAIS PRÓXIMO (um diagrama de Voronoi), até KIN_REACH de distância. Os
// agentes procuram-se na grelha de contacto da simulação (células de
// KIN_CELL unidades, uma lista ligada por célula: contact.wgsl).
@group(0) @binding(9) var<storage, read> kin_agents: array<Agent>;
@group(0) @binding(10) var<storage, read> kin_head: array<u32>;
@group(0) @binding(11) var<storage, read> kin_next: array<u32>;
@group(0) @binding(12) var<storage, read> kin_value: array<f32>;
const KIN_CELL: f32 = 120.0;
const KIN_N: u32 = u32(SIM_SIZE / KIN_CELL) + 1u;
const KIN_REACH: f32 = 230.0;
// Cor e opacidade do mapa no ponto w (mundo). px = unidades do mundo por píxel.
fn kin_map(w: vec2<f32>, px: f32) -> vec4<f32> {
    let c = vec2<i32>(floor(w / KIN_CELL));
    var best = 1e30;
    var second = 1e30;
    var q = -1.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let x = c.x + dx;
            let y = c.y + dy;
            if (x >= 0 && y >= 0 && x < i32(KIN_N) && y < i32(KIN_N)) {
                var e = kin_head[u32(y) * KIN_N + u32(x)];
                for (var guard = 0u; guard < 48u && e != 0xFFFFFFFFu; guard++) {
                    let a = kin_agents[e];
                    let k = kin_value[e];
                    if (a.alive != 0u && k >= 0.0) {
                        let d = w - vec2<f32>(a.pos_x, a.pos_y);
                        let d2 = dot(d, d);
                        if (d2 < best) {
                            second = best;
                            best = d2;
                            q = k;
                        } else if (d2 < second) {
                            second = d2;
                        }
                    }
                    e = kin_next[e];
                }
            }
        }
    }
    if (q < 0.0) { return vec4<f32>(0.0); }
    let d1 = sqrt(best);
    // Verde = genoma próximo do selecionado, amarelo, vermelho = distante
    // (a mesma rampa das bolas que isto substitui).
    let t = clamp(q, 0.0, 1.0);
    let col = select(mix(vec3<f32>(1.0, 0.85, 0.1), vec3<f32>(0.15, 1.0, 0.25), (t - 0.5) * 2.0),
                     mix(vec3<f32>(1.0, 0.12, 0.08), vec3<f32>(1.0, 0.85, 0.1), t * 2.0), t < 0.5);
    // Fronteira entre duas células (onde os dois mais próximos estão à mesma
    // distância): uma linha escura fina.
    let edge = smoothstep(0.0, max(2.5 * px, 2.0), sqrt(second) - d1);
    let fade = 1.0 - smoothstep(0.75 * KIN_REACH, KIN_REACH, d1);
    return vec4<f32>(col * (0.3 + 0.35 * edge), 0.85 * fade);
}
// Fontes das fumarolas (x = calor, y = química), na grelha do fluido: só
// para as marcar na vista enquanto se pintam (view.show_vents).
@group(0) @binding(13) var<storage, read> vent_src: array<vec2<f32>>;
@group(0) @binding(7) var sprites_tex: texture_2d<f32>;
@group(0) @binding(8) var sprites_samp: sampler;
const SPRITE_COLS: f32 = 9.0;
const SPRITE_ROWS: f32 = 32.0;
// Monómeros como moléculas: linha dos ativados (com a cauda de três fosfatos)
// e dos gastos (um fosfato); a coluna é o canal (A, U, G, C).
const SPRITE_ROW_MONOMER: f32 = 27.0;
const SPRITE_ROW_PEBBLE: f32 = 25.0;
const SPRITE_ROW_ROCK: f32 = 26.0;
// Raio (células) do sprite de um grão de entulho (vezes o seu tamanho) e de
// um bloco de rocha: maiores do que o espaço entre grãos, para se
// sobreporem (o de cima tapa o de baixo; fora da máscara vê-se o de baixo).
// Sombra de contacto entre grãos (ver SHADOW em agents_view.wgsl).
const GRAIN_SHADOW: f32 = 0.55;
const PEBBLE_SPRITE_R: f32 = 0.36;
const ROCK_SPRITE_R: f32 = 0.72;
// (luminância, máscara) do grão: d = posição relativa ao centro (células),
// r = raio, h = hash do grão (variante e rotação), px = píxel em células.
fn grain_sprite(row: f32, d: vec2<f32>, r: f32, h: u32, px: f32) -> vec3<f32> {
    let ang = f32((h >> 4u) & 0xFFu) * (6.2831853 / 256.0);
    let cs = cos(ang);
    let sn = sin(ang);
    let q = vec2<f32>(d.x * cs - d.y * sn, d.x * sn + d.y * cs) / r;
    let sc = vec2<f32>(0.5 / SPRITE_COLS, -0.5 / SPRITE_ROWS);
    let g = px / r;
    let s = textureSampleGrad(sprites_tex, sprites_samp, vec2<f32>((f32((h >> 12u) % 9u) + 0.5) / SPRITE_COLS, (row + 0.5) / SPRITE_ROWS) + clamp(q, vec2<f32>(-0.98), vec2<f32>(0.98)) * sc, vec2<f32>(g, 0.0) * sc, vec2<f32>(0.0, g) * sc);
    return vec3<f32>(s.r, s.g, s.b);
}

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
    // Relevo das moléculas que cobrem o ponto: soma da luminância dos
    // sprites e quantos são (a média dá o sombreado, à parte da quantidade).
    lum: f32,
    cover: f32,
    // Altura (em células) da molécula mais alta que cobre o ponto, e a que
    // cota (0..1, ao acaso por molécula) ela paira: no microscópio 3D as
    // moléculas não ficam todas no mesmo plano.
    h: f32,
    lift: f32,
    top: f32,
}

// Densidade de monómeros (por célula de área) no ponto pc (em células),
// somando os discos das células à volta. A rocha não tem monómeros.
fn soup_at(pc: vec2<f32>, radius: f32, px: f32) -> Soup {
    var s: Soup;
    s.act = vec4<f32>(0.0);
    s.spent = vec4<f32>(0.0);
    s.lum = 0.0;
    s.cover = 0.0;
    s.h = 0.0;
    s.lift = 0.0;
    s.top = 0.0;
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
    // (Mais a folga do tremor: uma molécula pode entrar no raio vinda da
    // célula ao lado.)
    let lo = vec2<i32>(floor(pc - vec2<f32>(r + 1.6 * MOL_JITTER)));
    let hi = vec2<i32>(floor(pc + vec2<f32>(r + 1.6 * MOL_JITTER)));
    // Relógio do tremor (passos da simulação: parada, as moléculas param).
    let jt = (f32(view.epoch % 1048576u) + view.clock_frac) * MOL_JITTER_RATE;
    let jseg = u32(jt);
    let jf = jt - f32(jseg);
    let jmix = jf * jf * (3.0 - 2.0 * jf);
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
                            // TREMOR TÉRMICO (só desenho): cada molécula oscila
                            // à volta do seu ponto, com a sua fase, menos do que
                            // o salto mínimo real (uma célula), e roda um pouco.
                            let h = dot_hash(cell * 131u + (ch * 2u + st) * 7919u + k * 104729u + 5u);
                            // (Um passeio ao acaso: de MOL_JITTER_RATE em
                            // MOL_JITTER_RATE cada molécula sorteia um novo
                            // desvio e uma nova rotação, e desliza do anterior
                            // para esse. Não se repete, ao contrário de uma onda.)
                            let j0 = dot_hash(h + jseg * 2654435761u);
                            let j1 = dot_hash(h + (jseg + 1u) * 2654435761u);
                            let w0 = vec3<f32>(f32(j0 & 0x3FFu), f32((j0 >> 10u) & 0x3FFu), f32((j0 >> 20u) & 0x3FFu)) / 511.5 - 1.0;
                            let w1 = vec3<f32>(f32(j1 & 0x3FFu), f32((j1 >> 10u) & 0x3FFu), f32((j1 >> 20u) & 0x3FFu)) / 511.5 - 1.0;
                            let wob3 = mix(w0, w1, jmix);
                            let wob = wob3.xy;
                            let d = pc - (origin + dot_pos(cell, ch * 2u + st, k) + MOL_JITTER * wob);
                            // MOLÉCULA: o sprite do nucleótido, rodado por um
                            // hash; pesa pela máscara e pelo relevo.
                            if (dot(d, d) < r * r) {
                                let ang = f32(h & 0xFFFu) * (6.2831853 / 4096.0) + MOL_SPIN * wob3.z;
                                let cs = cos(ang);
                                let sn = sin(ang);
                                let q = vec2<f32>(d.x * cs - d.y * sn, d.x * sn + d.y * cs) / r;
                                let sc = vec2<f32>(0.5 / SPRITE_COLS, -0.5 / SPRITE_ROWS);
                                let g = px / r;
                                let t = textureSampleGrad(sprites_tex, sprites_samp, vec2<f32>((f32(ch) + 0.5) / SPRITE_COLS, (SPRITE_ROW_MONOMER + f32(st) + 0.5) / SPRITE_ROWS) + clamp(q, vec2<f32>(-0.98), vec2<f32>(0.98)) * sc, vec2<f32>(g, 0.0) * sc, vec2<f32>(0.0, g) * sc);
                                // A quantidade conta a máscara inteira; o relevo
                                // (t.r) vai à parte, para não se perder quando a
                                // cor satura.
                                let cov = step(0.5, t.g);
                                sum += cov;
                                s.lum += cov * t.r;
                                s.cover += cov;
                                let lf = f32((h >> 12u) & 0xFFu) / 255.0;
                                let top = cov * (t.b * r * mix(0.45, 1.3, t.r) + MOL_LIFT * lf);
                                if (top > s.top) {
                                    s.top = top;
                                    // (A forma insuflada é um calhau liso; a
                                    // luminância do sprite, onde se veem os
                                    // átomos, dá-lhe as bossas.)
                                    s.h = cov * t.b * r * mix(0.45, 1.3, t.r);
                                    s.lift = lf;
                                }
                            }
                        }
                        let dens = sum * f32(count) / f32(max(shown, 1u));
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

// Tremor das moléculas: amplitude (células, por eixo), rotação (radianos)
// e ritmo (novos sorteios por passo da simulação: 1 = um em cada passo).
const MOL_JITTER: f32 = 0.1;
const MOL_SPIN: f32 = 0.7;
const MOL_JITTER_RATE: f32 = 1.0;
// Altura do relevo de uma molécula no microscópio 3D, em relação à forma
// insuflada do seu sprite.
const MOL_RELIEF: f32 = 2.5;
// Até onde (em células) uma molécula solta paira acima do fundo.
const MOL_LIFT: f32 = 0.075;

struct Ground {
    rock: f32,
    pebble: f32,
    // Tom (0..1) da rocha e do entulho ali, média dos grãos vizinhos.
    tone: f32,
    // Luminância do sprite do grão / bloco que fica por cima (-1 = nenhum).
    pebble_l: f32,
    rock_l: f32,
    // Altura (em células) do seixo de cima naquele ponto: uma cúpula.
    pebble_h: f32,
    rock_h: f32,
    // Raio (em células) desse seixo.
    pebble_r: f32,
    // Sombra de contacto (0..1) de um grão mais alto sobre o que ali está.
    shadow: f32,
}

fn ground_at(pc: vec2<f32>, px: f32) -> Ground {
    // O grão "de cima" é o de maior altura (um hash), para a ordem não
    // depender da célula de onde se olha.
    var pebble_l = -1.0;
    var pebble_z = -1.0;
    var pebble_zh = -1.0;
    var pebble_h = 0.0;
    var rock_h = 0.0;
    var pebble_r = 0.0;
    var rock_l = -1.0;
    var rock_z = -1.0;
    var halo = 0.0;
    var halo_z = -1.0;
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
                            let z = f32(h >> 24u);
                            // No microscópio 3D fica o bloco mais ALTO naquele
                            // ponto (a fronteira é onde os dois se cruzam, sem
                            // parede a pique); na vista normal, um sorteado, que
                            // poupa leituras.
                            let by_height = view.relief != 0u;
                            if (d2 < ROCK_SPRITE_R * ROCK_SPRITE_R && (by_height || z > rock_z)) {
                                let s = grain_sprite(SPRITE_ROW_ROCK, d, ROCK_SPRITE_R, h, px);
                                let key = select(z, 1000.0 * s.z + 0.001 * z, by_height);
                                if (s.y > 0.5 && key > rock_z) {
                                    rock_z = key;
                                    rock_h = ROCK_SPRITE_R * s.z;
                                    rock_l = s.x;
                                }
                            }
                        } else {
                            let size = 0.7 + 0.6 * t;
                            let w = exp(-d2 * peb_k / (size * size));
                            let r = PEBBLE_SPRITE_R * size;
                            let z = f32(h >> 24u);
                            if (d2 < r * r) {
                                let s = grain_sprite(SPRITE_ROW_PEBBLE, d, r, h, px);
                                // Onde dois seixos se sobrepõem, vê-se o que é
                                // MAIS ALTO naquele ponto (e não um sorteado):
                                // a fronteira é a linha onde as duas bolas se
                                // cruzam, sem degrau. No microscópio 3D isto
                                // tira as paredes a pique entre pedras.
                                let zh = r * s.z + 0.0005 * z;
                                if (s.y > 0.5 && zh > pebble_zh) {
                                    pebble_zh = zh;
                                    pebble_z = z;
                                    pebble_l = s.x;
                                    pebble_h = r * s.z;
                                    pebble_r = r;
                                    pebble = 1.0;
                                }
                            }
                            // Auréola deste grão (fica a do mais alto que a tiver).
                            let hx = 1.0 - smoothstep(0.75, 1.3, sqrt(d2) / r);
                            if (hx > 0.0 && z > halo_z) {
                                halo_z = z;
                                halo = hx;
                            }
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
    out.pebble_l = pebble_l;
    out.rock_l = rock_l;
    out.pebble_h = pebble_h;
    out.rock_h = rock_h;
    out.pebble_r = pebble_r;
    // Só escurece se vier de um grão mais alto do que o que ali se vê.
    out.shadow = select(0.0, halo, halo_z > pebble_z);
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
    var mol_relief = 1.0;
    var mol_edge = 0.0;
    var mol_h = 0.0;
    var mol_lift = 0.0;
    if (dots > 0.0 && view.view_mode <= 5u) {
        let soup = soup_at(world / f32(WORLD_UNITS_PER_CELL), view.coc_radius, pixel_cells);
        act = mix(act, soup.act, dots);
        spent = mix(spent, soup.spent, dots);
        // RELEVO das moléculas: escurece os vales e aclara as arestas.
        mol_h = soup.h;
        mol_lift = soup.lift;
        if (soup.cover > 0.0) {
            let l = pow(clamp(soup.lum / soup.cover, 0.0, 1.0), 0.8);
            mol_relief = mix(1.0, 0.15 + 1.6 * l, dots);
            mol_edge = dots * pow(l, 4.0) * 0.5;
        }
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
    var rock_tex = 1.0;
    var rubble_tex = 1.0;
    var pebble_h = 0.3;
    var rock_h = 0.0;
    var pebble_r = 0.3;
    var rock_field = 1.0;
    var ground_shadow = 1.0;
    // De perto, o terreno em grãos (ver ground_at) em vez de células.
    let grains = 1.0 - smoothstep(GRAIN_PIXEL_FULL, GRAIN_PIXEL_NONE, pixel_cells);
    if (grains > 0.0) {
        let gr = ground_at(world / f32(WORLD_UNITS_PER_CELL), pixel_cells);
        rock_m = mix(rock_m, smoothstep(0.40, 0.56, gr.rock), grains);
        rubble_m = mix(rubble_m, gr.pebble, grains);
        tone = mix(tone, gr.tone, grains);
        // Relevo dos sprites (1 = sem textura): escurece os vales e aclara as arestas.
        if (gr.rock_l >= 0.0) { rock_tex = mix(1.0, 0.55 + 1.0 * pow(gr.rock_l, 0.8), grains); }
        if (gr.pebble_l >= 0.0) { rubble_tex = mix(1.0, 0.5 + 1.3 * pow(gr.pebble_l, 0.8), grains); }
        pebble_h = gr.pebble_h;
        rock_h = gr.rock_h;
        pebble_r = gr.pebble_r;
        rock_field = gr.rock;
        ground_shadow = 1.0 - GRAIN_SHADOW * gr.shadow * grains;
    }
    // PARA O MICROSCÓPIO 3D: o VOLUME de cada pedra e de cada molécula (cimo,
    // fundo), em unidades do mundo acima do chão. Um seixo é uma bola meio
    // enterrada; um bloco de rocha sai do chão; uma molécula é um grãozinho
    // a pairar.
    if (view.height_pass != 0u) {
        let cell = f32(WORLD_UNITS_PER_CELL);
        var vol = vec2<f32>(0.0);
        if (rock_m > 0.5) {
            // O contorno da rocha é a curva de nível de um campo suave, e não
            // o do sprite do bloco: numa célula de rocha isolada cortava a
            // cúpula a pique e ficava um cilindro. A altura desce com o campo
            // até ao contorno (um ombro redondo).
            // (Chega a ZERO no contorno, que é onde o campo vale 0,48: com
            // uma sobra, a rocha acabava num degrau a pique contra o chão.)
            let shoulder = sqrt(clamp((rock_field - 0.48) / 0.4, 0.0, 1.0));
            vol = vec2<f32>((2.0 + 0.8 * rock_h * cell) * shoulder + 0.3, -30.0);
        } else if (rubble_m > 0.5) {
            // Um seixo é uma CÚPULA assente no chão (desce a zero no contorno),
            // e não uma bola com o equador no ar: o relevo só guarda uma pedra
            // por ponto, e a aba de uma bola por cima da vizinha mais baixa
            // ficava uma parede fina a pique entre as duas. Com cúpulas, o
            // monte é o máximo delas e é contínuo em todo o lado.
            // (A forma insuflada do sprite ainda vale ~0,14 raios no contorno
            // da máscara: tira-se essa sobra para o seixo chegar ao chão sem
            // um degrauzinho a toda a volta.)
            let rise = max(pebble_h - 0.14 * pebble_r, 0.0) * 1.16;
            vol = vec2<f32>(1.2 * rise * cell + 0.3, -30.0);
        }
        // As MOLÉCULAS vão à parte (azul e alfa), com o seu próprio intervalo
        // acima de onde assentam: o microscópio pousa-as no cimo das pedras
        // ou no relevo suave do entulho. Na mesma camada das pedras, a
        // fronteira entre uma pedra e uma molécula era uma parede a pique, e
        // uma molécula em cima de uma pedra deformava-a ao tremer.
        var mol = vec2<f32>(0.0);
        if (mol_h > 0.0 && view.monomer_brightness > 0.0 && view.height_pass == 1u) {
            let mid = 3.0 + MOL_LIFT * mol_lift * cell;
            // (MOL_RELIEF vezes a forma do sprite: senão os átomos mal se notam.)
            mol = vec2<f32>(mid + MOL_RELIEF * mol_h * cell, mid - MOL_RELIEF * mol_h * cell);
        }
        // Modo 2: quanto terreno há aqui e o cimo das pedras. O microscópio
        // desfoca os dois: o primeiro dá o relevo suave do chão, o segundo a
        // cota a que os agentes assentam (por cima das pedras, não dentro).
        if (view.height_pass == 2u) {
            return vec4<f32>(max(rock_m, 0.4 * rubble_m), max(vol.x, 0.0), 0.0, 1.0);
        }
        return vec4<f32>(vol, mol);
    }
    let rock = (vec3<f32>(0.30, 0.27, 0.24) + 0.08 * tone) * rock_tex;
    // A rocha soma a luz que lhe CHEGA (a da célula de cima): a
    // superfície fica dourada, o interior não.
    let ly_up = min(ly + 1u, LIGHT_SIZE - 1u);
    let light_up = light_view[ly_up * LIGHT_SIZE + u32(cell_f.x) / LIGHT_DIV];
    let rock_col = rock + LIGHT_GOLD * sqrt(clamp(light_up, 0.0, 1.0));
    let rubble = vec3<f32>(0.22, 0.20, 0.18) * (0.75 + 0.3 * tone) * rubble_tex;
    let back = mix(water, rubble, rubble_m) * ground_shadow;

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
    var c = mix(back, clamp(hue * mol_relief + vec3<f32>(mol_edge), vec3<f32>(0.0), vec3<f32>(1.0)), clamp(inten * view.monomer_brightness, 0.0, 1.0)) + glow;
    c = mix(c, rock_col, rock_m);
    if (view.show_vents != 0u) {
        // FUMAROLAS À VISTA (ao pintar): calor a laranja, química a
        // verde-amarelo, mais forte onde a fonte é mais forte; por cima de
        // tudo, rocha incluída.
        let s = vent_src[fluid_index_at_world(world)];
        c = mix(c, vec3<f32>(1.0, 0.45, 0.05), clamp(0.25 + 0.6 * s.x, 0.0, 0.85) * step(1e-4, s.x));
        c = mix(c, vec3<f32>(0.75, 1.0, 0.1), clamp(0.25 + 0.6 * s.y, 0.0, 0.85) * step(1e-4, s.y) * select(1.0, 0.6, s.x > 1e-4));
    }
    if (view.signal_view == 4u) {
        let km = kin_map(world, 1.0 / view.zoom);
        c = mix(c, km.rgb, km.a * (1.0 - rock_m));
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
