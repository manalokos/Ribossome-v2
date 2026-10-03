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

@fragment
fn fs_world(in: VsOut) -> @location(0) vec4<f32> {
    // Píxel -> mundo (y invertido: o ecrã cresce para baixo, o mundo para cima).
    let px = in.pos.xy - 0.5 * vec2<f32>(view.screen_w, view.screen_h);
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
    if (g >= 3u) {
        let rock = vec3<f32>(0.32, 0.29, 0.26) + 0.04 * f32(g % 3u);
        // A rocha soma a luz que lhe CHEGA (a da célula de cima): a
        // superfície fica dourada, o interior não.
        let ly_up = min(ly + 1u, LIGHT_SIZE - 1u);
        let light_up = light_view[ly_up * LIGHT_SIZE + u32(cell_f.x) / LIGHT_DIV];
        let rock_glow = LIGHT_GOLD * sqrt(clamp(light_up, 0.0, 1.0));
        return vec4<f32>(clamp(rock + rock_glow, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    let rubble = vec3<f32>(0.22, 0.20, 0.18) * (0.6 + 0.2 * f32(g));
    let back = select(water, rubble, g > 0u);

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
    let c = mix(back, hue, clamp(inten * view.monomer_brightness, 0.0, 1.0)) + glow;
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
