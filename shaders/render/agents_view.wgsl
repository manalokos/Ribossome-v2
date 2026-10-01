// Agentes: um quadrado instanciado por RESÍDUO (slot·64 + k), desenhado
// como disco com a cor da classe química (classes do v3). Os órgãos têm
// forma própria (desenhada por SDF no fragmento, orientada pela cadeia):
//   boca        disco com uma abertura escura virada para fora;
//   músculo     elipse ao longo da cadeia, com estrias;
//   sensores    TOTAIS: coroa de 6 antenas; DIRECIONAIS: 2 antenas, uma de
//               cada lado da cadeia (comida verde, luz amarela);
//   energia     disco com anel interior;
//   relógio     mostrador com um ponteiro que roda com o período;
//   relé        losango;
//   armazenamento disco grande com anéis.
// Vista de sinais (view.signal_view): a cor passa a ser o sinal α e/ou β.
// Instâncias sem resíduo viram triângulos degenerados. RNA nu = disco cinzento.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> agents_view: array<Agent>;
@group(0) @binding(2) var<storage, read> bodies_view: array<u32>;
@group(0) @binding(3) var<storage, read> body_pos_view: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> draw_list_view: array<u32>;
@group(0) @binding(5) var<storage, read> organs_view: array<u32>;
@group(0) @binding(6) var<storage, read> signals_view: array<vec2<f32>>;

const MAX_BODY_V: u32 = 64u;
const NO_ORGAN: u32 = 0xFFu;

struct AgentVsOut {
    @builtin(position) pos: vec4<f32>,
    // Coordenadas no quadrado (−1..1), eixos do mundo.
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    // Tipo do órgão (NO_ORGAN se não houver).
    @location(2) @interpolate(flat) organ: u32,
    // Tangente da cadeia (mundo, unitária).
    @location(3) @interpolate(flat) tangent: vec2<f32>,
    // Raio do disco do resíduo em fração do quadrado; ângulo do relógio.
    @location(4) @interpolate(flat) core_phase: vec2<f32>,
};

// Classes (v3): alifáticos A I L M V, aromáticos F W Y, polares S T N Q,
// C à parte, positivos K R H, negativos D E, G e P especiais.
fn class_color(aa: u32) -> vec3<f32> {
    switch aa {
        case 0u, 7u, 9u, 10u, 17u: { return vec3<f32>(0.72, 0.72, 0.62); } // A I L M V
        case 4u, 18u, 19u: { return vec3<f32>(0.70, 0.45, 0.95); }        // F W Y
        case 15u, 16u, 11u, 13u: { return vec3<f32>(0.40, 0.85, 0.45); }  // S T N Q
        case 1u: { return vec3<f32>(0.95, 0.90, 0.30); }                  // C
        case 8u, 14u, 6u: { return vec3<f32>(0.35, 0.55, 1.00); }         // K R H
        case 2u, 3u: { return vec3<f32>(1.00, 0.35, 0.30); }              // D E
        case 5u: { return vec3<f32>(0.95, 0.95, 0.95); }                  // G
        default: { return vec3<f32>(1.00, 0.60, 0.20); }                  // P
    }
}

// Cor de um sinal com sinal: positivo -> `pos`, negativo -> `neg`, zero -> cinzento escuro.
fn signed_color(v: f32, pos: vec3<f32>, neg: vec3<f32>) -> vec3<f32> {
    let t = tanh(abs(v));
    return mix(vec3<f32>(0.18), select(neg, pos, v >= 0.0), t);
}

// Tamanho do quadrado em múltiplos do raio do disco, por tipo de órgão.
fn organ_extent(t: u32) -> f32 {
    switch t {
        case ORGAN_FOOD_SENSOR, ORGAN_LIGHT_SENSOR: { return 2.6; }
        case ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: { return 3.4; }
        case ORGAN_MUSCLE: { return 1.6; }
        case ORGAN_STORAGE: { return 1.9; }
        case NO_ORGAN: { return 1.0; }
        default: { return 1.3; }
    }
}

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> AgentVsOut {
    var o: AgentVsOut;
    let slot = draw_list_view[inst / MAX_BODY_V];
    let k = inst % MAX_BODY_V;
    let a = agents_view[slot];
    let naked = a.body_len == 0u;
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot;
    if (a.alive == 0u || hidden || (k >= a.body_len && !(naked && k == 0u))) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    var centre = vec2<f32>(a.pos_x, a.pos_y);
    var r_world = 6.0;
    var col = vec3<f32>(0.55, 0.55, 0.55);
    var organ = NO_ORGAN;
    var tangent = vec2<f32>(1.0, 0.0);
    var phase = 0.0;
    if (!naked) {
        let base = slot * MAX_BODY_V;
        let aa = (bodies_view[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
        let lp = body_pos_view[base + k];
        let cr = cos(a.rot);
        let sr = sin(a.rot);
        centre += vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
        // Tangente da cadeia (vizinhos), rodada para o mundo.
        let ka = select(k - 1u, k, k == 0u);
        let kb = select(k + 1u, k, k + 1u >= a.body_len);
        let tl = body_pos_view[base + kb] - body_pos_view[base + ka];
        let tn = select(vec2<f32>(1.0, 0.0), tl / length(tl), length(tl) > 1e-5);
        tangent = vec2<f32>(cr * tn.x - sr * tn.y, sr * tn.x + cr * tn.y);
        // Espessura (v3): 4·√(volume/130), mais folga para os discos se tocarem.
        r_world = 4.0 * sqrt(AA_VOLUME[aa] / 130.0) + 2.0;
        col = class_color(aa);
        let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
        if (oc != 0u) {
            organ = (oc & 0xFu) - 1u;
            if (organ == ORGAN_CLOCK) {
                let p = (oc >> 4u) & 0xFu;
                let period = CLOCK_PERIOD_BASE * f32(1u << (p >> 1u));
                phase = 6.2831853 * f32(a.age) / period;
            }
        }
        let s = signals_view[base + k];
        switch view.signal_view {
            case 1u: { col = signed_color(s.x, vec3<f32>(1.0, 0.45, 0.1), vec3<f32>(0.1, 0.6, 1.0)); }
            case 2u: { col = signed_color(s.y, vec3<f32>(0.3, 1.0, 0.3), vec3<f32>(0.95, 0.3, 0.9)); }
            case 3u: { col = vec3<f32>(0.5 + 0.5 * tanh(s.x), 0.5 + 0.5 * tanh(s.y), 0.35); }
            default: {}
        }
    }
    let ext = organ_extent(organ);
    // Nunca menos de 1,5 píxeis, para se ver com o zoom afastado.
    let r = max(r_world * ext, 1.5 / view.zoom);
    let w = centre + c * r;
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.local = c;
    o.organ = organ;
    o.tangent = tangent;
    o.core_phase = vec2<f32>(1.0 / ext, phase);
    // Pouca energia = mais escuro (só na vista química).
    let dim = mix(0.35, 1.0, clamp(a.energy / max(f32(a.body_len), 1.0), 0.0, 1.0));
    o.color = select(col, col * dim, view.signal_view == 0u);
    return o;
}

// Distância de p ao segmento a–b.
fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * h);
}

// Antena do centro até `tip` (com um botão na ponta): 1 dentro, 0 fora.
fn antenna(p: vec2<f32>, tip: vec2<f32>, core: f32) -> f32 {
    let stalk = seg_dist(p, vec2<f32>(0.0), tip) < 0.05;
    let knob = length(p - tip) < 0.11;
    return select(0.0, 1.0, (stalk && length(p) > core * 0.8) || knob);
}

@fragment
fn fs_agent(in: AgentVsOut) -> @location(0) vec4<f32> {
    let p = in.local;
    let d = length(p);
    let core = in.core_phase.x;
    let t = in.tangent;
    let nrm = vec2<f32>(-t.y, t.x);
    // Coordenadas ao longo (u) e através (v) da cadeia.
    let u = dot(p, t);
    let v = dot(p, nrm);
    let white = vec3<f32>(1.0);
    let rim_mix = smoothstep(core * 0.6, core * 0.8, d);

    switch in.organ {
        case NO_ORGAN: {
            if (d > 1.0) { discard; }
            let rim = smoothstep(0.75, 0.98, d);
            return vec4<f32>(mix(in.color, vec3<f32>(0.0), rim * 0.7), 1.0);
        }
        case ORGAN_MOUTH: {
            if (d > core) { discard; }
            // Abertura em cunha, virada para o lado esquerdo da cadeia.
            if (v > 0.15 * core && abs(u) < v * 0.9) { return vec4<f32>(0.05, 0.02, 0.02, 1.0); }
            return vec4<f32>(mix(in.color, white, rim_mix), 1.0);
        }
        case ORGAN_MUSCLE: {
            let e = (u / (core * 1.5)) * (u / (core * 1.5)) + (v / (core * 0.85)) * (v / (core * 0.85));
            if (e > 1.0) { discard; }
            let stripe = step(0.0, sin(u / core * 12.0));
            let red = mix(in.color, vec3<f32>(0.85, 0.25, 0.25), 0.55);
            return vec4<f32>(mix(red, red * 0.6, stripe) * mix(1.0, 0.7, smoothstep(0.7, 1.0, e)), 1.0);
        }
        case ORGAN_FOOD_SENSOR, ORGAN_LIGHT_SENSOR, ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: {
            let is_light = in.organ == ORGAN_LIGHT_SENSOR || in.organ == ORGAN_LIGHT_SENSOR_DIR;
            let ant_col = select(vec3<f32>(0.45, 1.0, 0.45), vec3<f32>(1.0, 0.95, 0.4), is_light);
            if (d < core) { return vec4<f32>(mix(in.color, ant_col, rim_mix), 1.0); }
            var hit = 0.0;
            if (in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR) {
                // Duas antenas, uma de cada lado, ligeiramente inclinadas para a frente.
                hit = max(antenna(p, nrm * 0.85 + t * 0.3, core), antenna(p, -nrm * 0.85 + t * 0.3, core));
            } else {
                // Coroa de 6 antenas curtas (amostra à volta toda).
                for (var i = 0u; i < 6u; i++) {
                    let ang = f32(i) * 1.0471976;
                    let dir = t * cos(ang) + nrm * sin(ang);
                    hit = max(hit, antenna(p, dir * 0.82, core));
                }
            }
            if (hit < 0.5) { discard; }
            return vec4<f32>(ant_col, 1.0);
        }
        case ORGAN_ENERGY_SENSOR: {
            if (d > core) { discard; }
            let ring = abs(d - core * 0.45) < core * 0.12;
            return vec4<f32>(select(mix(in.color, white, rim_mix), vec3<f32>(1.0, 0.85, 0.2), ring), 1.0);
        }
        case ORGAN_CLOCK: {
            if (d > core) { discard; }
            let hand_dir = t * cos(in.core_phase.y) + nrm * sin(in.core_phase.y);
            if (seg_dist(p, vec2<f32>(0.0), hand_dir * core * 0.8) < core * 0.12) {
                return vec4<f32>(0.05, 0.05, 0.1, 1.0);
            }
            return vec4<f32>(mix(vec3<f32>(0.85, 0.9, 1.0), white, rim_mix), 1.0);
        }
        case ORGAN_RELAY: {
            if (abs(u) + abs(v) > core * 1.25) { discard; }
            let edge = smoothstep(core * 0.85, core * 1.1, abs(u) + abs(v));
            return vec4<f32>(mix(in.color, vec3<f32>(0.6, 0.9, 1.0), edge), 1.0);
        }
        default: {
            // Armazenamento: disco com anéis concêntricos.
            if (d > core) { discard; }
            let rings = step(0.5, fract(d / core * 3.0));
            return vec4<f32>(mix(in.color, vec3<f32>(0.15, 0.1, 0.0), rings * 0.7), 1.0);
        }
    }
}
