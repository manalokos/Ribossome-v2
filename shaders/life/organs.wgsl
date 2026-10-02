// ÓRGÃOS E SINAIS INTERNOS (ver src/life/organs.rs).
//
// organ byte por resíduo: 0 = nenhum; senão (tipo + 1) | (parâmetro << 4).
// Sinais: dois canais (α, β) por resíduo, CONDUZIDOS entre vizinhos, um
// salto por passo: cada resíduo recebe o sinal dos vizinhos N e C pesado
// pela condutividade do seu aminoácido (AA_CONDUCTANCE, v3), com perda
// SIGNAL_DECAY, limitado a ±SIGNAL_MAX. Os sensores e o relógio emitem; o
// relé converte; todas as juntas dobram conforme α e β. O atraso de
// condução desfasa juntas distantes: uma onda, que é o que permite nadar
// (uma dobra sozinha é recíproca e não desloca nada).

const SIGNAL_DECAY: f32 = 0.95;
const SIGNAL_MAX: f32 = 4.0;
// Resposta das juntas aos sinais (v3, jan. 2026): desvio = SIGNAL_GAIN ×
// (α·sens_α + β·sens_β), saturado suavemente no máximo do aminoácido
// (AA_MAX_BEND, Ramachandran): máx·tanh(x/máx).
const SIGNAL_GAIN: f32 = 4.0;
// Energia gasta por passo por radiano de desvio mantido (todas as juntas).
const BEND_COST: f32 = 0.0005;
// Raio de amostragem dos sensores de comida e luz (unidades do mundo).
const SENSOR_RADIUS: f32 = 90.0;
// Ganho dos sensores de variação (diferença por passo).
const SENSOR_CHANGE_GAIN: f32 = 20.0;
// Capacidade de energia acrescentada pelo armazenamento, por (parâmetro + 1).
const STORAGE_CAPACITY: f32 = 4.0;

fn organ_get(slot: u32, k: u32) -> u32 {
    return (organs[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
}

// Ganho da intensidade do órgão: 2^((índice − 32)/8).
fn organ_gain(o: u32) -> f32 {
    return exp2((f32(o >> 8u) - 32.0) / 8.0);
}

// Tipo do órgão (0..7) ou 0xFF se não houver.
fn organ_type(o: u32) -> u32 {
    return select(0xFFu, (o & 0xFu) - 1u, o != 0u);
}

fn organ_param(o: u32) -> u32 {
    return (o >> 4u) & 0xFu;
}

// Capacidade extra de energia e custo de manutenção dos órgãos do agente.
fn organ_capacity(slot: u32, n: u32) -> f32 {
    var c = 0.0;
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_STORAGE) { c += STORAGE_CAPACITY * f32(organ_param(o) + 1u); }
    }
    return c;
}

fn organ_upkeep(slot: u32, n: u32) -> f32 {
    var u = 0.0;
    for (var k = 0u; k < n; k++) {
        let t = organ_type(organ_get(slot, k));
        if (t != 0xFFu) { u += ORGAN_UPKEEP[t]; }
    }
    return u;
}

// Multiplicador da catálise de um resíduo: SÓ a boca come (pedido do
// Filipe: a energia dos monómeros ativados entra só por bocas); a força da
// boca é a propensão catalítica do seu aminoácido × (2 + parâmetro).
fn organ_catalysis_mult(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    return select(0.0, 2.0 + f32(organ_param(o)), organ_type(o) == ORGAN_MOUTH);
}

// Desvio da junta k pelos sinais (rad). TODAS as juntas respondem, cada
// aminoácido com a sua sensibilidade a α e a β; o órgão "músculo" amplifica
// a resposta local ×(2 + parâmetro/2).
fn signal_deflection(slot: u32, k: u32) -> f32 {
    let aa = body_get(slot, k);
    let s = signals[slot * MAX_BODY + k];
    let o = organ_get(slot, k);
    let amp = select(1.0, (2.0 + 0.5 * f32(organ_param(o))) * organ_gain(o), organ_type(o) == ORGAN_MUSCLE);
    let lim = AA_MAX_BEND[aa];
    return lim * tanh(SIGNAL_GAIN * amp * (s.x * AA_ALPHA_SENS[aa] + s.y * AA_BETA_SENS[aa]) / lim);
}

// Amostra as células num disco de raio SENSOR_RADIUS à volta de `pos`:
// `what` 0 = comida (ativados, 4 canais), 1 = luz UV. Total: média do disco.
// Direcional: média do lado esquerdo da cadeia (+perp) − média do direito.
fn sense_disc(pos: vec2<f32>, perp: vec2<f32>, what: u32, directional: bool) -> f32 {
    let w = f32(WORLD_UNITS_PER_CELL);
    let r = i32(ceil(SENSOR_RADIUS / w));
    let c0 = vec2<i32>(floor(pos / w));
    var sum_l = 0.0;
    var sum_r = 0.0;
    var n_l = 0.0;
    var n_r = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let c = c0 + vec2<i32>(dx, dy);
            if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(i32(GRID_SIZE)))) { continue; }
            let d = (vec2<f32>(c) + 0.5) * w - pos;
            if (length(d) > SENSOR_RADIUS) { continue; }
            let idx = u32(c.y) * GRID_SIZE + u32(c.x);
            var v = 0.0;
            if (what == 0u) {
                var cnt = 0u;
                for (var ch = 0u; ch < 4u; ch++) { cnt += chem_act_count(idx, ch); }
                v = f32(cnt) / 12.0;
            } else {
                v = uv_light_at_cell(u32(c.x), u32(c.y)) * 4.0;
            }
            let side = dot(d, perp);
            if (!directional || side > 0.25 * w) {
                sum_l += v;
                n_l += 1.0;
            } else if (side < -0.25 * w) {
                sum_r += v;
                n_r += 1.0;
            }
        }
    }
    if (!directional) { return sum_l / max(n_l, 1.0); }
    return sum_l / max(n_l, 1.0) - sum_r / max(n_r, 1.0);
}

// Normal (lado esquerdo) da cadeia no resíduo k, no MUNDO.
fn chain_normal(slot: u32, a: Agent, k: u32) -> vec2<f32> {
    let base = slot * MAX_BODY;
    let ka = select(k - 1u, k, k == 0u);
    let kb = select(k + 1u, k, k + 1u >= a.body_len);
    let t = body_pos[base + kb] - body_pos[base + ka];
    let l = length(t);
    let tw = rotate(select(vec2<f32>(1.0, 0.0), t / l, l > 1e-5), a.rot);
    return vec2<f32>(-tw.y, tw.x);
}

// Um passo dos sinais: emissões dos sensores/relógio/relé e condução ao longo
// da cadeia. Devolve a energia gasta pelas juntas a manter os desvios.
fn signals_step(slot: u32, a: Agent, cap: f32) -> f32 {
    let n = a.body_len;
    if (n == 0u) { return 0.0; }
    let base = slot * MAX_BODY;
    // Emissões (a partir dos sinais do passo anterior, para o relé).
    var emit: array<vec2<f32>, 64>;
    var bend = 0.0;
    for (var k = 0u; k < n; k++) {
        bend += abs(signal_deflection(slot, k));
        emit[k] = vec2<f32>(0.0);
        let o = organ_get(slot, k);
        let t = organ_type(o);
        if (t == 0xFFu) { continue; }
        let p = organ_param(o);
        let s = signals[base + k];
        // SENSORES (comida, luz, energia): bit 0 = canal (α/β), bit 1 =
        // sinal (+/−), bit 2 = nível ou VARIAÇÃO desde o passo anterior
        // (memória por resíduo; é assim que as bactérias fazem quimiotaxia).
        var sensed = 0.0;
        var is_sensor = true;
        switch t {
            case ORGAN_FOOD_SENSOR, ORGAN_LIGHT_SENSOR, ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: {
                let what = select(0u, 1u, t == ORGAN_LIGHT_SENSOR || t == ORGAN_LIGHT_SENSOR_DIR);
                let dir = t == ORGAN_FOOD_SENSOR_DIR || t == ORGAN_LIGHT_SENSOR_DIR;
                sensed = sense_disc(residue_world(slot, a, k), chain_normal(slot, a, k), what, dir);
            }
            case ORGAN_ENERGY_SENSOR: {
                sensed = clamp(a.energy / max(cap, 1e-3), 0.0, 1.0) * 2.0;
            }
            default: { is_sensor = false; }
        }
        if (is_sensor) {
            let mi = base + k;
            var v = sensed;
            if ((p & 4u) != 0u) {
                // Variação, amplificada (as mudanças por passo são pequenas).
                v = (sensed - sensor_mem[mi]) * SENSOR_CHANGE_GAIN;
            }
            sensor_mem[mi] = sensed;
            v = select(v, -v, (p & 2u) != 0u) * organ_gain(o);
            if ((p & 1u) == 0u) { emit[k].x = v; } else { emit[k].y = v; }
            continue;
        }
        switch t {
            case ORGAN_CLOCK: {
                // bit 0 = canal; bits 1–2 = período (20, 40, 80 ou 160 passos).
                let period = CLOCK_PERIOD_BASE * f32(1u << (p >> 1u));
                let v = sin(6.2831853 * f32(a.age) / period) * organ_gain(o);
                if ((p & 1u) == 0u) { emit[k].x = v; } else { emit[k].y = v; }
            }
            case ORGAN_RELAY: {
                // bits 0–1: 0 α->β, 1 β->α, 2 inverte α, 3 inverte β; bit 2 = ganho ×2.
                let g = select(1.0, 2.0, (p & 4u) != 0u) * organ_gain(o);
                switch (p & 3u) {
                    case 0u: { emit[k].y = g * s.x; }
                    case 1u: { emit[k].x = g * s.y; }
                    case 2u: { emit[k].x = -g * 2.0 * s.x; }
                    default: { emit[k].y = -g * 2.0 * s.y; }
                }
            }
            default: {}
        }
    }
    // Condução com um passo de atraso: o resíduo k recebe o que os vizinhos
    // k−1 (lado N) e k+1 (lado C) tinham no passo anterior, pesado pela
    // condutividade do seu aminoácido, perdendo SIGNAL_DECAY, mais a emissão.
    var prev = vec2<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let here = signals[base + k];
        let next = select(vec2<f32>(0.0), signals[base + k + 1u], k + 1u < n);
        let c = AA_CONDUCTANCE[body_get(slot, k)];
        let incoming = vec2<f32>(c.x * prev.x + c.y * next.x, c.z * prev.y + c.w * next.y);
        let s = SIGNAL_DECAY * incoming + emit[k];
        signals[base + k] = clamp(s, vec2<f32>(-SIGNAL_MAX), vec2<f32>(SIGNAL_MAX));
        prev = here;
    }
    return bend * BEND_COST;
}
