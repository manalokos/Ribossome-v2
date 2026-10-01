// ÓRGÃOS E SINAIS INTERNOS (ver src/life/organs.rs).
//
// organ byte por resíduo: 0 = nenhum; senão (tipo + 1) | (parâmetro << 4).
// Sinais: dois canais (α, β) por resíduo, CONDUZIDOS ao longo da cadeia no
// sentido N->C (a polaridade da cadeia), um resíduo por passo, com perda
// SIGNAL_DECAY por salto, limitados a ±SIGNAL_MAX. Os sensores e o relógio
// emitem; o relé converte; o músculo dobra a sua junta conforme α. O atraso
// de condução desfasa músculos distantes: uma onda, que é o que permite
// nadar (um músculo sozinho é recíproco e não desloca nada).

const SIGNAL_DECAY: f32 = 0.95;
const SIGNAL_MAX: f32 = 4.0;
// Resposta das juntas aos sinais (v3, jan. 2026): desvio = SIGNAL_GAIN ×
// (α·sens_α + β·sens_β), limitado a ±MAX_SIGNAL_ANGLE.
const SIGNAL_GAIN: f32 = 4.0;
const MAX_SIGNAL_ANGLE: f32 = 2.4;
// Energia gasta por passo por radiano de desvio mantido (todas as juntas).
const BEND_COST: f32 = 0.0005;
// Período do relógio: CLOCK_PERIOD_BASE × 2^(bits 1–2 do parâmetro) passos.
const CLOCK_PERIOD_BASE: f32 = 20.0;
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
    var up = ORGAN_UPKEEP;
    var u = 0.0;
    for (var k = 0u; k < n; k++) {
        let t = organ_type(organ_get(slot, k));
        if (t != 0xFFu) { u += up[t]; }
    }
    return u;
}

// Multiplicador da catálise de um resíduo (a boca come mais).
fn organ_catalysis_mult(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    return select(1.0, 2.0 + f32(organ_param(o)), organ_type(o) == ORGAN_MOUTH);
}

// Desvio da junta k pelos sinais (rad). TODAS as juntas respondem, cada
// aminoácido com a sua sensibilidade a α e a β; o órgão "músculo" amplifica
// a resposta local ×(2 + parâmetro/2).
fn signal_deflection(slot: u32, k: u32) -> f32 {
    var sa = AA_ALPHA_SENS;
    var sb = AA_BETA_SENS;
    let aa = body_get(slot, k);
    let s = signals[slot * MAX_BODY + k];
    let o = organ_get(slot, k);
    let amp = select(1.0, (2.0 + 0.5 * f32(organ_param(o))) * organ_gain(o), organ_type(o) == ORGAN_MUSCLE);
    return clamp(SIGNAL_GAIN * amp * (s.x * sa[aa] + s.y * sb[aa]), -MAX_SIGNAL_ANGLE, MAX_SIGNAL_ANGLE);
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
            case ORGAN_FOOD_SENSOR: {
                let cell = world_to_cell(residue_world(slot, a, k));
                var c = 0u;
                for (var ch = 0u; ch < 4u; ch++) { c += chem_act_count(cell, ch); }
                sensed = f32(c) / 12.0;
            }
            case ORGAN_LIGHT_SENSOR: {
                let cell = world_to_cell(residue_world(slot, a, k));
                sensed = uv_light_at_cell(cell % GRID_SIZE, cell / GRID_SIZE) * 4.0;
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
    // Condução N->C com um passo de atraso: o resíduo k recebe o sinal que
    // o k−1 tinha no passo anterior (perdendo SIGNAL_DECAY) mais a sua emissão.
    var prev = vec2<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let here = signals[base + k];
        let s = SIGNAL_DECAY * prev + emit[k];
        signals[base + k] = clamp(s, vec2<f32>(-SIGNAL_MAX), vec2<f32>(SIGNAL_MAX));
        prev = here;
    }
    return bend * BEND_COST;
}
