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
// Deflexão máxima de um músculo (rad) para o parâmetro máximo.
const MUSCLE_MAX: f32 = 0.6;
// Energia gasta por passo por unidade de atividade muscular |tanh(α)|.
const MUSCLE_COST: f32 = 0.002;
// Período do relógio: CLOCK_PERIOD_BASE × (parâmetro + 1) passos.
const CLOCK_PERIOD_BASE: f32 = 20.0;
// Capacidade de energia acrescentada pelo armazenamento, por (parâmetro + 1).
const STORAGE_CAPACITY: f32 = 4.0;

fn organ_get(slot: u32, k: u32) -> u32 {
    return (organs[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
}

// Tipo do órgão (0..7) ou 0xFF se não houver.
fn organ_type(o: u32) -> u32 {
    return select(0xFFu, (o & 0xFu) - 1u, o != 0u);
}

fn organ_param(o: u32) -> u32 {
    return o >> 4u;
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

// Deflexão de um músculo (rad) para o sinal α atual; 0 se não for músculo.
// Parâmetro 0–3: dobra para um lado; 4–7: para o outro; força (p % 4 + 1)/4.
fn muscle_deflection(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    if (organ_type(o) != ORGAN_MUSCLE) { return 0.0; }
    let p = organ_param(o);
    let dir = select(1.0, -1.0, p >= 4u);
    return dir * MUSCLE_MAX * f32(p % 4u + 1u) / 4.0 * tanh(signals[slot * MAX_BODY + k].x);
}

// Um passo dos sinais: emissões dos sensores/relógio/relé e difusão ao longo
// da cadeia. Devolve a energia gasta pelos músculos neste passo.
fn signals_step(slot: u32, a: Agent, cap: f32) -> f32 {
    let n = a.body_len;
    if (n == 0u) { return 0.0; }
    let base = slot * MAX_BODY;
    // Emissões (a partir dos sinais do passo anterior, para o relé).
    var emit: array<vec2<f32>, 64>;
    var muscle_activity = 0.0;
    for (var k = 0u; k < n; k++) {
        emit[k] = vec2<f32>(0.0);
        let o = organ_get(slot, k);
        let t = organ_type(o);
        if (t == 0xFFu) { continue; }
        let p = organ_param(o);
        let s = signals[base + k];
        switch t {
            case ORGAN_FOOD_SENSOR: {
                // p 0–3: só o nucleótido p (ativados); 4–7: todos os ativados.
                let cell = world_to_cell(residue_world(slot, a, k));
                var c = 0u;
                if (p < 4u) {
                    c = chem_act_count(cell, p);
                } else {
                    for (var ch = 0u; ch < 4u; ch++) { c += chem_act_count(cell, ch); }
                }
                emit[k].x = f32(c) / 12.0;
            }
            case ORGAN_LIGHT_SENSOR: {
                let cell = world_to_cell(residue_world(slot, a, k));
                emit[k].x = uv_light_at_cell(cell % GRID_SIZE, cell / GRID_SIZE) * f32(p + 1u);
            }
            case ORGAN_ENERGY_SENSOR: {
                emit[k].y = clamp(a.energy / max(cap, 1e-3), 0.0, 1.0) * f32(p + 1u) / 4.0;
            }
            case ORGAN_CLOCK: {
                let period = CLOCK_PERIOD_BASE * f32(p + 1u);
                emit[k].x = sin(6.2831853 * f32(a.age) / period);
            }
            case ORGAN_RELAY: {
                // p % 3: 0 α->β, 1 β->α, 2 inverte α.
                let m = p % 3u;
                if (m == 0u) { emit[k].y = s.x; }
                else if (m == 1u) { emit[k].x = s.y; }
                else { emit[k].x = -2.0 * s.x; }
            }
            case ORGAN_MUSCLE: {
                muscle_activity += abs(tanh(s.x));
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
    return muscle_activity * MUSCLE_COST;
}
