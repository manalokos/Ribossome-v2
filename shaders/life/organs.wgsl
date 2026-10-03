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

// Propriedades da variante do órgão (assets/orgaos.json; ordem em
// organs::ORGAN_PROPS).
fn organ_var(o: u32) -> OrganVariant {
    return organ_variants[organ_type(o) * ORGAN_VARIANTS + min(organ_param(o), ORGAN_VARIANTS - 1u)];
}

// Propriedades físicas (custos) da variante do órgão `o` (o != 0).
fn organ_cost(o: u32) -> OrganProps {
    return organ_props[organ_type(o) * ORGAN_VARIANTS + min(organ_param(o), ORGAN_VARIANTS - 1u)];
}

// Capacidade extra de energia e custo de manutenção dos órgãos do agente.
fn organ_capacity(slot: u32, n: u32) -> f32 {
    var c = 0.0;
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_STORAGE) { c += max(organ_var(o).p0, 0.0) * organ_gain(o); }
    }
    return c;
}

fn organ_upkeep(slot: u32, n: u32) -> f32 {
    var u = 0.0;
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (o != 0u) {
            let c = organ_cost(o);
            u += c.upkeep * select(1.0, organ_gain(o), c.gain_pays > 0.5);
        }
    }
    return u;
}

// Multiplicador da catálise de um resíduo: SÓ a boca come (pedido do
// Filipe: a energia dos monómeros ativados entra só por bocas); a força da
// boca é a da variante × ganho (× a propensão catalítica do aminoácido).
fn organ_catalysis_mult(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    return select(0.0, max(organ_var(o).p0, 0.0) * organ_gain(o), organ_type(o) == ORGAN_MOUTH);
}

// Desvio da junta k pelos sinais (rad). TODAS as juntas respondem, cada
// aminoácido com a sua sensibilidade a α e a β; o órgão "músculo" amplifica
// a resposta local ×(2 + parâmetro/2).
fn signal_deflection(slot: u32, k: u32) -> f32 {
    let aa = body_get(slot, k);
    let s = signals[slot * MAX_BODY + k];
    let o = organ_get(slot, k);
    // Músculo: amplifica o canal escolhido pela variante (0 α, 1 β, 2 ambos).
    var amp_a = 1.0;
    var amp_b = 1.0;
    if (organ_type(o) == ORGAN_MUSCLE) {
        let mv = organ_var(o);
        let amp = mv.p0 * organ_gain(o);
        if (mv.p1 < 0.5 || mv.p1 > 1.5) { amp_a = amp; }
        if (mv.p1 > 0.5) { amp_b = amp; }
    }
    let pr = aa_props[aa];
    let lim = max(pr.max_bend, 1e-3);
    return lim * tanh(SIGNAL_GAIN * (s.x * pr.sens_alpha * amp_a + s.y * pr.sens_beta * amp_b) / lim);
}

// GRELHA DOS CORPOS: resíduos de agentes por célula (BODY_DIV × BODY_DIV
// células do ambiente), para os sensores de corpos.
const BODY_DIV: u32 = 2u;
const BODY_SIZE: u32 = GRID_SIZE / BODY_DIV;

@compute @workgroup_size(256)
fn body_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i < BODY_SIZE * BODY_SIZE) { atomicStore(&body_grid[i], 0u); }
}

@compute @workgroup_size(64)
fn body_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    for (var k = 0u; k < max(a.body_len, 1u); k++) {
        var pk = vec2<f32>(a.pos_x, a.pos_y);
        if (a.body_len > 0u) { pk = residue_world(slot, a, k); }
        let c = world_to_cell(pk);
        let bx = (c % GRID_SIZE) / BODY_DIV;
        let by = (c / GRID_SIZE) / BODY_DIV;
        atomicAdd(&body_grid[by * BODY_SIZE + bx], 1u);
    }
}

// Amostra as células num disco de raio SENSOR_RADIUS à volta de `pos`:
// `what` 0 = comida (ativados, 4 canais), 1 = luz UV, 2 = gastos (4
// canais), 3 = corpos de agentes (a grelha dos corpos), 4 = temperatura,
// 5 = redutor das fumarolas. Total: média do disco. Direcional: média do
// lado esquerdo da cadeia (+perp) − a do direito.
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
            } else if (what == 2u) {
                var cnt = 0u;
                for (var ch = 0u; ch < 4u; ch++) { cnt += atomicLoad(&chem_grid[idx * 4u + ch]) >> 16u; }
                v = f32(cnt) / 12.0;
            } else if (what == 4u || what == 5u) {
                // Temperatura (÷ 4) ou redutor (÷ 5) no fluido sob a célula.
                let fi = fluid_index_at_world((vec2<f32>(c) + 0.5) * w);
                v = select(temp_in[fi] / 4.0, redox_in[fi] / 5.0, what == 5u);
            } else if (what == 3u) {
                // Residuos por célula do ambiente (a grelha dos corpos é BODY_DIV² maior).
                let b = atomicLoad(&body_grid[(u32(c.y) / BODY_DIV) * BODY_SIZE + u32(c.x) / BODY_DIV]);
                v = f32(b) / f32(BODY_DIV * BODY_DIV);
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

// O que o próprio corpo conta no disco do sensor de corpos (para o
// descontar: o agente não se sente a si mesmo), na mesma escala de sense_disc.
fn own_body_in_disc(slot: u32, a: Agent, pos: vec2<f32>, perp: vec2<f32>, directional: bool) -> f32 {
    let w = f32(WORLD_UNITS_PER_CELL);
    let r = ceil(SENSOR_RADIUS / w);
    // Células no disco (como em sense_disc, aproximado pela área).
    let cells = max(3.14159265 * r * r, 1.0);
    var l = 0.0;
    var rt = 0.0;
    for (var k = 0u; k < a.body_len; k++) {
        let d = residue_world(slot, a, k) - pos;
        if (length(d) > SENSOR_RADIUS) { continue; }
        let side = dot(d, perp);
        if (!directional || side > 0.25 * w) { l += 1.0; } else if (side < -0.25 * w) { rt += 1.0; }
    }
    if (!directional) { return l / cells; }
    return (l - rt) / (0.5 * cells);
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
                var what = select(0u, 1u, t == ORGAN_LIGHT_SENSOR || t == ORGAN_LIGHT_SENSOR_DIR);
                // O alvo vem da variante (p4): comida 0 ativados, 1 gastos,
                // 2 corpos; físicos 0 luz, 1 temperatura, 2 redutor.
                let alvo = organ_var(o).p4;
                if (what == 0u) {
                    if (alvo > 1.5) { what = 3u; } else if (alvo > 0.5) { what = 2u; }
                } else {
                    if (alvo > 1.5) { what = 5u; } else if (alvo > 0.5) { what = 4u; }
                }
                let dir = t == ORGAN_FOOD_SENSOR_DIR || t == ORGAN_LIGHT_SENSOR_DIR;
                let here = residue_world(slot, a, k);
                let perp = chain_normal(slot, a, k);
                sensed = sense_disc(here, perp, what, dir);
                if (what == 3u) { sensed -= own_body_in_disc(slot, a, here, perp, dir); }
            }
            case ORGAN_ENERGY_SENSOR: {
                sensed = clamp(a.energy / max(cap, 1e-3), 0.0, 1.0) * 2.0;
            }
            default: { is_sensor = false; }
        }
        let ov = organ_var(o);
        if (is_sensor) {
            // Variante: p0 canal, p1 ganho (com sinal), p2 modo (0 nível, 1
            // variação), p3 memória da referência na variação.
            let mi = base + k;
            var v = sensed;
            if (ov.p2 >= 0.5) {
                // Variação, amplificada (as mudanças por passo são pequenas);
                // a referência segue o sentido com a memória da variante.
                v = (sensed - sensor_mem[mi]) * SENSOR_CHANGE_GAIN;
                sensor_mem[mi] = mix(sensed, sensor_mem[mi], clamp(ov.p3, 0.0, 0.999));
            } else {
                sensor_mem[mi] = sensed;
            }
            v *= ov.p1 * organ_gain(o);
            if (ov.p0 < 0.5) { emit[k].x = v; } else { emit[k].y = v; }
            continue;
        }
        switch t {
            case ORGAN_BIAS: {
                // p0 canal, p1 valor: emite sempre o mesmo (um "bias").
                let v = ov.p1 * organ_gain(o);
                if (ov.p0 < 0.5) { emit[k].x = v; } else { emit[k].y = v; }
            }
            case ORGAN_CLOCK: {
                // p0 canal, p1 período, p2/p3: o relógio acelera com α/β (um
                // oscilador controlado). A fase vive em sensor_mem.
                let mi = base + k;
                let rate = max(0.05, 1.0 + ov.p2 * s.x + ov.p3 * s.y);
                var phase = sensor_mem[mi] + 6.2831853 * rate / max(ov.p1, 2.0);
                phase = phase - 6.2831853 * floor(phase / 6.2831853);
                sensor_mem[mi] = phase;
                let v = sin(phase) * organ_gain(o);
                if (ov.p0 < 0.5) { emit[k].x = v; } else { emit[k].y = v; }
            }
            case ORGAN_RELAY: {
                // p0 entrada, p1 saída, p2 ganho, p3 limiar (porta).
                let x = select(s.x, s.y, ov.p0 >= 0.5);
                let y = sign(x) * max(abs(x) - max(ov.p3, 0.0), 0.0) * ov.p2 * organ_gain(o);
                if (ov.p1 < 0.5) { emit[k].x = y; } else { emit[k].y = y; }
            }
            default: {}
        }
    }
    // Ligações a outros agentes: o resíduo ligado conta como mais um vizinho.
    if (params.bond_signal > 0.0) {
        for (var i = 0u; i < MAX_BONDS; i++) {
            let b = bond_at(slot, i);
            if (b.x == BOND_NONE || !bond_partner_ok(b)) { continue; }
            let k = bond_mine(b);
            if (k < n) { emit[k] += params.bond_signal * SIGNAL_DECAY * signals[b.x * MAX_BODY + bond_theirs(b)]; }
        }
    }
    // Condução com um passo de atraso: o resíduo k recebe o que os vizinhos
    // k−1 (lado N) e k+1 (lado C) tinham no passo anterior, pesado pela
    // condutividade do seu aminoácido, perdendo SIGNAL_DECAY, mais a emissão.
    var prev = vec2<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let here = signals[base + k];
        let next = select(vec2<f32>(0.0), signals[base + k + 1u], k + 1u < n);
        let pr = aa_props[body_get(slot, k)];
        let c = vec4<f32>(pr.cond_alpha_n, pr.cond_alpha_c, pr.cond_beta_n, pr.cond_beta_c);
        let incoming = vec2<f32>(c.x * prev.x + c.y * next.x, c.z * prev.y + c.w * next.y);
        let s = SIGNAL_DECAY * incoming + emit[k];
        signals[base + k] = clamp(s, vec2<f32>(-SIGNAL_MAX), vec2<f32>(SIGNAL_MAX));
        prev = here;
    }
    return bend * BEND_COST;
}
