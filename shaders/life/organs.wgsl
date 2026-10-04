// ÓRGÃOS E SINAIS INTERNOS (ver src/life/organs.rs).
//
// código por resíduo: 0 = nenhum; senão (tipo + 1) | (parâmetro << 5) |
// (intensidade << 8).
// Sinais: dois canais (α, β) por resíduo, CONDUZIDOS entre vizinhos, um
// salto por passo: cada resíduo recebe o sinal dos vizinhos N e C pesado
// pela condutividade do seu aminoácido (AA_CONDUCTANCE, v3), com perda
// SIGNAL_DECAY, limitado a ±SIGNAL_MAX. Os sensores e o relógio emitem; o
// relé converte; todas as juntas dobram conforme α e β. O atraso de
// condução desfasa juntas distantes: uma onda, que é o que permite nadar
// (uma dobra sozinha é recíproca e não desloca nada).

const SIGNAL_DECAY: f32 = 0.95;
// EMISSÃO POR CONTACTO (osciladores sem relógio): um resíduo emissor que
// toca (a menos de CONTACT_EMIT_RADIUS) num resíduo NÃO vizinho na cadeia
// (|i − k| >= 3) da classe que procura emite CONTACT_EMIT no seu canal. Um
// corpo enrolado pode fazer o ciclo toca -> emite -> o sinal abre a dobra ->
// deixa de tocar -> relaxa -> toca outra vez: o ritmo vem da forma. Grátis.
const CONTACT_EMIT_RADIUS: f32 = 10.0;
const CONTACT_EMIT: f32 = 1.0;
// Dinâmica do v3 (modo 0 dos sinais).
const V3_SIGNAL_DECAY: f32 = 0.997;
const V3_SIGNAL_UPDATE: f32 = 0.75;
const SIGNAL_MAX: f32 = 4.0;
// Resposta das juntas aos sinais (v3, jan. 2026): desvio = SIGNAL_GAIN ×
// (α·sens_α + β·sens_β), saturado suavemente no máximo do aminoácido
// (AA_MAX_BEND, Ramachandran): máx·tanh(x/máx).
const SIGNAL_GAIN: f32 = 4.0;
// Resposta uniforme das juntas (params.signal_mode >= 1): α dobra para a
// esquerda da cadeia (+), β para a direita (−), igual em todos os resíduos
// (0,3 ≈ a média do módulo das sensibilidades por aminoácido).
const SIGNAL_UNIFORM_SENS: f32 = 0.3;
// Energia gasta por passo por radiano de desvio mantido (todas as juntas).
const BEND_COST: f32 = 0.0005;
// Raio de amostragem dos sensores de comida e luz (unidades do mundo).
// Raio do disco dos sensores (mundo): 6 células (era 3: com ~0,4 ativados por
// célula, cada lado via ~4 monómeros e o ruído de contagem afogava o sinal).
const SENSOR_RADIUS: f32 = 180.0;
// Carga do sensor: no modo NÍVEL a fração que fica em cada passo é a
// "memória" da variante (descarga = 1 − memória; tempo ~1/(1 − memória)
// passos). No modo VARIAÇÃO a carga rápida usa esta fração fixa e a
// referência lenta usa a memória da variante.
const SENSOR_FAST_KEEP: f32 = 0.7;
// Sensor acabado de nascer (ainda sem carga): ver bindings.wgsl.
const SENSOR_UNSET: f32 = -1e30;
// AMOSTRAGEM ESTOCÁSTICA: em cada passo o sensor lê este número de células
// ao acaso dentro do raio (como moléculas a chegar a um recetor), não o
// disco inteiro; a carga integra as chegadas no tempo. Um sensor exato a 6
// células de distância seria "ação à distância" (precisava de um campo).
const SENSOR_SAMPLES: u32 = 16u;
const S_SENSE: u32 = 12u << 16u;    // + resíduo·16 + amostra
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
    return select(0xFFu, (o & 0x1Fu) - 1u, o != 0u);
}

fn organ_param(o: u32) -> u32 {
    return (o >> 5u) & 0x7u;
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
            // gain_pays = expoente da intensidade na manutenção: 1 = linear
            // (órgãos que fazem trabalho), 0,5 = raiz (sinais), 0 = não conta.
            u += c.upkeep * pow(organ_gain(o), clamp(c.gain_pays, 0.0, 1.0));
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
    // 4 canais: α e β dobram (e os músculos amplificam-nos); γ e δ dobram só
    // pela sensibilidade do aminoácido (modos 0 e 3).
    var sens = vec4<f32>(pr.sens_alpha, pr.sens_beta, pr.sens_gamma, pr.sens_delta);
    // Modos 1 e 2: resposta igual em todas as juntas (α para um lado, β para
    // o outro; γ e δ são mensageiros internos, não dobram: só agem depois de
    // um relé os passar para α ou β). Modo 3: a de cada aminoácido.
    if (params.signal_mode >= 0.5 && params.signal_mode < 2.5) {
        sens = vec4<f32>(SIGNAL_UNIFORM_SENS, -SIGNAL_UNIFORM_SENS, 0.0, 0.0);
    }
    return lim * tanh(SIGNAL_GAIN * (s.x * sens.x * amp_a + s.y * sens.y * amp_b + s.z * sens.z + s.w * sens.w) / lim);
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
// 5 = redutor das fumarolas, 6 = terreno (grãos por célula, 0 água .. 1
// rocha). Total: média do disco. Direcional: média do lado esquerdo da
// cadeia (+perp) − a do direito.
// RESPOSTA DO RECETOR (adaptação ao fundo, como nos recetores reais):
// - nível: OCUPAÇÃO c / (c + K), que satura (0..1). Numa sopa pobre a
//   leitura já é uma fração apreciável, em vez de ~0,03;
// - direcional: CONTRASTE relativo (esq − dir) / (esq + dir + K), em −1..1
//   (lei de Weber: sente "mais 20%" tanto na sopa rica como na pobre).
// K = meia-saturação, nas unidades de sense_value de cada alvo.
fn sense_k(what: u32) -> f32 {
    switch what {
        case 0u, 2u: { return 1.0 / 12.0; }   // 1 monómero por célula
        case 3u: { return 0.2; }              // corpos: 0,2 resíduos por célula
        case 4u: { return 0.5; }              // temperatura 2
        case 5u: { return 1.0; }              // redutor 5
        case 6u: { return 0.3; }              // ~1 grão por célula
        default: { return 1.0; }              // luz 0,25
    }
}

// O que uma célula vale para um sensor (ver `what` em sense_disc). `aff` =
// peso de cada canal (A, U, G, C; 1 cada = sem preferência): os sensores de
// monómeros são específicos, como os recetores reais (ver sensor_affinity).
fn sense_value(what: u32, c: vec2<i32>, aff: vec4<f32>) -> f32 {
    let idx = u32(c.y) * GRID_SIZE + u32(c.x);
    if (what == 0u) {
        var cnt = 0.0;
        for (var ch = 0u; ch < 4u; ch++) { cnt += aff[ch] * f32(chem_act_count(idx, ch)); }
        return cnt / 12.0;
    } else if (what == 2u) {
        var cnt = 0.0;
        for (var ch = 0u; ch < 4u; ch++) { cnt += aff[ch] * f32(atomicLoad(&chem_grid[idx * 4u + ch]) >> 16u); }
        return cnt / 12.0;
    } else if (what == 4u || what == 5u) {
        let fi = fluid_index_at_world((vec2<f32>(c) + 0.5) * f32(WORLD_UNITS_PER_CELL));
        return select(temp_in[fi] / 4.0, redox_in[fi] / 5.0, what == 5u);
    } else if (what == 6u) {
        return f32(min(gamma_count(idx), GAMMA_SOLID_THRESHOLD)) / f32(GAMMA_SOLID_THRESHOLD);
    }
    return uv_light_at_cell(u32(c.x), u32(c.y)) * 4.0;
}

// ESPECIFICIDADE do sensor de monómeros: a "antena" é o resíduo SEGUINTE da
// cadeia (um bolso de ligação feito com o vizinho); os pesos por canal são as
// afinidades de substrato desse aminoácido (as que as bocas usam; somam 1,
// por isso × 4: sem preferência = 1 em cada canal, como antes). Sem vizinho
// (sensor na ponta C) não há preferência.
fn sensor_affinity(slot: u32, k: u32, n: u32) -> vec4<f32> {
    if (k + 1u >= n) { return vec4<f32>(1.0); }
    let pr = aa_props[body_get(slot, k + 1u)];
    return 4.0 * vec4<f32>(pr.sub_a, pr.sub_u, pr.sub_g, pr.sub_c);
}

// Leitura ESTOCÁSTICA: SENSOR_SAMPLES células ao acaso no disco (uniformes
// em área). Total: média das amostras. Direcional: metade das amostras de
// cada lado da cadeia (espelhadas para o lado certo), esquerda − direita.
// `key`/`salt` escolhem a sequência de sorteio (id do agente, resíduo).
fn sense_sample(pos: vec2<f32>, perp: vec2<f32>, what: u32, directional: bool, key: u32, salt: u32, aff: vec4<f32>) -> f32 {
    let w = f32(WORLD_UNITS_PER_CELL);
    var sum_l = 0.0;
    var sum_r = 0.0;
    var n_l = 0.0;
    var n_r = 0.0;
    for (var i = 0u; i < SENSOR_SAMPLES; i += 2u) {
        let q = rng_f4(key, params.epoch, S_SENSE + salt * 16u + i / 2u);
        for (var h = 0u; h < 2u; h++) {
            let u = select(q.xy, q.zw, h == 1u);
            let rad = SENSOR_RADIUS * sqrt(u.x);
            let ang = 6.2831853 * u.y;
            var d = rad * vec2<f32>(cos(ang), sin(ang));
            let left = !directional || h == 0u;
            if (directional) {
                // Espelha para o lado pedido (h = 0 esquerda, 1 direita).
                let side = dot(d, perp);
                if ((side < 0.0) == left) { d -= 2.0 * side * perp; }
            }
            let c = vec2<i32>(floor((pos + d) / w));
            if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(i32(GRID_SIZE)))) { continue; }
            let v = sense_value(what, c, aff);
            if (left) {
                sum_l += v;
                n_l += 1.0;
            } else {
                sum_r += v;
                n_r += 1.0;
            }
        }
    }
    let kk = sense_k(what);
    let l = sum_l / max(n_l, 1.0);
    if (!directional) { return l / (l + kk); }
    let r = sum_r / max(n_r, 1.0);
    return (l - r) / (l + r + kk);
}

// Leitura EXATA do disco inteiro (só para os corpos de agentes, onde é
// preciso descontar o próprio corpo célula a célula).
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
            } else if (what == 6u) {
                v = f32(min(gamma_count(idx), GAMMA_SOLID_THRESHOLD)) / f32(GAMMA_SOLID_THRESHOLD);
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
    var emit: array<vec4<f32>, 64>;
    // Portas: 1 = o canal passa neste resíduo, 0 = travado (relés).
    var gate: array<vec4<f32>, 64>;
    var bend = 0.0;
    for (var k = 0u; k < n; k++) {
        bend += abs(signal_deflection(slot, k));
        emit[k] = vec4<f32>(0.0);
        gate[k] = vec4<f32>(1.0);
        // Emissão por contacto (qualquer resíduo, com ou sem órgão).
        let cpr = aa_props[body_get(slot, k)];
        if (cpr.contact_channel >= 0.0) {
            let pk = body_pos[base + k];
            for (var j = 0u; j < n; j++) {
                if (j + 2u >= k && j <= k + 2u) { continue; }
                if (aa_props[body_get(slot, j)].contact_class != cpr.contact_want) { continue; }
                let dj = body_pos[base + j] - pk;
                if (dot(dj, dj) < CONTACT_EMIT_RADIUS * CONTACT_EMIT_RADIUS) {
                    emit[k][u32(clamp(cpr.contact_channel, 0.0, 3.0))] += CONTACT_EMIT;
                    break;
                }
            }
        }
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
                // 2 corpos; físicos 0 luz, 1 temperatura, 2 redutor, 3 terreno.
                let alvo = organ_var(o).p4;
                if (what == 0u) {
                    if (alvo > 1.5) { what = 3u; } else if (alvo > 0.5) { what = 2u; }
                } else {
                    if (alvo > 2.5) { what = 6u; } else if (alvo > 1.5) { what = 5u; } else if (alvo > 0.5) { what = 4u; }
                }
                let dir = t == ORGAN_FOOD_SENSOR_DIR || t == ORGAN_LIGHT_SENSOR_DIR;
                let here = residue_world(slot, a, k);
                let perp = chain_normal(slot, a, k);
                if (what == 3u) {
                    // Corpos: só se conhece a diferença (ou o total) já sem o
                    // próprio corpo; a mesma saturação, com sinal.
                    let raw = sense_disc(here, perp, what, dir) - own_body_in_disc(slot, a, here, perp, dir);
                    sensed = raw / (abs(raw) + sense_k(3u));
                } else {
                    sensed = sense_sample(here, perp, what, dir, a.id, k, sensor_affinity(slot, k, n));
                }
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
            let keep = clamp(ov.p3, 0.0, 0.999);
            // Nasce já carregado com o que há à volta (sem transitório).
            let unset = sensor_avg[mi] < -1e20;
            var v = 0.0;
            if (ov.p2 >= 0.5) {
                // VARIAÇÃO: carga rápida − referência lenta, amplificada (as
                // mudanças por passo são pequenas).
                let fast = select(mix(sensed, sensor_avg[mi], SENSOR_FAST_KEEP), sensed, unset);
                let slow = select(sensor_mem[mi], sensed, unset);
                sensor_avg[mi] = fast;
                v = (fast - slow) * SENSOR_CHANGE_GAIN;
                sensor_mem[mi] = mix(fast, slow, keep);
            } else {
                // NÍVEL: a carga (integrador com fuga pela variante).
                let q = select(mix(sensed, sensor_avg[mi], keep), sensed, unset);
                sensor_avg[mi] = q;
                sensor_mem[mi] = q;
                v = q;
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
                let v = sin(phase) * organ_gain(o) * (1.0 - clamp(params.clock_mute, 0.0, 1.0));
                if (ov.p0 < 0.5) { emit[k].x = v; } else { emit[k].y = v; }
            }
            case ORGAN_RELAY: {
                // RELÉ = lógica entre os 4 canais. O 3.º codão (o da
                // intensidade) escolhe os canais: entrada = bits 0–1, saída =
                // bits 2–3, força = bits 4–5 (×0,5, 1, 2, 4). A variante dá a
                // função (p0), o ganho (p1) e o limiar (p2): assim cada parte
                // do corpo pode dar a um canal o significado que quiser.
                let gi = o >> 8u;
                let cin = gi & 3u;
                let cout = (gi >> 2u) & 3u;
                let mag = ov.p1 * exp2(f32((gi >> 4u) & 3u) - 1.0);
                let x = s[cin];
                let th = max(ov.p2, 0.0);
                let f = u32(clamp(ov.p0, 0.0, 5.0) + 0.5);
                if (f == 0u) {
                    // SWITCH: o sinal muda de canal e a entrada fica travada aqui.
                    if (cin != cout) {
                        emit[k][cout] = mag * x;
                        gate[k][cin] = 0.0;
                    }
                } else if (f == 1u) {
                    if (cin != cout) { emit[k][cout] = mag * x; }
                } else if (f == 2u) {
                    emit[k][cout] = -mag * x;
                } else if (f == 3u) {
                    // GATE que fecha: entrada alta trava o canal de saída aqui.
                    if (abs(x) > th) { gate[k][cout] = 0.0; }
                } else if (f == 4u) {
                    // GATE que abre: a saída só passa com a entrada alta.
                    if (abs(x) <= th) { gate[k][cout] = 0.0; }
                } else {
                    emit[k][cout] = mag * sign(x) * max(abs(x) - th, 0.0);
                }
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
    var prev = vec4<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let here = signals[base + k];
        // As portas dos relés travam o que um resíduo PASSA aos vizinhos (ele
        // próprio continua a receber e a ler o canal).
        var next = vec4<f32>(0.0);
        if (k + 1u < n) { next = signals[base + k + 1u] * gate[k + 1u]; }
        let pr = aa_props[body_get(slot, k)];
        // Condução por canal (α, β, γ, δ): peso do vizinho do lado N e do C.
        var cn = vec4<f32>(pr.cond_alpha_n, pr.cond_beta_n, pr.cond_gamma_n, pr.cond_delta_n);
        var cc = vec4<f32>(pr.cond_alpha_c, pr.cond_beta_c, pr.cond_gamma_c, pr.cond_delta_c);
        // Um órgão tem a sua própria condução (v3: cada parte a sua).
        let ok = organ_get(slot, k);
        if (ok != 0u) {
            let op = organ_cost(ok);
            if (op.cond_alpha_n < 1e8) {
                cn = vec4<f32>(op.cond_alpha_n, op.cond_beta_n, op.cond_gamma_n, op.cond_delta_n);
                cc = vec4<f32>(op.cond_alpha_c, op.cond_beta_c, op.cond_gamma_c, op.cond_delta_c);
            }
        }
        // Modos uniformes: difusão isotrópica (metade de cada vizinho) ou
        // condução direcional (só do lado N).
        if (params.signal_mode >= 1.5 && params.signal_mode < 2.5) {
            cn = vec4<f32>(1.0);
            cc = vec4<f32>(0.0);
        } else if (params.signal_mode >= 0.5 && params.signal_mode < 1.5) {
            cn = vec4<f32>(0.5);
            cc = vec4<f32>(0.5);
        } else if (params.signal_mode >= 2.5) {
            cn = vec4<f32>(1.0);
            cc = vec4<f32>(0.0);
        }
        let incoming = cn * prev + cc * next;
        var s = SIGNAL_DECAY * incoming + emit[k];
        var lim = SIGNAL_MAX;
        if (params.signal_mode < 0.5) {
            // Modo 0 = a dinâmica do v3: perda de 0,3% por salto, 75% do
            // valor novo + 25% do antigo (cada parte tem memória), ±1.
            s = mix(here, V3_SIGNAL_DECAY * incoming + emit[k], V3_SIGNAL_UPDATE);
            lim = 1.0;
        }
        signals[base + k] = clamp(s, vec4<f32>(-lim), vec4<f32>(lim));
        prev = here * gate[k];
    }
    return bend * BEND_COST;
}
