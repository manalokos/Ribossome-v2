// CICLO DE VIDA: sementes, deriva, comer, morte, emparelhamento, nascimento.
//
// Matéria exata: o genoma e os complementos capturados são monómeros reais
// tirados da grelha; voltam todos, contados ao quantum. (No v3: devoluções
// arredondadas ao acaso, défices de mutação "não pagos", sementes da geração
// 0 criadas do nada, e uma morte entre o fim do emparelhamento e o reset que
// devolvia os complementos duas vezes. Nada disso existe aqui.)
// A energia é ativação colhida: não é matéria e evapora na morte.

const S_SPAWN: u32 = 5u;
const S_DEATH: u32 = 6u;
const S_PAIR: u32 = 7u;
const S_BIRTH: u32 = 8u;
const S_EAT: u32 = 3u << 16u;       // + índice do resíduo
const S_MUT: u32 = 4u << 16u;       // + índice da base

const SPAWN_GATHER_RADIUS: i32 = 16;
// Emparelhamento (v3): no máximo 8 tentativas por passo; alcance 2,5 (na
// prática "a célula debaixo do resíduo").
const PAIRING_MAX_PER_STEP: u32 = 8u;
// Alcance da captura à volta do resíduo (mundo): ~1 célula (30 unidades).
// (Era 2,5: na prática só a célula do próprio resíduo.)
const PAIRING_REACH: f32 = 30.0;
// Mortalidade (v3): frio ×0,1, quente ×10 (T >= 8); risco UV independente da energia.
const COLD_DEATH_MULT: f32 = 0.1;
const HOT_DEATH_MULT: f32 = 10.0;
const UV_HAZARD_SCALE: f32 = 0.001;
const MIN_GENE_LEN: u32 = 6u;
const S_BROWN: u32 = 9u;
const S_BIOTURB: u32 = 10u;
// Passos da média da velocidade de natação (o avanço que o ganho amplifica).
const SWIM_AVG_STEPS: f32 = 100.0;
const S_PHOTOSYS: u32 = 7u << 16u;   // + índice do resíduo
const S_CHEMO: u32 = 9u << 16u;      // + índice do resíduo
// Quimiossíntese: fração do redutor da célula do fluido que cada órgão
// consome por passo (× ganho × eficiência).
const CHEMO_TAKE: f32 = 0.02;
// Fotossistema: o rendimento (energia por unidade de luz absorvida) é
// params.photo_yield. Um fotossistema sozinho absorve 1 − e^−0,15 ≈ 14% da
// luz que lhe chega. O modo reciclador reativa com probabilidade
// photo_yield·potência/food_power: a mesma energia por luz nos dois modos.
// Ciclo catalítico: probabilidade por passo de hidrolisar o ligando ligado e
// de soltar o produto (taxas globais, iguais para todos).
const MOTOR_P_HYDROLYSIS: f32 = 0.2;
const MOTOR_P_RELEASE: f32 = 0.1;
// Difusioforese (v3): limite de velocidade por passo, em unidades do mundo.
const PHORETIC_MAX_STEP: f32 = 3.0;

fn genome_get(slot: u32, i: u32) -> u32 {
    return (genomes[slot * GENOME_WORDS + i / 16u] >> ((i % 16u) * 2u)) & 3u;
}

fn gget(g: ptr<function, array<u32, 16>>, i: u32) -> u32 {
    return ((*g)[i / 16u] >> ((i % 16u) * 2u)) & 3u;
}

fn gset(g: ptr<function, array<u32, 16>>, i: u32, b: u32) {
    let w = i / 16u;
    let sh = (i % 16u) * 2u;
    (*g)[w] = ((*g)[w] & ~(3u << sh)) | (b << sh);
}

// ---- Pilha de slots livres (pop e push nunca no mesmo despacho) ----
fn slot_pop() -> u32 {
    var slot = 0xFFFFFFFFu;
    loop {
        let top = atomicLoad(&life_counters[LC_FREE_TOP]);
        if (top == 0u) { break; }
        let r = atomicCompareExchangeWeak(&life_counters[LC_FREE_TOP], top, top - 1u);
        if (r.exchanged) { slot = free_slots[top - 1u]; break; }
    }
    return slot;
}

fn slot_push(slot: u32) {
    let i = atomicAdd(&life_counters[LC_FREE_TOP], 1u);
    free_slots[i] = slot;
}

fn world_to_cell(p: vec2<f32>) -> u32 {
    let c = clamp(vec2<i32>(floor(p / f32(WORLD_UNITS_PER_CELL))), vec2<i32>(0), vec2<i32>(i32(GRID_SIZE) - 1));
    return u32(c.y) * GRID_SIZE + u32(c.x);
}

// Tira um monómero ATIVADO do canal `ch` da célula mais próxima que o tenha.
fn take_nearest(cx: i32, cy: i32, ch: u32) -> bool {
    for (var r = 0; r <= SPAWN_GATHER_RADIUS; r++) {
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                if (abs(dx) != r && abs(dy) != r) { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if (x < 0 || y < 0 || x >= i32(GRID_SIZE) || y >= i32(GRID_SIZE)) { continue; }
                if (chem_take_state_one((u32(y) * GRID_SIZE + u32(x)) * 4u + ch, false)) { return true; }
            }
        }
    }
    return false;
}

// Tira um monómero do canal `ch` (gasto primeiro, depois ativado) do 3×3 à
// volta de `cell` (mutações: a base nova vem do meio, como no v3).
fn take_around(cell: u32, ch: u32) -> bool {
    let cx = i32(cell % GRID_SIZE);
    let cy = i32(cell / GRID_SIZE);
    for (var s = 0u; s < 2u; s++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let x = cx + dx;
                let y = cy + dy;
                if (x < 0 || y < 0 || x >= i32(GRID_SIZE) || y >= i32(GRID_SIZE)) { continue; }
                if (chem_take_state_one((u32(y) * GRID_SIZE + u32(x)) * 4u + ch, s == 0u)) { return true; }
            }
        }
    }
    return false;
}

fn new_agent(slot: u32, pos: vec2<f32>, rot: f32, energy: f32, gene_len: u32, generation: u32, parent: u32) {
    var a: Agent;
    a.pos_x = pos.x;
    a.pos_y = pos.y;
    a.vel_x = 0.0;
    a.vel_y = 0.0;
    a.rot = rot;
    a.energy = energy;
    a.alive = 1u;
    a.gene_len = gene_len;
    a.pair_count = 0u;
    a.generation = generation;
    a.age = 0u;
    a.parent = parent;
    a.id = atomicAdd(&life_counters[LC_NEXT_ID], 1u);
    var span = 0u;
    a.body_len = translate_agent(slot, gene_len, &span);
    a.coding_span = span;
    a.radius = contact_radius(slot, a.body_len);
    agents[slot] = a;
    bonds_clear(slot);
}

// SEMENTES: cada pedido monta o genoma com os monómeros ATIVADOS mais
// próximos, pela ordem da distância (a sequência espelha a sopa local).
// Dentro de uma célula, a base é sorteada em proporção do que lá há.
// Se faltarem bases ou slots, devolve exatamente o que tirou.
@compute @workgroup_size(64)
fn spawn_seeds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if (ri >= params.spawn_count) { return; }
    let req = spawn_requests[ri];
    let want = clamp(req.gene_len, 3u, MAX_GENE_LEN);
    let c0 = world_to_cell(vec2<f32>(req.pos_x, req.pos_y));
    let cx = i32(c0 % GRID_SIZE);
    let cy = i32(c0 / GRID_SIZE);

    var g = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var taken = vec4<u32>(0u);
    var n = 0u;
    var ok = true;

    // GENOMA ESCOLHIDO: cada base, por ordem, é tirada da sopa (a mais
    // próxima do tipo pedido). Matéria exata como nas outras sementes.
    if ((req.flags & 2u) != 0u) {
        var words = array<u32, 16>(req.genome0, req.genome1, req.genome2, req.genome3, req.genome4, req.genome5,
            req.genome6, req.genome7, req.genome8, req.genome9, req.genome10, req.genome11, req.genome12,
            req.genome13, req.genome14, req.genome15);
        for (var i = 0u; i < want; i++) {
            let b = (words[i / 16u] >> ((i % 16u) * 2u)) & 3u;
            if (!take_nearest(cx, cy, b)) { ok = false; break; }
            gset(&g, n, b);
            taken[b] += 1u;
            n += 1u;
        }
    }

    // Opção: começar por AUG, com A, U e G também tirados da vizinhança.
    if ((req.flags & 3u) == 1u) {
        for (var b = 0u; b < 3u; b++) {
            if (!take_nearest(cx, cy, b)) { ok = false; break; }
            gset(&g, n, b);
            taken[b] += 1u;
            n += 1u;
        }
    }

    for (var r = 0; r <= SPAWN_GATHER_RADIUS && ok && n < want; r++) {
        for (var dy = -r; dy <= r && n < want; dy++) {
            for (var dx = -r; dx <= r && n < want; dx++) {
                if (abs(dx) != r && abs(dy) != r) { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if (x < 0 || y < 0 || x >= i32(GRID_SIZE) || y >= i32(GRID_SIZE)) { continue; }
                let cell = u32(y) * GRID_SIZE + u32(x);
                var guard = 0u;
                loop {
                    if (n >= want || guard >= 64u) { break; }
                    guard += 1u;
                    var avail = vec4<u32>(0u);
                    for (var ch = 0u; ch < 4u; ch++) { avail[ch] = chem_act_count(cell, ch); }
                    let tot = avail.x + avail.y + avail.z + avail.w;
                    if (tot == 0u) { break; }
                    var u = u32(rng_f4(ri, params.epoch, (S_SPAWN << 16u) + n).x * f32(tot));
                    var b = 3u;
                    for (var ch = 0u; ch < 4u; ch++) {
                        if (u < avail[ch]) { b = ch; break; }
                        u -= avail[ch];
                    }
                    if (chem_take_state_one(cell * 4u + b, false)) {
                        gset(&g, n, b);
                        taken[b] += 1u;
                        n += 1u;
                    }
                }
            }
        }
    }

    var slot = 0xFFFFFFFFu;
    if (ok && n >= want) { slot = slot_pop(); }
    if (slot == 0xFFFFFFFFu) {
        let back = chem_open_cell(c0);
        for (var ch = 0u; ch < 4u; ch++) { chem_add_state(back, ch, taken[ch], false); }
        atomicAdd(&life_counters[LC_SPAWN_FAILED], 1u);
        return;
    }
    for (var w = 0u; w < GENOME_WORDS; w++) { genomes[slot * GENOME_WORDS + w] = g[w]; }
    new_agent(slot, vec2<f32>(req.pos_x, req.pos_y), rng_f4(ri, params.epoch, S_SPAWN).y * 6.2831853,
        params.spawn_energy, n, 0u, 0xFFFFFFFFu);
    atomicAdd(&life_counters[LC_SPAWNED], 1u);
}

// Matéria presa num agente, por canal: genoma + complementos capturados
// (complemento de Watson-Crick das primeiras pair_count bases).
fn agent_matter(slot: u32) -> vec4<u32> {
    var m = vec4<u32>(0u);
    let a = agents[slot];
    for (var i = 0u; i < a.gene_len; i++) {
        let b = genome_get(slot, i);
        m[b] += 1u;
        if (i < a.pair_count) { m[b ^ 1u] += 1u; }
    }
    return m;
}

// MORTE: toda a matéria volta ao meio, repartida pelas células onde estão
// os resíduos do corpo (os restos ficam onde o corpo estava, sem despejar
// tudo numa célula só). O genoma volta GASTO; os complementos já capturados
// para a cópia em curso voltam ATIVADOS: foram tirados ativados e ainda só
// estavam emparelhados, não ligados (a ativação não foi gasta).
fn die(slot: u32, a_in: Agent) {
    var a = a_in;
    var m_act = vec4<u32>(0u);
    for (var i = 0u; i < min(a.pair_count, a.gene_len); i++) { m_act[genome_get(slot, i) ^ 1u] += 1u; }
    let m_spent = agent_matter(slot) - m_act;
    let n = max(a.body_len, 1u);
    for (var k = 0u; k < n; k++) {
        var pk = vec2<f32>(a.pos_x, a.pos_y);
        if (a.body_len > 0u) { pk = residue_world(slot, a, k); }
        let cell = chem_open_cell(world_to_cell(pk));
        for (var ch = 0u; ch < 4u; ch++) {
            // Parte igual por resíduo; o resto da divisão vai para os primeiros.
            let s_sp = m_spent[ch] / n + select(0u, 1u, k < m_spent[ch] % n);
            if (s_sp > 0u) { chem_add_state(cell, ch, s_sp, true); }
            let s_ac = m_act[ch] / n + select(0u, 1u, k < m_act[ch] % n);
            if (s_ac > 0u) { chem_add_state(cell, ch, s_ac, false); }
        }
    }
    a.alive = 0u;
    agents[slot] = a;
    slot_push(slot);
    atomicAdd(&life_counters[LC_DEATHS], 1u);
    if (a.energy < 1.0) { atomicAdd(&life_counters[LC_STARVED], 1u); }
}

fn energy_capacity(slot: u32, a: Agent) -> f32 {
    // v3: cada aminoácido guarda 1 (o RNA nu guarda 1); o armazenamento soma.
    return max(f32(a.body_len), 1.0) + organ_capacity(slot, a.body_len);
}

@compute @workgroup_size(64)
fn agents_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }

    // ---- SINAIS INTERNOS (sensores, relógio, relé) e custo dos músculos ----
    // Capacidade de energia: depende só dos órgãos; calcula-se uma vez.
    let cap = energy_capacity(slot, a);
    a.energy -= signals_step(slot, a, cap);

    // ---- JUNTAS: dobragem ao nascer, depois agitação térmica e músculos ----
    let t_here = temp_in[fluid_index_at_world(vec2<f32>(a.pos_x, a.pos_y))];
    let kt_here = params.thermal_kt * (1.0 + t_here / 12.0);
    // METABOLISMO (Q10): toda a química do agente (comer, quimiossíntese,
    // copiar, manter-se) anda mais depressa no quente e mais devagar no frio.
    // A luz não depende da temperatura. m = 1 em T = metabolic_ref.
    var metab = pow(max(params.metabolic_q10, 1e-3), (t_here - params.metabolic_ref) / max(params.metabolic_span, 1e-3));
    // DORMÊNCIA (como as proteínas de hibernação dos ribossomas): cada órgão
    // multiplica o metabolismo do agente pelo seu fator (^intensidade). As
    // variantes fixas atuam sempre; as outras, em proporção do sinal γ ou δ
    // (positivo, até 1) que chega ao órgão. É uma troca, não uma poupança:
    // o agente gasta menos mas também come e copia mais devagar.
    var dorm = 1.0;
    for (var k = 0u; k < a.body_len; k++) {
        let od = organ_get(slot, k);
        if (organ_type(od) != ORGAN_DORMANCY) { continue; }
        let dv = organ_var(od);
        var drive = 1.0;
        if (dv.p1 >= 0.0) {
            drive = clamp(signals[slot * MAX_BODY + k][u32(clamp(dv.p1, 0.0, 3.0))], 0.0, 1.0);
        }
        dorm *= pow(clamp(dv.p0, 0.01, 1.0), organ_gain(od) * drive);
    }
    metab *= max(dorm, DORMANCY_FLOOR);
    // O ganho de natação escala SÓ a translação. A rotação fica a física:
    // escalá-la exagerava o balanço de cada abrir-e-fechar (o corpo rodava
    // muito para um lado e para o outro) e a orientação errada estragava a
    // natação; com a rotação física, um movimento recíproco não desloca nada.
    let js = joints_step(slot, a, kt_here);
    a.energy -= params.bioturbation_cost * f32(js.pushed) + params.motion_cost * js.dissipated;
    // O ganho de natação amplifica só o AVANÇO MÉDIO da natação, não o
    // vaivém de cada batida (que se anula num ciclo; multiplicá-lo fazia os
    // agentes andar ~25× mais de lado do que em frente). O transporte pela
    // água (corrente nos resíduos, com rotação) entra sem ganho.
    var swim_v = vec2<f32>(0.0);
    var aux = rna_tail[slot * 2u + 1u];
    if (a.age == 0u) { aux = vec4<f32>(0.0); } // slot reutilizado: sem herança
    // Orientação a meio do passo (o corpo roda durante o passo).
    // Transporte pela água × flow_coupling (1 = físico).
    let fc = clamp(params.flow_coupling, 0.0, 1.0);
    let rot_mid = a.rot + 0.5 * (js.swim.z + fc * js.flow.z);
    let sv_phys = rotate(js.swim.xy, rot_mid);
    let avg = mix(aux.zw, sv_phys, 1.0 / SWIM_AVG_STEPS);
    aux = vec4<f32>(aux.xy, avg);
    // Avanço médio × ganho + vaivém (o resto) × swim_wobble.
    let sv = max(params.swim_gain, 0.0) * avg + clamp(params.swim_wobble, 0.0, 1.0) * (sv_phys - avg);
    swim_v = sv;
    var flow_w = fc * rotate(js.flow.xy, rot_mid);
    if (a.body_len < 2u && params.fluid_enabled != 0u) {
        // Sem corpo articulado (RNA nu, um resíduo): levado pela água no centro.
        flow_w = fc * water_at(vec2<f32>(a.pos_x, a.pos_y), true);
    }
    // INÉRCIA: a deslocação do passo aproxima-se da alvo (natação +
    // corrente) com peso 1/(1 + inércia × massa relativa). vel guarda a
    // velocidade do passo anterior. Sem inércia (0) é a de sempre.
    let mass = body_mass(slot, a.body_len);
    let dt_s = max(params.dt, 1e-6);
    let want = sv + flow_w;
    var step = want;
    if (params.inertia > 0.0 && a.age > 0u) {
        let alpha = 1.0 / (1.0 + params.inertia * mass / MASS_BODY_REF);
        step = mix(vec2<f32>(a.vel_x, a.vel_y) * dt_s, want, alpha);
    }
    let np0 = clamp(vec2<f32>(a.pos_x, a.pos_y) + step, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np0)) < GAMMA_SOLID_THRESHOLD) {
        a.pos_x = np0.x;
        a.pos_y = np0.y;
    } else {
        step = vec2<f32>(0.0); // bateu na rocha: para
    }
    a.rot += js.swim.z + fc * js.flow.z + js.phi;
    a.vel_x = step.x / dt_s;
    a.vel_y = step.y / dt_s;
    a.age += 1u;
    // (zw = média da velocidade de natação; xy = curvatura dos fios, a seguir)
    rna_tail[slot * 2u + 1u] = aux;
    update_rna_tails(slot, a);

    // ---- BIOTURBAÇÃO: um resíduo (ao acaso) que atravessa ENTULHO empurra
    // um grão para a célula seguinte, na direção em que se move (translação
    // + rotação do corpo). Só entulho (1–2 grãos) e só para onde continua a
    // ser entulho: os agentes não fazem nem partem rocha. Custa energia. O
    // grão move-se com gamma_move_one: matéria e terreno conservam-se. ----
    if (params.bioturbation > 0.0 && a.body_len > 0u) {
        let q = rng_f4(a.id, params.epoch, S_BIOTURB);
        let k = min(u32(q.x * f32(a.body_len)), a.body_len - 1u);
        let rel = rotate(body_pos[slot * MAX_BODY + k], a.rot);
        let v_res = swim_v + js.swim.z * vec2<f32>(-rel.y, rel.x);
        let cells_moved = length(v_res) / f32(WORLD_UNITS_PER_CELL);
        let p_push = clamp(params.bioturbation * cells_moved * f32(a.body_len), 0.0, 1.0);
        if (q.y < p_push && a.energy > params.bioturbation_cost) {
            let rw = vec2<f32>(a.pos_x, a.pos_y) + rel;
            let src = world_to_cell(rw);
            let g = gamma_count(src);
            if (g > 0u && g < GAMMA_SOLID_THRESHOLD) {
                // Vizinho no eixo dominante do movimento.
                var d = vec2<i32>(select(-1, 1, v_res.x > 0.0), 0);
                if (abs(v_res.y) > abs(v_res.x)) { d = vec2<i32>(0, select(-1, 1, v_res.y > 0.0)); }
                let c = vec2<i32>(i32(src % GRID_SIZE), i32(src / GRID_SIZE)) + d;
                if (all(c >= vec2<i32>(0)) && all(c < vec2<i32>(i32(GRID_SIZE)))) {
                    let dst = u32(c.y) * GRID_SIZE + u32(c.x);
                    if (gamma_count(dst) + 1u < GAMMA_SOLID_THRESHOLD && gamma_move_one(src, dst)) {
                        a.energy -= params.bioturbation_cost;
                    }
                }
            }
        }
    }

    // (A deriva pela água entra no RFT, resíduo a resíduo: ver joints_step.)

    // ---- COMER = CICLO CATALÍTICO de cada resíduo ----
    // livre --liga um ativado--> ligado --hidrolisa--> produto --solta--> livre
    // A ligação tem probabilidade proporcional à PROPENSÃO CATALÍTICA medida
    // (M-CSA), à comida na célula e à fome. Na hidrólise o monómero fica no
    // lugar, gasto, e a ativação vira energia do agente. Cada estado desvia
    // o ângulo da junta (fold.wgsl): o ciclo irreversível é um motor.
    // Fluxo de consumo por direção (difusioforese): soma de taxa × direção
    // do resíduo a partir do centro de massa. Consumo simétrico cancela.
    var phoretic = vec2<f32>(0.0);
    var p = vec2<f32>(a.pos_x, a.pos_y);
    // Seno e cosseno da orientação, uma vez; e contagem dos resíduos em água
    // livre (para a sedimentação, mais abaixo) aproveitando a célula de cada um.
    let cr_e = cos(a.rot);
    let sr_e = sin(a.rot);
    var in_water = 0u;
    for (var k = 0u; k < a.body_len; k++) {
        let lp = body_pos[slot * MAX_BODY + k];
        let rw = p + vec2<f32>(cr_e * lp.x - sr_e * lp.y, sr_e * lp.x + cr_e * lp.y);
        let cell = world_to_cell(rw);
        if (gamma_count(cell) == 0u) { in_water += 1u; }
        var avail = vec4<u32>(0u);
        for (var ch = 0u; ch < 4u; ch++) { avail[ch] = chem_act_count(cell, ch); }
        let tot = avail.x + avail.y + avail.z + avail.w;
        // ESPECIFICIDADE: cada aminoácido prefere certos canais (AA_SUBSTRATE,
        // soma 1). Substrato efetivo = 4·Σ afinidade·disponível (sem
        // preferência dá o total, como antes).
        let aa_k = body_get(slot, k);
        let prk = aa_props[aa_k];
        var aff = vec4<f32>(prk.sub_a, prk.sub_u, prk.sub_g, prk.sub_c);
        let om = organ_get(slot, k);
        if (organ_type(om) == ORGAN_MOUTH) {
            // Viés da boca: + prefere A/U, − prefere G/C (renormalizado).
            let b = clamp(organ_var(om).p1, -0.95, 0.95);
            aff *= vec4<f32>(1.0 + b, 1.0 + b, 1.0 - b, 1.0 - b);
            aff /= max(aff.x + aff.y + aff.z + aff.w, 1e-6);
        }
        let w_avail = aff * vec4<f32>(avail);
        let eff = 4.0 * (w_avail.x + w_avail.y + w_avail.z + w_avail.w);
        // Uma enzima real não sabe se a célula está cheia: por omissão
        // catalisa sempre que há substrato e a energia a mais perde-se como
        // calor. (A regulação pela fome do v3 fica como opção.)
        let hunger = select(1.0, clamp(1.0 - a.energy / cap, 0.0, 1.0), params.hunger_regulation != 0u);
        let pe = clamp(params.uptake_rate * metab * prk.catalytic * organ_catalysis_mult(slot, k) * eff * hunger, 0.0, 1.0);
        let si = slot * MAX_BODY + k;
        let st = joint_state[si];
        let r = rng_f4(a.id, params.epoch, S_EAT + k);
        if (st == 0u) {
            let full = params.hunger_regulation != 0u && a.energy + params.food_power > cap;
            if (r.x < pe && !full) {
                joint_state[si] = 1u;
                let rc = rw - p;
                let rl = length(rc);
                if (rl > 1e-4) { phoretic += rc / rl * pe; }
            }
        } else if (st == 1u) {
            if (r.x < MOTOR_P_HYDROLYSIS) {
                // Canal escolhido pela afinidade × disponível.
                var u = r.y * (w_avail.x + w_avail.y + w_avail.z + w_avail.w);
                var b = 3u;
                for (var ch = 0u; ch < 4u; ch++) {
                    if (u < w_avail[ch]) { b = ch; break; }
                    u -= w_avail[ch];
                }
                if (tot > 0u && chem_spend_one(cell * 4u + b)) {
                    a.energy += params.food_power;
                    joint_state[si] = 2u;
                } else {
                    joint_state[si] = 0u;
                }
            }
        } else if (r.x < MOTOR_P_RELEASE) {
            joint_state[si] = 0u;
        }

        // QUIMIOSSÍNTESE: consome o redutor das fumarolas onde está (o que
        // consome sai do campo). Modo 0: energia; modo 1: reativa gastos da
        // célula (a mesma energia por redutor nos dois modos).
        let oc = organ_get(slot, k);
        if (organ_type(oc) == ORGAN_CHEMO && params.fluid_enabled != 0u) {
            let fi = fluid_index_at_world(rw);
            let cv = organ_var(oc);
            let take = min(CHEMO_TAKE * metab, 1.0) * redox_in[fi] * organ_gain(oc) * max(cv.p1, 0.0);
            if (take > 0.0) {
                atomicAdd(&redox_eaten[fi], u32(take * REDOX_FP));
                let rec = clamp(cv.p0, 0.0, 1.0);
                // TRANSBORDO (como no fotossistema): o que já não cabe no
                // agente cheio vai reativar gastos da célula.
                let gain = (1.0 - rec) * params.chemo_yield * take;
                let kept = min(gain, max(cap - a.energy, 0.0));
                a.energy += kept;
                let to_food = rec * params.chemo_yield * take + (gain - kept);
                if (to_food > 0.0) {
                    let q = rng_f4(a.id, params.epoch, S_CHEMO + k);
                    if (q.x < clamp(to_food / max(params.food_power, 1e-3), 0.0, 1.0)) {
                        let ch0 = min(u32(q.y * 4.0), 3u);
                        for (var t = 0u; t < 4u; t++) {
                            if (chem_activate_one(cell * 4u + (ch0 + t) % 4u)) { break; }
                        }
                    }
                }
            }
        }

        // FOTOSSISTEMA: capta a luz onde está. Modo 0: energia para o agente
        // (produtor primário). Modo 1: usa-a para REATIVAR os gastos da
        // célula (recicla comida para si e para os outros). Mais luz também
        // é mais dano UV: há um compromisso.
        let ok = organ_get(slot, k);
        if (organ_type(ok) == ORGAN_PHOTOSYSTEM) {
            // LUZ CONSERVADA: os fotossistemas de uma célula da luz absorvem,
            // juntos, a luz que lá CHEGA × (1 − exp(−τ·S)) (S = absorventes
            // na célula, em triptofanos equivalentes) e repartem-na. Um
            // sozinho recebe ~o de sempre; muitos juntos dividem a mesma luz
            // (e fazem sombra aos de baixo): a produção por área é limitada
            // pelo sol, o que dá a capacidade de carga dos produtores.
            let lx = (cell % GRID_SIZE) / LIGHT_DIV;
            let ly = (cell / GRID_SIZE) / LIGHT_DIV;
            var incoming = params.sun_now;
            if (ly + 1u < LIGHT_SIZE) { incoming = light_above(lx, ly + 1u); }
            let s_abs = max(f32(atomicLoad(&shade_grid[ly * LIGHT_SIZE + lx])) / f32(SHADE_ONE), 1.0);
            let share = (1.0 - exp(-AGENT_UV_ABSORB * s_abs)) / s_abs;
            let og = organ_gain(ok);
            let pv = organ_var(ok);
            let power = max(incoming, 0.0) * max(params.uv_strength, 0.0) * share * og * max(pv.p1, 0.0);
            // Variante: p0 = fração da luz para reciclar, p1 = eficiência.
            let recycle = clamp(pv.p0, 0.0, 1.0);
            // TRANSBORDO (como o fitoplâncton que exsuda o carbono que fixa a
            // mais): a energia que já não cabe no agente cheio não se perde,
            // vai reativar gastos da célula. De dia, quem está cheio enche o
            // meio de ativados, que de noite pode voltar a comer.
            let gain = (1.0 - recycle) * params.photo_yield * power;
            let kept = min(gain, max(cap - a.energy, 0.0));
            a.energy += kept;
            let to_food = recycle * params.photo_yield * power + (gain - kept);
            if (to_food > 0.0) {
                let q = rng_f4(a.id, params.epoch, S_PHOTOSYS + k);
                // Reativar um monómero guarda food_power de energia: custa a
                // luz que daria essa energia no modo produtor (senão
                // reciclar + comer criava energia do nada).
                if (q.x < clamp(to_food / max(params.food_power, 1e-3), 0.0, 1.0)) {
                    let ch0 = min(u32(q.y * 4.0), 3u);
                    for (var t = 0u; t < 4u; t++) {
                        if (chem_activate_one(cell * 4u + (ch0 + t) % 4u)) { break; }
                    }
                }
            }
        }
    }
    a.energy = clamp(a.energy, 0.0, cap) - params.maintenance_cost * metab * (f32(a.body_len) + organ_upkeep(slot, a.body_len));

    // ---- SEDIMENTAÇÃO (Stokes): afunda ∝ √n × massa média por resíduo (os
    // órgãos pesados, como o armazenamento, afundam mais), só a parte em
    // água livre. ----
    if (params.sedimentation > 0.0) {
        let free_frac = select(1.0, f32(in_water) / f32(max(a.body_len, 1u)), a.body_len > 0u);
        let nb = f32(max(a.body_len, 1u));
        let dens = body_mass(slot, a.body_len) / (nb * MASS_RESIDUE_REF);
        let fall = params.sedimentation * sqrt(nb) * dens * free_frac;
        let ns = vec2<f32>(p.x, max(p.y - fall, 0.0));
        if (gamma_count(world_to_cell(ns)) < GAMMA_SOLID_THRESHOLD) {
            p = ns;
            a.pos_x = p.x;
            a.pos_y = p.y;
        }
    }

    // ---- DIFUSIOFORESE (v3): consumo assimétrico empurra o corpo para o
    // lado onde consome. (O sentido real depende de a superfície atrair ou
    // repelir o soluto; o v3 usava este.) ----
    var dp = phoretic * params.phoretic_gain;
    let dl = length(dp);
    if (dl > PHORETIC_MAX_STEP) { dp *= PHORETIC_MAX_STEP / dl; }

    // ---- MOVIMENTO BROWNIANO: agitação térmica. Translação ∝ 1/√raio por
    // passo (D ∝ 1/raio, Stokes-Einstein); rotação D_r ∝ 1/raio³. O raio
    // conta-se em resíduos (√n para uma cadeia enrolada). ----
    let radius = max(sqrt(f32(max(a.body_len, 1u))), 1.0);
    let bq = rng_f4(a.id, params.epoch, S_BROWN);
    // Box-Muller: dois desvios normais.
    let bm_r = sqrt(-2.0 * log(max(bq.x, 1e-7)));
    let gauss = vec2<f32>(bm_r * cos(6.2831853 * bq.y), bm_r * sin(6.2831853 * bq.y));
    dp += gauss * params.brownian / sqrt(radius);
    a.rot += (bq.z * 2.0 - 1.0) * 1.7320508 * 0.15 / pow(radius, 1.5);
    let np2 = clamp(p + dp, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np2)) < GAMMA_SOLID_THRESHOLD) {
        p = np2;
        a.pos_x = p.x;
        a.pos_y = p.y;
    }

    // ---- MORTE (v3): base ÷ energia × temperatura + risco UV ----
    let cell_here = world_to_cell(p);
    let light = uv_light_at_cell(cell_here % GRID_SIZE, cell_here / GRID_SIZE);
    let uv_mult = mix(1.0, max(params.uv_damage, 1.0), light);
    let fi = fluid_index_at_world(p);
    let wt = temp_in[fi];
    var thermal = mix(COLD_DEATH_MULT, 1.0, clamp(wt / 2.0, 0.0, 1.0));
    if (wt > 2.0) { thermal = mix(1.0, HOT_DEATH_MULT, clamp((wt - 2.0) / 6.0, 0.0, 1.0)); }
    let uv_hazard = params.death_probability * (uv_mult - 1.0) * UV_HAZARD_SCALE;
    // DESNATURAÇÃO: acima do limiar, o calor desfaz as proteínas (não
    // depende de estar bem alimentado); a composição do corpo protege.
    var heat_hazard = 0.0;
    if (wt > params.denature_temp && params.heat_kill > 0.0) {
        var stab = 0.0;
        for (var k = 0u; k < a.body_len; k++) { stab += aa_props[body_get(slot, k)].thermo; }
        stab = clamp(stab / f32(max(a.body_len, 1u)), 0.0, 1.0);
        heat_hazard = params.heat_kill * (wt - params.denature_temp) / 10.0 * (1.0 - stab);
    }
    // A reserva de energia protege até death_energy_cap (0 = sem teto, v3).
    var e_eff = max(a.energy, 0.01);
    if (params.death_energy_cap > 0.0) { e_eff = min(e_eff, params.death_energy_cap); }
    let p_death = clamp(params.death_probability / e_eff * thermal + uv_hazard + heat_hazard, 0.0, 1.0);
    if (a.energy <= 0.0 || rng_f4(a.id, params.epoch, S_DEATH).x < p_death) {
        die(slot, a);
        return;
    }

    // ---- EMPARELHAMENTO (v3): captura complementos ATIVADOS da vizinhança ----
    // O molde é o genoma; a captura é na célula de um resíduo ao acaso (ou
    // do próprio agente, se for RNA nu). Base a base, pela ordem do genoma.
    if (a.pair_count < a.gene_len && a.energy > 1.0 + params.pairing_cost) {
        let rr = rng_f4(a.id, params.epoch, S_PAIR);
        let pr = params.pairing_rate * metab;
        var attempts = u32(pr);
        if (rr.x < fract(pr)) { attempts += 1u; }
        attempts = min(attempts, PAIRING_MAX_PER_STEP);
        for (var t = 0u; t < attempts && a.pair_count < a.gene_len; t++) {
            let q = rng_f4(a.id, params.epoch, (S_PAIR << 16u) + t);
            var site = p;
            if (a.body_len > 0u) {
                site = residue_world(slot, a, min(u32(q.x * f32(a.body_len)), a.body_len - 1u));
            }
            let ang = q.y * 6.2831853;
            site += vec2<f32>(cos(ang), sin(ang)) * q.z * PAIRING_REACH;
            let comp = genome_get(slot, a.pair_count) ^ 1u;
            if (a.energy < 1.0 + params.pairing_cost) { break; }
            // Tentativas independentes: uma falha (não havia o complemento
            // ali) não impede as outras deste passo, noutros sítios.
            if (!chem_take_state_one(world_to_cell(site) * 4u + comp, false)) { continue; }
            a.pair_count += 1u;
            a.energy -= params.pairing_cost;
        }
    }
    agents[slot] = a;
}

// Índice da célula do fluido num ponto do mundo.
fn fluid_index_at_world(p: vec2<f32>) -> u32 {
    let f = clamp(vec2<i32>(floor(p / SIM_SIZE * f32(FLUID_SIZE))), vec2<i32>(0), vec2<i32>(i32(FLUID_SIZE) - 1));
    return u32(f.y) * FLUID_SIZE + u32(f.x);
}

// O metabolismo nunca desce abaixo disto por dormência (um agente nunca
// fica totalmente parado: continua a pagar e a poder acordar).
const DORMANCY_FLOOR: f32 = 0.1;

// NASCIMENTO: quando o emparelhamento está completo, os complementos
// capturados formam o filho = complemento reverso do genoma (v3). Mutações
// com a matéria reconciliada AO QUANTUM: a base nova é tirada do meio e a
// velha devolvida (gasta); inserções e duplicações tiram as bases do meio;
// remoções devolvem-nas. Se o meio não tiver a base, a mutação não acontece.
// Tipos: pontuais, indels de 1–3 bases (1–2 = frameshift) e duplicações em
// tandem de 3–24 bases.
@compute @workgroup_size(64)
fn agents_birth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u || a.gene_len == 0u || a.pair_count < a.gene_len) { return; }

    let L = a.gene_len;
    let cell = chem_open_cell(world_to_cell(vec2<f32>(a.pos_x, a.pos_y)));
    // REVISÃO: os órgãos de revisão do pai (enzimas que corrigem a cópia)
    // dividem a taxa de mutação por 1 + a soma das suas proteções.
    var protect = 0.0;
    for (var k = 0u; k < a.body_len; k++) {
        let op = organ_get(slot, k);
        if (organ_type(op) == ORGAN_PROOFREAD) { protect += max(organ_var(op).p0, 0.0) * organ_gain(op); }
    }
    let m = clamp(params.mutation_rate / (1.0 + protect), 0.0, 1.0);
    var g = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    for (var i = 0u; i < L; i++) {
        gset(&g, i, genome_get(slot, L - 1u - i) ^ 1u);
    }
    var n = L;
    let mr = rng_f4(a.id, params.epoch, S_BIRTH);

    // Mutações pontuais (por base, probabilidade m; base nova uniforme).
    for (var i = 0u; i < n; i++) {
        let q = rng_f4(a.id, params.epoch, S_MUT + i);
        if (q.x >= m) { continue; }
        let old = gget(&g, i);
        let nb = min(u32(q.y * 4.0), 3u);
        if (nb == old) { continue; }
        if (take_around(cell, nb)) {
            chem_add_state(cell, old, 1u, true);
            gset(&g, i, nb);
        }
    }
    // INDELS (cada um com probabilidade 4m): metade das vezes 3 bases (um
    // codão, mantém a fase de leitura); na outra metade 1 ou 2 bases
    // (FRAMESHIFT: muda a leitura de tudo o que vem a seguir).
    let qi = rng_f4(a.id, params.epoch, S_BIRTH + 20u);
    // Remoção: as bases voltam ao meio, gastas. Fica >= MIN_GENE_LEN.
    let del_len = select(1u + u32(qi.y * 2.0), 3u, qi.x < 0.5);
    if (mr.x < 4.0 * m && n >= MIN_GENE_LEN + del_len) {
        let at = min(u32(mr.y * f32(n - del_len + 1u)), n - del_len);
        for (var k = 0u; k < del_len; k++) { chem_add_state(cell, gget(&g, at + k), 1u, true); }
        for (var i = at; i + del_len < n; i++) { gset(&g, i, gget(&g, i + del_len)); }
        n -= del_len;
        for (var i = n; i < n + del_len; i++) { gset(&g, i, 0u); }
    }
    // Inserção de bases ao acaso, tiradas do meio.
    let ins_len = select(1u + u32(qi.w * 2.0), 3u, qi.z < 0.5);
    if (mr.z < 4.0 * m && n + ins_len <= MAX_GENE_LEN) {
        let q = rng_f4(a.id, params.epoch, S_BIRTH + 1u);
        let at = min(u32(q.x * f32(n + 1u)), n);
        let nb = vec3<u32>(min(u32(q.y * 4.0), 3u), min(u32(q.z * 4.0), 3u), min(u32(q.w * 4.0), 3u));
        var got = 0u;
        for (var k = 0u; k < ins_len; k++) {
            if (!take_around(cell, nb[k])) { break; }
            got += 1u;
        }
        if (got == ins_len) {
            for (var i = n; i > at; i--) { gset(&g, i + ins_len - 1u, gget(&g, i - 1u)); }
            for (var k = 0u; k < ins_len; k++) { gset(&g, at + k, nb[k]); }
            n += ins_len;
        } else {
            for (var k = 0u; k < got; k++) { chem_add_state(cell, nb[k], 1u, true); }
        }
    }
    // DUPLICAÇÃO EM TANDEM (probabilidade 2m): um troço de 3 a 24 bases é
    // copiado logo a seguir a si próprio, com bases tiradas do meio (se faltar
    // alguma, nada acontece e o que se tirou volta). É assim que nascem genes
    // novos: uma cópia mantém a função, a outra pode mudar.
    let qd = rng_f4(a.id, params.epoch, S_BIRTH + 21u);
    if (qd.x < 2.0 * m && n >= 3u) {
        let max_len = min(24u, min(n, MAX_GENE_LEN - n));
        if (max_len >= 3u) {
            let len = 3u + min(u32(qd.y * f32(max_len - 2u)), max_len - 3u);
            let at = min(u32(qd.z * f32(n - len + 1u)), n - len);
            var got = 0u;
            for (var k = 0u; k < len; k++) {
                if (!take_around(cell, gget(&g, at + k))) { break; }
                got += 1u;
            }
            if (got == len) {
                // Abre espaço depois do troço e copia-o.
                for (var i = n; i > at + len; i--) { gset(&g, i + len - 1u, gget(&g, i - 1u)); }
                for (var k = 0u; k < len; k++) { gset(&g, at + len + k, gget(&g, at + k)); }
                n += len;
            } else {
                for (var k = 0u; k < got; k++) { chem_add_state(cell, gget(&g, at + k), 1u, true); }
            }
        }
    }

    let child = slot_pop();
    if (child == 0xFFFFFFFFu) {
        // População cheia: a cópia volta ao meio, exatamente.
        for (var i = 0u; i < n; i++) { chem_add_state(cell, gget(&g, i), 1u, true); }
        a.pair_count = 0u;
        agents[slot] = a;
        return;
    }
    for (var w = 0u; w < GENOME_WORDS; w++) { genomes[child * GENOME_WORDS + w] = g[w]; }
    // Posição: 5–15 unidades ao lado, fora da rocha (8 tentativas). Com uma
    // âncora livre (gemulação), nasce ao pé dela, para fora do corpo do pai.
    let bk = bud_anchor(slot, a);
    var cp = vec2<f32>(a.pos_x, a.pos_y);
    var out_dir = -1.0;
    if (bk != BOND_NONE) {
        let ap = residue_world(slot, a, bk);
        let d = ap - cp;
        out_dir = select(atan2(d.y, d.x), -1.0, dot(d, d) < 1e-6);
        cp = ap;
    }
    for (var t = 0u; t < 8u; t++) {
        let q = rng_f4(a.id, params.epoch, S_BIRTH + 2u + t);
        // Gemulação: para fora (±45°); senão em qualquer direção.
        let ang = select(q.x * 6.2831853, out_dir + (q.x - 0.5) * 1.5708, out_dir > -1.0);
        let tp = clamp(cp + vec2<f32>(cos(ang), sin(ang)) * (5.0 + 10.0 * q.y), vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
        if (gamma_count(world_to_cell(tp)) < GAMMA_SOLID_THRESHOLD) { cp = tp; break; }
    }
    // Energia (v3): metade para o filho, metade fica com o pai.
    let half = a.energy * 0.5;
    new_agent(child, cp, mr.w * 6.2831853, half, n, a.generation + 1u, a.id);
    birth_bond(slot, a, child, bk);
    a.energy -= half;
    a.pair_count = 0u;
    agents[slot] = a;
    atomicAdd(&life_counters[LC_BIRTHS], 1u);
}

// Livro-razão: matéria presa nos agentes vivos, por canal (ledger[8..11]).
@compute @workgroup_size(64)
fn agents_ledger(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents || agents[slot].alive == 0u) { return; }
    let m = agent_matter(slot);
    for (var ch = 0u; ch < 4u; ch++) {
        if (m[ch] > 0u) { atomicAdd(&ledger[8u + ch], m[ch]); }
    }
}

// FIOS DE RNA DAS PONTAS (só visual): cada fio é passivo e mole. Quando a
// ponta do corpo se mexe de lado, o fio fica para trás e curva no sentido
// oposto; parado, endireita aos poucos (relaxação sobreamortecida).
const TAIL_DRAG: f32 = 0.08;
const TAIL_RELAX: f32 = 0.96;
const TAIL_MAX_BEND: f32 = 2.5;

fn update_rna_tails(slot: u32, a: Agent) {
    let n = a.body_len;
    if (n == 0u) { return; }
    let s0 = rna_tail[slot * 2u];
    var bend = rna_tail[slot * 2u + 1u]; // zw = média da natação (não mexer)
    let pn = residue_world(slot, a, 0u);
    let pc = residue_world(slot, a, n - 1u);
    if (a.age > 1u) {
        // Direção para fora de cada ponta e o seu perpendicular.
        var dn = vec2<f32>(-1.0, 0.0);
        var dc = vec2<f32>(1.0, 0.0);
        if (n > 1u) {
            dn = normalize(pn - residue_world(slot, a, 1u) + vec2<f32>(1e-6, 0.0));
            dc = normalize(pc - residue_world(slot, a, n - 2u) + vec2<f32>(1e-6, 0.0));
        }
        let vn = pn - s0.xy;
        let vc = pc - s0.zw;
        // Movimento de lado (no perpendicular esquerdo) curva o fio para o outro lado.
        bend.x = clamp(bend.x * TAIL_RELAX - TAIL_DRAG * dot(vn, vec2<f32>(-dn.y, dn.x)), -TAIL_MAX_BEND, TAIL_MAX_BEND);
        bend.y = clamp(bend.y * TAIL_RELAX - TAIL_DRAG * dot(vc, vec2<f32>(-dc.y, dc.x)), -TAIL_MAX_BEND, TAIL_MAX_BEND);
    } else {
        bend = vec4<f32>(0.0, 0.0, bend.z, bend.w);
    }
    rna_tail[slot * 2u] = vec4<f32>(pn, pc);
    rna_tail[slot * 2u + 1u] = bend;
}
