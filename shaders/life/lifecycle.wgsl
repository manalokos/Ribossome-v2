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
const PAIRING_REACH: f32 = 2.5;
// Mortalidade (v3): frio ×0,1, quente ×10 (T >= 8); risco UV independente da energia.
const COLD_DEATH_MULT: f32 = 0.1;
const HOT_DEATH_MULT: f32 = 10.0;
const UV_HAZARD_SCALE: f32 = 0.001;
const MIN_GENE_LEN: u32 = 6u;
const S_BROWN: u32 = 9u;
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

fn new_agent(slot: u32, pos: vec2<f32>, rot: f32, energy: f32, gene_len: u32, generation: u32) {
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
    a.id = atomicAdd(&life_counters[LC_NEXT_ID], 1u);
    a.body_len = translate_agent(slot, gene_len);
    a.radius = contact_radius(slot, a.body_len);
    agents[slot] = a;
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

    // Opção: começar por AUG, com A, U e G também tirados da vizinhança.
    if ((req.flags & 1u) != 0u) {
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
        params.spawn_energy, n, 0u);
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

fn die(slot: u32, a_in: Agent) {
    var a = a_in;
    let m = agent_matter(slot);
    let cell = chem_open_cell(world_to_cell(vec2<f32>(a.pos_x, a.pos_y)));
    for (var ch = 0u; ch < 4u; ch++) { chem_add_state(cell, ch, m[ch], true); }
    a.alive = 0u;
    agents[slot] = a;
    slot_push(slot);
    atomicAdd(&life_counters[LC_DEATHS], 1u);
}

fn energy_capacity(a: Agent) -> f32 {
    // v3: cada aminoácido guarda 1. O RNA nu (sem corpo) guarda 1.
    return max(f32(a.body_len), 1.0);
}

@compute @workgroup_size(64)
fn agents_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }
    a.age += 1u;

    // ---- Deriva passiva: levado à velocidade da água (baixo Reynolds). ----
    var p = vec2<f32>(a.pos_x, a.pos_y);
    if (params.fluid_enabled != 0u) {
        let v = fluid_velocity_at_world(p) * (SIM_SIZE / f32(FLUID_SIZE));
        let np = clamp(p + v * params.dt, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
        if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) { p = np; }
        a.vel_x = v.x;
        a.vel_y = v.y;
    }
    a.pos_x = p.x;
    a.pos_y = p.y;
    let cap = energy_capacity(a);

    // ---- COMER = hidrólise catalisada pelos resíduos ----
    // Cada resíduo catalisa a hidrólise de monómeros ativados na sua célula
    // com uma taxa proporcional à sua PROPENSÃO CATALÍTICA medida (M-CSA).
    // O monómero fica no lugar, gasto; a ativação vira energia do agente.
    var cat = AA_CATALYTIC;
    // Fluxo de consumo por direção (difusioforese): soma de taxa × direção
    // do resíduo a partir do centro de massa. Consumo simétrico cancela.
    var phoretic = vec2<f32>(0.0);
    for (var k = 0u; k < a.body_len; k++) {
        if (a.energy + params.food_power > cap) { break; }
        let rw = residue_world(slot, a, k);
        let cell = world_to_cell(rw);
        var avail = vec4<u32>(0u);
        for (var ch = 0u; ch < 4u; ch++) { avail[ch] = chem_act_count(cell, ch); }
        let tot = avail.x + avail.y + avail.z + avail.w;
        if (tot == 0u) { continue; }
        let hunger = clamp(1.0 - a.energy / cap, 0.0, 1.0);
        let pe = clamp(params.uptake_rate * cat[body_get(slot, k)] * f32(tot) * hunger, 0.0, 1.0);
        let rc = rw - p;
        let rl = length(rc);
        if (rl > 1e-4) { phoretic += rc / rl * pe; }
        let r = rng_f4(a.id, params.epoch, S_EAT + k);
        if (r.x < pe) {
            var u = u32(r.y * f32(tot));
            var b = 3u;
            for (var ch = 0u; ch < 4u; ch++) {
                if (u < avail[ch]) { b = ch; break; }
                u -= avail[ch];
            }
            if (chem_spend_one(cell * 4u + b)) { a.energy += params.food_power; }
        }
    }
    a.energy = clamp(a.energy, 0.0, cap) - params.maintenance_cost * f32(a.body_len);

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
    let p_death = clamp(params.death_probability / max(a.energy, 0.01) * thermal + uv_hazard, 0.0, 1.0);
    if (a.energy <= 0.0 || rng_f4(a.id, params.epoch, S_DEATH).x < p_death) {
        die(slot, a);
        return;
    }

    // ---- EMPARELHAMENTO (v3): captura complementos ATIVADOS da vizinhança ----
    // O molde é o genoma; a captura é na célula de um resíduo ao acaso (ou
    // do próprio agente, se for RNA nu). Base a base, pela ordem do genoma.
    if (a.pair_count < a.gene_len && a.energy > 1.0) {
        let rr = rng_f4(a.id, params.epoch, S_PAIR);
        var attempts = u32(params.pairing_rate);
        if (rr.x < fract(params.pairing_rate)) { attempts += 1u; }
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
            if (!chem_take_state_one(world_to_cell(site) * 4u + comp, false)) { break; }
            a.pair_count += 1u;
        }
    }
    agents[slot] = a;
}

// Índice da célula do fluido num ponto do mundo.
fn fluid_index_at_world(p: vec2<f32>) -> u32 {
    let f = clamp(vec2<i32>(floor(p / SIM_SIZE * f32(FLUID_SIZE))), vec2<i32>(0), vec2<i32>(i32(FLUID_SIZE) - 1));
    return u32(f.y) * FLUID_SIZE + u32(f.x);
}

// NASCIMENTO: quando o emparelhamento está completo, os complementos
// capturados formam o filho = complemento reverso do genoma (v3). Mutações
// com a matéria reconciliada AO QUANTUM: a base nova é tirada do meio e a
// velha devolvida (gasta); inserções e duplicações tiram as bases do meio;
// remoções devolvem-nas. Se o meio não tiver a base, a mutação não acontece.
@compute @workgroup_size(64)
fn agents_birth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u || a.gene_len == 0u || a.pair_count < a.gene_len) { return; }

    let L = a.gene_len;
    let cell = chem_open_cell(world_to_cell(vec2<f32>(a.pos_x, a.pos_y)));
    let m = clamp(params.mutation_rate, 0.0, 1.0);
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
    // Remoção de 3 bases (probabilidade 4m), se ficar >= MIN_GENE_LEN.
    if (mr.x < 4.0 * m && n >= MIN_GENE_LEN + 3u) {
        let at = min(u32(mr.y * f32(n - 2u)), n - 3u);
        for (var k = 0u; k < 3u; k++) { chem_add_state(cell, gget(&g, at + k), 1u, true); }
        for (var i = at; i + 3u < n; i++) { gset(&g, i, gget(&g, i + 3u)); }
        n -= 3u;
        for (var i = n; i < n + 3u; i++) { gset(&g, i, 0u); }
    }
    // Inserção de 3 bases ao acaso (probabilidade 4m), tiradas do meio.
    if (mr.z < 4.0 * m && n + 3u <= MAX_GENE_LEN) {
        let q = rng_f4(a.id, params.epoch, S_BIRTH + 1u);
        let at = min(u32(q.x * f32(n + 1u)), n);
        let nb = vec3<u32>(min(u32(q.y * 4.0), 3u), min(u32(q.z * 4.0), 3u), min(u32(q.w * 4.0), 3u));
        var got = 0u;
        for (var k = 0u; k < 3u; k++) {
            if (!take_around(cell, nb[k])) { break; }
            got += 1u;
        }
        if (got == 3u) {
            for (var i = n; i > at; i--) { gset(&g, i + 2u, gget(&g, i - 1u)); }
            for (var k = 0u; k < 3u; k++) { gset(&g, at + k, nb[k]); }
            n += 3u;
        } else {
            for (var k = 0u; k < got; k++) { chem_add_state(cell, nb[k], 1u, true); }
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
    // Posição: 5–15 unidades ao lado, fora da rocha (8 tentativas).
    var cp = vec2<f32>(a.pos_x, a.pos_y);
    for (var t = 0u; t < 8u; t++) {
        let q = rng_f4(a.id, params.epoch, S_BIRTH + 2u + t);
        let ang = q.x * 6.2831853;
        let tp = clamp(cp + vec2<f32>(cos(ang), sin(ang)) * (5.0 + 10.0 * q.y), vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
        if (gamma_count(world_to_cell(tp)) < GAMMA_SOLID_THRESHOLD) { cp = tp; break; }
    }
    // Energia (v3): metade para o filho, metade fica com o pai.
    let half = a.energy * 0.5;
    new_agent(child, cp, mr.w * 6.2831853, half, n, a.generation + 1u);
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
