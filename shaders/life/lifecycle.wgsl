// CICLO DE VIDA: sementes da geração 0, deriva, morte, livro-razão.
// Matéria exata: o genoma (e os complementos capturados) são monómeros reais
// tirados da grelha; na morte voltam todos, como monómeros gastos, contados
// ao quantum (no v3 a devolução era arredondada ao acaso e só conservava em
// média, e as sementes da geração 0 eram matéria criada do nada).

const S_SPAWN: u32 = 5u;
const S_DEATH: u32 = 6u;
// Raio máximo (células) da recolha de bases para uma semente.
const SPAWN_GATHER_RADIUS: i32 = 16;

fn genome_get(slot: u32, i: u32) -> u32 {
    return (genomes[slot * GENOME_WORDS + i / 16u] >> ((i % 16u) * 2u)) & 3u;
}

// ---- Pilha de slots livres ----
// pop e push correm em passes diferentes (nunca no mesmo despacho).
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

// Tira um monómero ATIVADO do canal `ch` da célula mais próxima que o tenha
// (anéis de raio crescente). Devolve a célula, ou 0xFFFFFFFF.
fn take_nearest(cx: i32, cy: i32, ch: u32) -> u32 {
    for (var r = 0; r <= SPAWN_GATHER_RADIUS; r++) {
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                if (abs(dx) != r && abs(dy) != r) { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if (x < 0 || y < 0 || x >= i32(GRID_SIZE) || y >= i32(GRID_SIZE)) { continue; }
                let cell = u32(y) * GRID_SIZE + u32(x);
                if (chem_take_state_one(cell * 4u + ch, false)) { return cell; }
            }
        }
    }
    return 0xFFFFFFFFu;
}

// SEMENTES: cada pedido monta o genoma com os monómeros ATIVADOS mais
// próximos, pela ordem da distância (a sequência espelha a sopa local).
// Dentro de uma célula, a base é sorteada em proporção do que lá há.
// Se não houver bases suficientes ou slots livres, devolve tudo.
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
        let aug = vec3<u32>(0u, 1u, 2u);
        for (var k = 0u; k < 3u; k++) {
            let b = aug[k];
            if (take_nearest(cx, cy, b) == 0xFFFFFFFFu) { ok = false; break; }
            g[n / 16u] |= b << ((n % 16u) * 2u);
            taken[b] += 1u;
            n += 1u;
        }
    }

    // Recolha por anéis de distância crescente.
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
                        g[n / 16u] |= b << ((n % 16u) * 2u);
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
        // Falhou: devolve exatamente o que tirou (ativado, como estava).
        let back = chem_open_cell(c0);
        for (var ch = 0u; ch < 4u; ch++) { chem_add_state(back, ch, taken[ch], false); }
        atomicAdd(&life_counters[LC_SPAWN_FAILED], 1u);
        return;
    }
    for (var w = 0u; w < GENOME_WORDS; w++) { genomes[slot * GENOME_WORDS + w] = g[w]; }
    var a: Agent;
    a.pos_x = req.pos_x;
    a.pos_y = req.pos_y;
    a.vel_x = 0.0;
    a.vel_y = 0.0;
    a.rot = rng_f4(ri, params.epoch, S_SPAWN).y * 6.2831853;
    a.energy = params.spawn_energy;
    a.alive = 1u;
    a.gene_len = n;
    a.pair_count = 0u;
    a.body_len = 0u;
    a.generation = 0u;
    a.age = 0u;
    a.id = atomicAdd(&life_counters[LC_NEXT_ID], 1u);
    agents[slot] = a;
    atomicAdd(&life_counters[LC_SPAWNED], 1u);
}

// Matéria presa num agente, por canal: o genoma + os complementos já
// capturados (complemento de Watson-Crick das primeiras pair_count bases).
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

// PASSO DO AGENTE (fase 3a): deriva com a água e morte.
@compute @workgroup_size(64)
fn agents_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }
    a.age += 1u;

    // Deriva passiva: o corpo é levado à velocidade da água (baixo
    // Reynolds, sem inércia). Não entra em rocha sólida.
    var p = vec2<f32>(a.pos_x, a.pos_y);
    if (params.fluid_enabled != 0u) {
        let v = fluid_velocity_at_world(p) * (SIM_SIZE / f32(FLUID_SIZE));
        let np = clamp(p + v * params.dt, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
        if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) {
            p = np;
        }
        a.vel_x = v.x;
        a.vel_y = v.y;
    }
    a.pos_x = p.x;
    a.pos_y = p.y;

    // Morte (v3): base ÷ energia. (Térmica e UV entram com o metabolismo.)
    let p_death = clamp(params.death_probability / max(a.energy, 0.01), 0.0, 1.0);
    if (a.energy <= 0.0 || rng_f4(a.id, params.epoch, S_DEATH).x < p_death) {
        // Devolve TODA a matéria, como monómeros gastos, na célula de água
        // livre mais próxima. A energia evapora.
        let m = agent_matter(slot);
        let cell = chem_open_cell(world_to_cell(p));
        for (var ch = 0u; ch < 4u; ch++) { chem_add_state(cell, ch, m[ch], true); }
        a.alive = 0u;
        agents[slot] = a;
        slot_push(slot);
        atomicAdd(&life_counters[LC_DEATHS], 1u);
        return;
    }
    agents[slot] = a;
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
