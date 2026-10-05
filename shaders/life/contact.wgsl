// CONTACTO ENTRE AGENTES: repulsão simples entre centros de massa (fase 3).
//
// Cada agente é um disco centrado no centro de massa, com o RAIO DE GIRAÇÃO
// do corpo (distância quadrática média dos resíduos ao centro, mais a
// espessura de um resíduo), limitado a CONTACT_R_MAX. O RNA nu tem
// CONTACT_R_NAKED. Uma grelha grossa (células de 2·CONTACT_R_MAX) guarda,
// por célula, uma lista ligada de agentes; cada agente soma os empurrões
// de sobreposição dos vizinhos e desloca-se sobreamortecido (baixo
// Reynolds: sem inércia nem rotação). Os deslocamentos ficam em
// contact_disp e só são aplicados num passe seguinte.

const CONTACT_R_NAKED: f32 = 4.0;
const CONTACT_R_MAX: f32 = 60.0;
const CONTACT_CELL: f32 = 2.0 * CONTACT_R_MAX;
const CONTACT_N: u32 = u32(SIM_SIZE / CONTACT_CELL) + 1u;
const NO_ENTRY: u32 = 0xFFFFFFFFu;
// Fração da sobreposição corrigida por passo (cada lado faz metade).
const CONTACT_RELAX: f32 = 0.5;
const CONTACT_MAX_STEP: f32 = 4.0;

fn contact_cell_xy(p: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(floor(p / CONTACT_CELL)), vec2<i32>(0), vec2<i32>(i32(CONTACT_N) - 1));
}

@compute @workgroup_size(256)
fn contact_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= CONTACT_N * CONTACT_N) { return; }
    atomicStore(&contact_head[gid.x], NO_ENTRY);
}

@compute @workgroup_size(64)
fn contact_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents || agents[slot].alive == 0u) { return; }
    let c = contact_cell_xy(vec2<f32>(agents[slot].pos_x, agents[slot].pos_y));
    contact_next[slot] = atomicExchange(&contact_head[u32(c.y) * CONTACT_N + u32(c.x)], slot);
    // O que os atacantes precisam de saber desta vítima (frações de
    // resíduos-alvo de cada família e de prolina), calculado uma vez por
    // passo e empacotado em contact_disp.w (4 × 6 bits, exato num f32).
    contact_disp[slot] = vec4<f32>(0.0, 0.0, 0.0, pack_defence(slot, agents[slot].body_len));
}

// PROTEASES POR CONTACTO (predação química). Não há órgão: o sítio ativo de
// uma protease forma-se quando a dobragem do corpo encosta dois resíduos NÃO
// vizinhos na cadeia, como nas proteases reais:
//   família 1, de serina:    serina + histidina    (tripsina)  corta K, R
//   família 2, de cisteína:  cisteína + histidina  (caspases, legumaína) corta D, N
//   família 3, aspártica:    aspartato + aspartato (pepsina)   corta F, Y, W, L
// (as classes vêm da tabela dos aminoácidos). LISE: uma vítima tocada por
// sítios ativos tem, em cada passo, um RISCO de se desfazer por inteiro
// (como uma célula que rebenta quando a parede cede: tudo ou nada). O risco
// soma, por atacante, PRED_HAZARD × sítios × fração de resíduos da vítima
// que a família corta × (1 − defesa da prolina) × params.protease_power.
// A regra é cega (nenhum genoma é comparado): quem não tem os
// aminoácidos-alvo é imune, e os parentes, com a mesma composição, poupam-se
// uns aos outros por isso. Não há autodigestão: um sítio só corta OUTROS.
// A vítima morre COM a energia que tinha: uma fração (params.lysis_yield)
// fica nos restos como ativação, um monómero por cada food_power de energia,
// primeiro os do seu próprio genoma, depois gastos à volta (die_release). O
// atacante não recebe nada diretamente: tem de comer os restos.
const PRED_HAZARD: f32 = 0.01;
// Um sítio contra uma vítima com 10% de resíduos-alvo = PRED_HAZARD por passo.
const PRED_SITE_SCALE: f32 = 10.0;
const PRED_PROLINE_DEFENSE: f32 = 0.9;
const AA_PROLINE: u32 = 12u;
const BITE_SCALE: f32 = 100000.0;
const S_DIGEST: u32 = 13u;

// Família (1..3) do sítio ativo formado no resíduo k, ou 0. `once`: num par
// de dois nucleófilos da mesma família (aspartato + aspartato) só conta o
// de índice menor.
fn protease_site(slot: u32, n: u32, k: u32, once: bool) -> u32 {
    let f = u32(max(aa_props[body_get(slot, k)].protease_site, 0.0) + 0.5);
    // (Sem `return` nem `continue` dentro do ciclo: com eles, chamada a
    // partir de agents_step, a GPU ficava presa e o dispositivo perdia-se.)
    var found = 0u;
    if (f != 0u) {
        let bit = 1u << (f - 1u);
        let base = slot * MAX_BODY;
        let pk = body_pos[base + k];
        for (var j = 0u; j < n; j++) {
            let far = j + 2u < k || j > k + 2u;
            let aj = body_get(slot, j);
            let partner = (u32(max(aa_props[aj].protease_partner, 0.0) + 0.5) & bit) != 0u;
            let twin = once && j < k && u32(max(aa_props[aj].protease_site, 0.0) + 0.5) == f;
            let dj = body_pos[base + j] - pk;
            if (far && partner && !twin && dot(dj, dj) < CONTACT_EMIT_RADIUS * CONTACT_EMIT_RADIUS) {
                found = f;
                break;
            }
        }
    }
    return found;
}

// Sítios ativos formados no corpo, por família.
fn protease_sites(slot: u32, n: u32) -> vec3<f32> {
    var s = vec3<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let f = protease_site(slot, n, k, true);
        if (f > 0u) { s[f - 1u] += 1.0; }
    }
    return s;
}

// Fração dos resíduos do corpo que cada família corta.
fn protease_targets(slot: u32, n: u32) -> vec3<f32> {
    var t = vec3<f32>(0.0);
    if (n == 0u) { return t; }
    for (var k = 0u; k < n; k++) {
        let m = u32(max(aa_props[body_get(slot, k)].protease_target, 0.0) + 0.5);
        if ((m & 1u) != 0u) { t.x += 1.0; }
        if ((m & 2u) != 0u) { t.y += 1.0; }
        if ((m & 4u) != 0u) { t.z += 1.0; }
    }
    return t / f32(n);
}

// (alvo fam. 1, alvo fam. 2, alvo fam. 3, prolina), cada um 0..1 em 6 bits.
fn pack_defence(slot: u32, n: u32) -> f32 {
    let t = protease_targets(slot, n);
    let q = vec4<u32>(round(clamp(vec4<f32>(t, proline_fraction(slot, n)), vec4<f32>(0.0), vec4<f32>(1.0)) * 63.0));
    return f32(q.x | (q.y << 6u) | (q.z << 12u) | (q.w << 18u));
}

fn unpack_defence(w: f32) -> vec4<f32> {
    let u = u32(max(w, 0.0));
    return vec4<f32>(f32(u & 63u), f32((u >> 6u) & 63u), f32((u >> 12u) & 63u), f32((u >> 18u) & 63u)) / 63.0;
}

fn proline_fraction(slot: u32, n: u32) -> f32 {
    if (n == 0u) { return 0.0; }
    var c = 0u;
    for (var k = 0u; k < n; k++) {
        if (body_get(slot, k) == AA_PROLINE) { c += 1u; }
    }
    return f32(c) / f32(n);
}

@compute @workgroup_size(64)
fn contact_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    let p = vec2<f32>(a.pos_x, a.pos_y);
    let c = contact_cell_xy(p);
    var push = vec2<f32>(0.0);
    let sites = protease_sites(slot, a.body_len);
    let armed = sites.x + sites.y + sites.z > 0.0;
    // Risco de lise que este agente está a causar a outros neste passo.
    var gained = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let x = c.x + dx;
            let y = c.y + dy;
            if (x < 0 || y < 0 || x >= i32(CONTACT_N) || y >= i32(CONTACT_N)) { continue; }
            var e = atomicLoad(&contact_head[u32(y) * CONTACT_N + u32(x)]);
            var guard = 0u;
            loop {
                if (e == NO_ENTRY || guard >= 1024u) { break; }
                guard += 1u;
                if (e != slot) {
                    let b = agents[e];
                    let d = p - vec2<f32>(b.pos_x, b.pos_y);
                    let dist = length(d);
                    let overlap = a.radius + b.radius - dist;
                    if (overlap > 0.0) {
                        // Coincidentes: direção determinista pelo par.
                        var dir = select(vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), slot < e);
                        if (dist > 1e-4) { dir = d / dist; }
                        push += dir * overlap;
                    }
                    // Ataque: só em contacto.
                    if (overlap > 0.0 && armed) {
                        let def = unpack_defence(contact_disp[e].w);
                        let power = dot(sites, def.xyz) * PRED_SITE_SCALE;
                        let resist = 1.0 - PRED_PROLINE_DEFENSE * def.w;
                        let hazard = min(PRED_HAZARD * max(params.protease_power, 0.0) * power * resist, 1.0);
                        if (hazard > 0.0) {
                            atomicAdd(&bitten[e], u32(hazard * BITE_SCALE));
                            gained += hazard;
                        }
                    }
                }
                e = contact_next[e];
            }
        }
    }
    var dp = push * CONTACT_RELAX;
    let dl = length(dp);
    if (dl > CONTACT_MAX_STEP) { dp *= CONTACT_MAX_STEP / dl; }
    // .w (defesa) fica igual: outros atacantes ainda a podem estar a ler.
    contact_disp[slot] = vec4<f32>(dp, gained, contact_disp[slot].w);
}

// Aplica os deslocamentos (não entra em rocha sólida).
@compute @workgroup_size(64)
fn contact_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }
    // Contacto + ligações (bond_disp: dx, dy, dθ, energia trocada).
    let bd = bond_disp[slot];
    let np = clamp(vec2<f32>(a.pos_x, a.pos_y) + contact_disp[slot].xy + bd.xy, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) {
        a.pos_x = np.x;
        a.pos_y = np.y;
    }
    a.rot += bd.z;
    a.energy = min(max(a.energy + bd.w, 0.0), energy_capacity(slot, a));
    // LISE: o risco somado de todos os atacantes; um só sorteio por vítima.
    let hazard = f32(atomicExchange(&bitten[slot], 0u)) / BITE_SCALE;
    let ql = rng_f4(a.id, params.epoch, S_DIGEST);
    if (hazard > 0.0 && ql.x < hazard) {
        // Monómeros que saem ativados (arredondamento ao acaso da fração).
        let want = max(a.energy, 0.0) * clamp(params.lysis_yield, 0.0, 1.0) / max(params.food_power, 1e-3);
        var budget = u32(want);
        if (ql.y < fract(want)) { budget += 1u; }
        atomicAdd(&life_counters[LC_BITES], 1u);
        die_release(slot, a, min(budget, 512u));
        contact_disp[slot] = vec4<f32>(0.0);
        return;
    }
    agents[slot] = a;
    // Para a vista (flash) e para o sinal das proteases: .x = risco de lise
    // que sofreu neste passo, .z = o que causou a outros. contact_build repõe tudo no
    // passo seguinte.
    contact_disp[slot] = vec4<f32>(hazard, 0.0, contact_disp[slot].z, contact_disp[slot].w);
}
