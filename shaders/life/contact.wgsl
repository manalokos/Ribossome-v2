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
    atomicStore(&matter_claim[slot], BOND_NONE);
}

// PROTEASE (predação química): um órgão. Cada variante tem uma FAMÍLIA, que
// decide o que corta na vítima:
//   família 1 (tipo tripsina):            lisina, arginina
//   família 2 (tipo caspase / legumaína): aspartato, asparagina
//   família 3 (tipo pepsina):             fenilalanina, tirosina, triptofano, leucina
// e pode estar sempre ativa ou só com um sinal interno (γ ou δ) positivo: a
// criatura pode evoluir "abrir a boca" só quando interessa.
// (as classes vêm da tabela dos aminoácidos). Em cada passo de contacto, as
// proteases ativas de um atacante TIRAM à vítima uma quantidade certa de
// energia: PRED_DRAIN × força das proteases × fração de resíduos da vítima
// que a família corta × (1 − defesa da prolina) × params.protease_power. É
// determinista e acumula: contactos curtos repetidos somam. Quando a energia
// chega a zero a vítima morre (os complementos que tinha capturado saem
// ativados, como em qualquer morte).
// A regra é cega (nenhum genoma é comparado): quem não tem os
// aminoácidos-alvo é imune, e os parentes, com a mesma composição, poupam-se
// uns aos outros por isso. Não há autodigestão: uma protease só corta OUTROS.
// Para onde vai a energia tirada: params.protease_direct vai direta para o
// atacante; do resto, params.lysis_yield reativa monómeros gastos na célula
// da vítima (um por cada food_power), que qualquer boca pode comer.
const PRED_DRAIN: f32 = 0.2;
// Força 1 contra uma vítima com 10% de resíduos-alvo = PRED_DRAIN por passo.
const PRED_SITE_SCALE: f32 = 10.0;
const PRED_PROLINE_DEFENSE: f32 = 0.9;
const AA_PROLINE: u32 = 12u;
const BITE_SCALE: f32 = 10000.0;
const S_DIGEST: u32 = 13u;

// Força das proteases ATIVAS do agente, por família.
fn protease_sites(slot: u32, n: u32) -> vec3<f32> {
    var s = vec3<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_PROTEASE) {
            let v = organ_var(o);
            var drive = 1.0;
            if (v.p2 >= 0.0) { drive = clamp(signals[slot * MAX_BODY + k][u32(clamp(v.p2, 0.0, 3.0))], 0.0, 1.0); }
            s[u32(clamp(v.p0, 1.0, 3.0)) - 1u] += max(v.p1, 0.0) * organ_gain(o) * drive;
        }
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
    // Energia que este agente tira a outros neste passo.
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
                    if (overlap > 0.0 && armed && b.energy > 0.0) {
                        let def = unpack_defence(contact_disp[e].w);
                        let power = dot(sites, def.xyz) * PRED_SITE_SCALE;
                        let resist = 1.0 - PRED_PROLINE_DEFENSE * def.w;
                        let bite = min(PRED_DRAIN * max(params.protease_power, 0.0) * power * resist, b.energy);
                        if (bite > 0.0) {
                            atomicAdd(&bitten[e], u32(bite * BITE_SCALE));
                            gained += bite;
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
    // PARTILHA DE MATÉRIA (decidida em bond_maintain): quem recebeu fica
    // com mais um complemento; quem foi escolhido como dador fica sem o
    // último. A base é a mesma nos dois, por isso a matéria conserva-se
    // canal a canal.
    let mc = atomicLoad(&matter_claim[slot]);
    if (mc == MATTER_RECEIVED) {
        a.pair_count += 1u;
    } else if (mc != BOND_NONE && mc != MATTER_LOCK && a.pair_count > 0u) {
        a.pair_count -= 1u;
    }
    // Fica limpo já aqui: um slot que morra e renasça a meio de um passo
    // não pode herdar a reserva do morto.
    atomicStore(&matter_claim[slot], BOND_NONE);
    // O que este agente tirou a outros: a parte direta entra-lhe na energia.
    let direct = clamp(params.protease_direct, 0.0, 1.0);
    a.energy = min(max(a.energy + contact_disp[slot].z * direct + bd.w, 0.0), energy_capacity(slot, a));
    // O que lhe tiraram: sai da energia; a parte que não foi direta para os
    // atacantes reativa gastos da célula onde está (× lysis_yield, um por
    // cada food_power, arredondamento ao acaso). Sem gastos ali, perde-se.
    let lost = min(f32(atomicExchange(&bitten[slot], 0u)) / BITE_SCALE, max(a.energy, 0.0));
    if (lost > 0.0) {
        let q = rng_f4(a.id, params.epoch, S_DIGEST);
        let want = lost * (1.0 - direct) * clamp(params.lysis_yield, 0.0, 1.0) / max(params.food_power, 1e-3);
        var count = u32(want);
        if (q.x < fract(want)) { count += 1u; }
        let cell = world_to_cell(vec2<f32>(a.pos_x, a.pos_y));
        let ch0 = min(u32(q.y * 4.0), 3u);
        for (var i = 0u; i < min(count, 8u); i++) {
            for (var t = 0u; t < 4u; t++) {
                if (chem_activate_one(cell * 4u + (ch0 + i + t) % 4u)) { break; }
            }
        }
        a.energy -= lost;
        // Sem energia: morre aqui, e conta como morte por protease.
        if (a.energy <= 0.0) {
            atomicAdd(&life_counters[LC_BITES], 1u);
            die(slot, a);
            contact_disp[slot] = vec4<f32>(0.0);
            return;
        }
    }
    agents[slot] = a;
    // Para a vista (flash) e para o sinal das proteases: .x = energia que
    // lhe tiraram neste passo, .z = a que tirou a outros. contact_build repõe tudo no
    // passo seguinte.
    contact_disp[slot] = vec4<f32>(lost, 0.0, contact_disp[slot].z, contact_disp[slot].w);
}
