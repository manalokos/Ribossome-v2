// LIGAÇÕES ENTRE AGENTES (pontes salinas): um resíduo com carga + de um
// agente que toca num resíduo com carga − de outro pode ligar-se a ele
// (carga = propriedade do aminoácido: K, R +; D, E −). Nenhuma regra olha
// para o genoma: a sequência só decide onde ficam as cargas. Um filho nasce
// sobreposto ao pai; se as cargas forem complementares, ficam colados e
// formam colónias.
//
// A ligação é uma mola sobreamortecida entre os dois resíduos (sem
// inércia), não leva matéria, e por ela passam energia (por gradiente) e os
// sinais α/β (o resíduo do outro conta como mais um vizinho da cadeia).
//
// Cada agente guarda as suas ligações (até MAX_BONDS) e o outro lado guarda
// a mesma, ao contrário. Para os dois lados concordarem sempre:
//   bond_maintain: valida, calcula forças/energia e decide quebras com um
//                  sorteio SIMÉTRICO pelo par (os dois lados tiram o mesmo);
//   bond_propose:  cada agente com 2+ lugares livres propõe-se a UM vizinho
//                  (atomicMin: o vizinho escolhe o proponente de slot menor);
//   bond_accept:   o escolhido regista a ligação e o proponente também, se
//                  foi ele o escolhido. As forças entram em contact_apply.

const MAX_BONDS: u32 = 4u;
// Por slot: MAX_BONDS ligações + a proposta deste passo.
const BOND_STRIDE: u32 = MAX_BONDS + 1u;
const BOND_NONE: u32 = 0xFFFFFFFFu;
// Distância máxima entre os dois resíduos para se ligarem (mundo).
const BOND_RANGE: f32 = 12.0;
// Comprimento de repouso da mola.
const BOND_LEN: f32 = 8.0;
// Esticada além disto, parte-se.
const BOND_BREAK_LEN: f32 = 40.0;
// Fração do desvio corrigida por passo (cada lado faz metade) e limite.
const BOND_RELAX: f32 = 0.25;
const BOND_MAX_STEP: f32 = 4.0;
const BOND_MAX_TURN: f32 = 0.1;
// Carga mínima (em módulo) para um resíduo poder ligar.
const BOND_MIN_CHARGE: f32 = 0.5;
const S_BOND: u32 = 8u << 16u;      // + 16 bits do id maior do par
const S_BOND_PROP: u32 = 11u;

// Ligação: x = slot do outro (BOND_NONE = livre), y = id do outro,
// z = o meu resíduo | (o resíduo do outro << 16), w = tipo: 0 = ponte
// salina, 1 + n = ligação de nascimento com n G/C nas pontas do genoma (na
// proposta, w = lugares livres depois da manutenção).
fn bond_at(slot: u32, i: u32) -> vec4<u32> {
    return bonds[slot * BOND_STRIDE + i];
}

fn residue_charge(slot: u32, k: u32) -> f32 {
    return aa_props[body_get(slot, k)].charge;
}

// A ligação i de `slot` ainda é válida (o outro vive e é o mesmo agente)?
fn bond_partner_ok(b: vec4<u32>) -> bool {
    if (b.x == BOND_NONE || b.x >= params.max_agents) { return false; }
    let o = agents[b.x];
    return o.alive != 0u && o.id == b.y && (b.z >> 16u) < o.body_len;
}


// Marca as ligações de um slot novo como livres.
fn bonds_clear(slot: u32) {
    for (var i = 0u; i < BOND_STRIDE; i++) {
        bonds[slot * BOND_STRIDE + i] = vec4<u32>(BOND_NONE, 0u, 0u, 0u);
    }
}

@compute @workgroup_size(64)
fn bond_maintain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    atomicStore(&bond_accept[slot], BOND_NONE);
    let a = agents[slot];
    if (a.alive == 0u) {
        bond_disp[slot] = vec4<f32>(0.0);
        return;
    }
    let pa = vec2<f32>(a.pos_x, a.pos_y);
    var dp = vec2<f32>(0.0);
    var drot = 0.0;
    var de = 0.0;
    var free = 0u;
    for (var i = 0u; i < MAX_BONDS; i++) {
        let b = bond_at(slot, i);
        if (b.x == BOND_NONE) {
            free += 1u;
            continue;
        }
        var keep = bond_partner_ok(b) && (b.z & 0xFFFFu) < a.body_len;
        if (keep) {
            let o = agents[b.x];
            let ra = residue_world(slot, a, b.z & 0xFFFFu);
            let rb = residue_world(b.x, o, b.z >> 16u);
            let d = rb - ra;
            let dist = length(d);
            // Sorteio simétrico: os dois lados usam o mesmo par (id menor,
            // id maior) e a mesma distância, por isso decidem o mesmo.
            let lo = min(a.id, o.id);
            let hi = max(a.id, o.id);
            let q = rng_f4(lo, params.epoch, S_BOND + (hi & 0xFFFFu));
            // Ponte salina: quebra uniforme. Nascimento: o duplex das pontas
            // segura mais com mais G/C (3 pontes de hidrogénio contra 2).
            var p_break = params.bond_break;
            if (b.w >= 1u) { p_break = params.birth_bond_break * exp2(-0.5 * f32(b.w - 1u)); }
            if (dist > BOND_BREAK_LEN || q.x < p_break) {
                keep = false;
            } else {
                // Mola: cada lado corrige metade do desvio.
                if (dist > 1e-4) {
                    let f = d / dist * (dist - BOND_LEN) * BOND_RELAX * 0.5;
                    dp += f;
                    let r = ra - pa;
                    drot += (r.x * f.y - r.y * f.x) / (dot(r, r) + a.radius * a.radius + 1.0);
                }
                // Energia por gradiente (os dois lados veem as mesmas energias).
                de += params.bond_energy_share * 0.5 * (o.energy - a.energy);
            }
        }
        if (!keep) {
            bonds[slot * BOND_STRIDE + i] = vec4<u32>(BOND_NONE, 0u, 0u, 0u);
            free += 1u;
        }
    }
    let l = length(dp);
    if (l > BOND_MAX_STEP) { dp *= BOND_MAX_STEP / l; }
    bond_disp[slot] = vec4<f32>(dp, clamp(drot, -BOND_MAX_TURN, BOND_MAX_TURN), de);
    bonds[slot * BOND_STRIDE + MAX_BONDS] = vec4<u32>(BOND_NONE, 0u, 0u, free);
}

fn bonded_to(slot: u32, other: u32) -> bool {
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(slot, i).x == other) { return true; }
    }
    return false;
}

@compute @workgroup_size(64)
fn bond_propose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u || a.body_len == 0u || params.bond_rate <= 0.0) { return; }
    let prop_i = slot * BOND_STRIDE + MAX_BONDS;
    let free = bonds[prop_i].w;
    if (free < 2u) { return; }
    let q = rng_f4(a.id, params.epoch, S_BOND_PROP);
    if (q.x >= params.bond_rate) { return; }
    // Um resíduo ao acaso: só os carregados ligam.
    let k = min(u32(q.y * f32(a.body_len)), a.body_len - 1u);
    let ca = residue_charge(slot, k);
    if (abs(ca) < BOND_MIN_CHARGE) { return; }
    let ra = residue_world(slot, a, k);
    let c = contact_cell_xy(ra);
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
                    let far = length(vec2<f32>(b.pos_x, b.pos_y) - ra) > b.radius + BOND_RANGE + 20.0;
                    if (!far && b.body_len > 0u && !bonded_to(slot, e)) {
                        for (var j = 0u; j < b.body_len; j++) {
                            if (ca * residue_charge(e, j) > -BOND_MIN_CHARGE * BOND_MIN_CHARGE) { continue; }
                            if (length(residue_world(e, b, j) - ra) > BOND_RANGE) { continue; }
                            bonds[prop_i] = vec4<u32>(e, b.id, k | (j << 16u), free);
                            atomicMin(&bond_accept[e], slot);
                            return;
                        }
                    }
                }
                e = contact_next[e];
            }
        }
    }
}

// Bases das pontas do genoma (cada ponta) que seguram a cópia ao pai.
const BIRTH_BOND_ENDS: u32 = 6u;

// LIGAÇÃO DE NASCIMENTO (chamada em agents_birth): a cópia foi feita por
// emparelhamento com o genoma do pai e fica presa pelas pontas hibridadas
// até se separar. A última posição do corpo do pai liga à primeira do filho
// (cadeias cabeça-cauda: filamentos). Sem lugar livre no pai, separam-se.
fn birth_bond(parent: u32, pa: Agent, child: u32) {
    if (params.birth_bond_break >= 1.0 || pa.body_len == 0u) { return; }
    let ca = agents[child];
    if (ca.body_len == 0u) { return; }
    let L = pa.gene_len;
    var gc = 0u;
    for (var i = 0u; i < min(BIRTH_BOND_ENDS, L); i++) {
        gc += select(0u, 1u, genome_get(parent, i) >= 2u) + select(0u, 1u, genome_get(parent, L - 1u - i) >= 2u);
    }
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(parent, i).x == BOND_NONE) {
            let kp = pa.body_len - 1u;
            bonds[parent * BOND_STRIDE + i] = vec4<u32>(child, ca.id, kp, 1u + gc);
            bonds[child * BOND_STRIDE] = vec4<u32>(parent, pa.id, kp << 16u, 1u + gc);
            return;
        }
    }
}

// Regista a ligação no primeiro lugar livre.
fn bond_add(slot: u32, partner: u32, partner_id: u32, mine: u32, theirs: u32) {
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(slot, i).x == BOND_NONE) {
            bonds[slot * BOND_STRIDE + i] = vec4<u32>(partner, partner_id, mine | (theirs << 16u), 0u);
            return;
        }
    }
}

@compute @workgroup_size(64)
fn bond_accept_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    let mine = bonds[slot * BOND_STRIDE + MAX_BONDS];
    // Como escolhido: aceita o proponente de slot menor (se houver lugar).
    let w = atomicLoad(&bond_accept[slot]);
    if (w != BOND_NONE && mine.w >= 1u) {
        let p = bonds[w * BOND_STRIDE + MAX_BONDS];
        if (p.x == slot) {
            bond_add(slot, w, agents[w].id, p.z >> 16u, p.z & 0xFFFFu);
        }
    }
    // Como proponente: fica ligado se o outro me escolheu e tinha lugar.
    if (mine.x != BOND_NONE && mine.x < params.max_agents) {
        let other = bonds[mine.x * BOND_STRIDE + MAX_BONDS];
        if (atomicLoad(&bond_accept[mine.x]) == slot && other.w >= 1u) {
            bond_add(slot, mine.x, mine.y, mine.z & 0xFFFFu, mine.z >> 16u);
        }
    }
}
