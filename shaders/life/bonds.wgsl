// LIGAÇÕES ENTRE AGENTES POR ÂNCORAS: o órgão "âncora" (promotor tirosina,
// como as colas DOPA dos mexilhões) tem polaridade + ou − e uma força (a
// variante). Uma âncora livre que toca numa âncora livre de polaridade
// oposta de outro agente liga-se a ela; cada âncora segura UMA ligação.
// A força da variante decide quanto dura (as permanentes só se soltam se a
// ligação esticar demais ou o outro morrer). Nenhuma regra olha para o
// genoma: só para os órgãos que o corpo tem.
//
// Nascimento: se o pai tiver uma âncora livre e o filho uma de polaridade
// oposta, nascem ligados (colónias, filamentos).
//
// A ligação é uma mola sobreamortecida entre os dois resíduos, com binário,
// não leva matéria, e por ela passam energia (por gradiente) e os sinais
// α/β (o resíduo do outro conta como mais um vizinho da cadeia).
//
// Cada agente guarda as suas ligações (até MAX_BONDS) e o outro lado a
// mesma, ao contrário. Para os dois lados concordarem sempre:
//   bond_maintain: valida, calcula forças/energia e decide quebras com um
//                  sorteio SIMÉTRICO pelo par (os dois lados tiram o mesmo);
//   bond_propose:  cada agente com um lugar livre propõe UMA ligação a um
//                  vizinho (atomicMin: o vizinho escolhe o de slot menor);
//   bond_accept:   quem não propôs aceita o escolhido; o proponente sabe
//                  que foi aceite pelas mesmas regras. Forças em contact_apply.

const MAX_BONDS: u32 = 4u;
// Por slot: MAX_BONDS ligações + a proposta deste passo.
const BOND_STRIDE: u32 = MAX_BONDS + 1u;
const BOND_NONE: u32 = 0xFFFFFFFFu;
// Distância máxima entre as duas âncoras para se ligarem (mundo).
const BOND_RANGE: f32 = 16.0;
// Comprimento de repouso da mola.
const BOND_LEN: f32 = 6.0;
// Esticada além disto, parte-se mesmo sendo permanente.
const BOND_BREAK_LEN: f32 = 80.0;
// Fração do desvio corrigida por passo (cada lado faz metade) e limites:
// forte, quase como a ligação da própria cadeia.
const BOND_RELAX: f32 = 0.6;
const BOND_MAX_STEP: f32 = 6.0;
const BOND_MAX_TURN: f32 = 0.15;
const S_BOND: u32 = 8u << 16u;      // + 16 bits do id maior do par
const S_BOND_PROP: u32 = 11u;
// Tipo da ligação (bit 16 de z): 0 = por contacto, 1 = de nascimento.
const BOND_KIND_BIRTH: u32 = 1u << 16u;

// Ligação: x = slot do outro (BOND_NONE = livre), y = id do outro,
// z = o meu resíduo | (o resíduo do outro << 8) | (tipo << 16),
// w = probabilidade de quebra por passo (bits de f32). Na proposta (índice
// MAX_BONDS): w = lugares livres depois da manutenção.
fn bond_at(slot: u32, i: u32) -> vec4<u32> {
    return bonds[slot * BOND_STRIDE + i];
}

fn bond_mine(b: vec4<u32>) -> u32 {
    return b.z & 0xFFu;
}

fn bond_theirs(b: vec4<u32>) -> u32 {
    return (b.z >> 8u) & 0xFFu;
}

// Polaridade da âncora no resíduo k (+1, −1) ou 0 se não for âncora.
fn anchor_polarity(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    if (organ_type(o) != ORGAN_ANCHOR) { return 0.0; }
    return select(-1.0, 1.0, organ_var(o).p0 >= 0.0);
}

// Probabilidade de quebra por passo da âncora no resíduo k.
fn anchor_break(slot: u32, k: u32) -> f32 {
    return max(organ_var(organ_get(slot, k)).p1, 0.0);
}

// A ligação ainda é válida (o outro vive e é o mesmo agente)?
fn bond_partner_ok(b: vec4<u32>) -> bool {
    if (b.x == BOND_NONE || b.x >= params.max_agents) { return false; }
    let o = agents[b.x];
    return o.alive != 0u && o.id == b.y && bond_theirs(b) < o.body_len;
}

// A âncora k deste agente já segura uma ligação?
fn anchor_busy(slot: u32, k: u32) -> bool {
    for (var i = 0u; i < MAX_BONDS; i++) {
        let b = bond_at(slot, i);
        if (b.x != BOND_NONE && bond_mine(b) == k) { return true; }
    }
    return false;
}

// Marca as ligações de um slot novo como livres.
fn bonds_clear(slot: u32) {
    for (var i = 0u; i < BOND_STRIDE; i++) {
        bonds[slot * BOND_STRIDE + i] = vec4<u32>(BOND_NONE, 0u, 0u, 0u);
    }
}

fn bond_write(slot: u32, i: u32, partner: u32, partner_id: u32, mine: u32, theirs: u32, kind: u32, p_break: f32) {
    bonds[slot * BOND_STRIDE + i] = vec4<u32>(partner, partner_id, mine | (theirs << 8u) | kind, bitcast<u32>(p_break));
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
        var keep = bond_partner_ok(b) && bond_mine(b) < a.body_len;
        if (keep) {
            let o = agents[b.x];
            let ra = residue_world(slot, a, bond_mine(b));
            let rb = residue_world(b.x, o, bond_theirs(b));
            let d = rb - ra;
            let dist = length(d);
            // Sorteio simétrico: os dois lados usam o mesmo par (id menor,
            // id maior), a mesma distância e a mesma probabilidade.
            let lo = min(a.id, o.id);
            let hi = max(a.id, o.id);
            let q = rng_f4(lo, params.epoch, S_BOND + (hi & 0xFFFFu));
            if (dist > BOND_BREAK_LEN || q.x < bitcast<f32>(b.w)) {
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
    if (free < 1u) { return; }
    let q = rng_f4(a.id, params.epoch, S_BOND_PROP);
    if (q.x >= params.bond_rate) { return; }
    // Uma âncora livre, a começar num resíduo ao acaso.
    let n = a.body_len;
    let k0 = min(u32(q.y * f32(n)), n - 1u);
    var k = BOND_NONE;
    for (var t = 0u; t < n; t++) {
        let kk = (k0 + t) % n;
        if (anchor_polarity(slot, kk) != 0.0 && !anchor_busy(slot, kk)) {
            k = kk;
            break;
        }
    }
    if (k == BOND_NONE) { return; }
    let pol = anchor_polarity(slot, k);
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
                            if (anchor_polarity(e, j) != -pol || anchor_busy(e, j)) { continue; }
                            if (length(residue_world(e, b, j) - ra) > BOND_RANGE) { continue; }
                            bonds[prop_i] = vec4<u32>(e, b.id, k | (j << 8u), free);
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

fn bond_add(slot: u32, partner: u32, partner_id: u32, mine: u32, theirs: u32, p_break: f32) {
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(slot, i).x == BOND_NONE) {
            bond_write(slot, i, partner, partner_id, mine, theirs, 0u, p_break);
            return;
        }
    }
}

// Um agente aceita o proponente escolhido se não propôs ele próprio neste
// passo (a sua âncora podia ser a mesma) e tem lugar. Os dois lados avaliam
// exatamente esta condição.
fn accepts(target_slot: u32, proposer: u32) -> bool {
    let t = bonds[target_slot * BOND_STRIDE + MAX_BONDS];
    return atomicLoad(&bond_accept[target_slot]) == proposer && t.x == BOND_NONE && t.w >= 1u;
}

@compute @workgroup_size(64)
fn bond_accept_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    let mine = bonds[slot * BOND_STRIDE + MAX_BONDS];
    // A ligação dura o que dura a âncora mais fraca das duas.
    if (mine.x == BOND_NONE) {
        // Como escolhido.
        let w = atomicLoad(&bond_accept[slot]);
        if (w != BOND_NONE && accepts(slot, w)) {
            let p = bonds[w * BOND_STRIDE + MAX_BONDS];
            let k = (p.z >> 8u) & 0xFFu;
            let kp = p.z & 0xFFu;
            let pb = max(anchor_break(slot, k), anchor_break(w, kp));
            bond_add(slot, w, agents[w].id, k, kp, pb);
        }
    } else if (mine.x < params.max_agents && accepts(mine.x, slot)) {
        // Como proponente aceite.
        let k = mine.z & 0xFFu;
        let j = (mine.z >> 8u) & 0xFFu;
        let pb = max(anchor_break(slot, k), anchor_break(mine.x, j));
        bond_add(slot, mine.x, mine.y, k, j, pb);
    }
}

// LIGAÇÃO DE NASCIMENTO (chamada em agents_birth): uma âncora livre do pai
// e uma de polaridade oposta do filho ligam-se logo (sem lugar livre no
// pai, ou sem âncoras compatíveis, separam-se).
fn birth_bond(parent: u32, pa: Agent, child: u32) {
    if (pa.body_len == 0u) { return; }
    let ca = agents[child];
    if (ca.body_len == 0u) { return; }
    var slot_i = BOND_NONE;
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(parent, i).x == BOND_NONE) {
            slot_i = i;
            break;
        }
    }
    if (slot_i == BOND_NONE) { return; }
    for (var k = 0u; k < pa.body_len; k++) {
        let pol = anchor_polarity(parent, k);
        if (pol == 0.0 || anchor_busy(parent, k)) { continue; }
        for (var j = 0u; j < ca.body_len; j++) {
            if (anchor_polarity(child, j) != -pol) { continue; }
            let pb = max(anchor_break(parent, k), anchor_break(child, j));
            bond_write(parent, slot_i, child, ca.id, k, j, BOND_KIND_BIRTH, pb);
            bond_write(child, 0u, parent, pa.id, j, k, BOND_KIND_BIRTH, pb);
            return;
        }
    }
}
