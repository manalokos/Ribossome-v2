// LIGAÇÕES ENTRE AGENTES POR ÂNCORAS: o órgão "âncora" (promotor tirosina,
// como as colas DOPA dos mexilhões) tem polaridade + ou − e uma força (a
// variante). Uma âncora livre que toca numa âncora livre de polaridade
// oposta de outro agente liga-se a ela; cada âncora segura UMA ligação.
// A força da variante decide quanto dura (as permanentes só se soltam se a
// ligação esticar demais ou o outro morrer). Nenhuma regra olha para o
// genoma: só para os órgãos que o corpo tem.
//
// Nascimento (GEMULAÇÃO): se o pai tiver uma âncora livre, o filho nasce ao
// pé dela e agarrado a ela (só o pai precisa do órgão; o filho agarra-se com
// uma âncora sua de polaridade oposta, se tiver, senão pela ponta do corpo).
// Uma âncora na ponta faz filamentos; várias, ramos e colónias.
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
// PARTILHA DE MATÉRIA (matter_claim, um valor por agente e por passo):
// BOND_NONE = livre; MATTER_LOCK = este agente tentou receber (não pode ser
// dador neste passo); MATTER_RECEIVED = recebeu um complemento; qualquer
// outro valor = o slot de quem o escolheu como DADOR.
const MATTER_LOCK: u32 = 0xFFFFFFFEu;
const MATTER_RECEIVED: u32 = 0xFFFFFFFDu;
const S_MATTER: u32 = 14u;
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

// 1 se o resíduo k é uma âncora, 0 se não. (As âncoras tinham polaridade +/−
// e só se ligavam às de sinal contrário; agora ligam-se a qualquer âncora.
// A coluna "polaridade" da tabela ficou só a escolher a cor do anel.)
fn anchor_polarity(slot: u32, k: u32) -> f32 {
    return select(0.0, 1.0, organ_type(organ_get(slot, k)) == ORGAN_ANCHOR);
}

// Probabilidade de quebra por passo da âncora no resíduo k.
fn anchor_break(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    // Um órgão que não é âncora (ventosa, relé agarrado pela âncora de outro) não manda na duração.
    if (organ_type(o) != ORGAN_ANCHOR) { return 0.0; }
    return max(organ_var(o).p1, 0.0);
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
    // PARTILHA DE MATÉRIA (uma folha alimenta a raiz): este agente pode
    // RECEBER, por passo, um complemento já capturado por um parceiro que
    // tenha a cópia mais adiantada, se o último que ele capturou for a base
    // de que este precisa a seguir. Para a matéria ser exata, cada agente dá
    // ou recebe no máximo um por passo: quem quer receber tranca-se primeiro
    // (deixa de poder ser dador) e depois reserva o dador, com trocas
    // atómicas; contact_apply aplica as duas metades.
    var want_matter = params.bond_matter_share > 0.0 && a.pair_count < a.gene_len
        && rng_f4(a.id, params.epoch, S_MATTER).x < params.bond_matter_share;
    var locked = false;
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
                    // (Limite POR LIGAÇÃO, igual dos dois lados: ver CONTACT_MAX_STEP.)
                    let f = d / dist * clamp((dist - BOND_LEN) * BOND_RELAX * 0.5, -BOND_MAX_STEP, BOND_MAX_STEP);
                    dp += f;
                    let r = ra - pa;
                    drot += (r.x * f.y - r.y * f.x) / (dot(r, r) + a.radius * a.radius + 1.0);
                }
                // ENERGIA: difusão pela ligação. Corre do mais CHEIO para o
                // mais vazio (energia ÷ capacidade), até os dois ficarem com
                // o mesmo enchimento; um corpo com pilha grande não esvazia
                // um pequeno. Os dois lados veem os mesmos valores, por isso
                // o que um perde é exatamente o que o outro ganha.
                let cap_a = energy_capacity(slot, a);
                let cap_o = energy_capacity(b.x, o);
                let cap_red = cap_a * cap_o / max(cap_a + cap_o, 1e-6);
                de += clamp(params.bond_energy_share, 0.0, 0.5) * cap_red * (o.energy / max(cap_o, 1e-6) - a.energy / max(cap_a, 1e-6));
                // Matéria: o parceiro tem a cópia mais adiantada (em fração
                // do genoma, e continua a tê-la depois de dar: nivela sem
                // andar para trás e para a frente) e o seu último
                // complemento é o que falta aqui?
                if (want_matter && o.pair_count > 0u && o.pair_count <= o.gene_len
                    && f32(o.pair_count - 1u) * f32(a.gene_len) >= f32(a.pair_count + 1u) * f32(o.gene_len)
                    && genome_get(b.x, o.pair_count - 1u) == genome_get(slot, a.pair_count)) {
                    if (!locked) {
                        locked = atomicCompareExchangeWeak(&matter_claim[slot], BOND_NONE, MATTER_LOCK).exchanged;
                        // Já foi escolhido como dador por outro: não recebe.
                        if (!locked) { want_matter = false; }
                    }
                    if (locked && atomicCompareExchangeWeak(&matter_claim[b.x], BOND_NONE, slot).exchanged) {
                        atomicStore(&matter_claim[slot], MATTER_RECEIVED);
                        want_matter = false;
                    }
                }
            }
        }
        if (!keep) {
            bonds[slot * BOND_STRIDE + i] = vec4<u32>(BOND_NONE, 0u, 0u, 0u);
            free += 1u;
        }
    }
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
                        // ADESÃO ESPECÍFICA: a âncora só se agarra a certos
                        // órgãos do outro corpo (os seus "recetores"), o
                        // mais próximo que lhe toque e esteja livre:
                        //   - outra âncora;
                        //   - uma ventosa (o pé de um agarra o outro);
                        //   - um relé (a ligação passa sinais: fica uma
                        //     sinapse, com o relé a decidir o que entra).
                        // A um resíduo comum não se agarra: senão qualquer
                        // toque colava e as colónias eram ao acaso.
                        var best_j = BOND_NONE;
                        var best_d = BOND_RANGE;
                        for (var j = 0u; j < b.body_len; j++) {
                            let pj = anchor_polarity(e, j);
                            let tj = organ_type(organ_get(e, j));
                            let ok = (pj != 0.0 || tj == ORGAN_HOLDFAST || tj == ORGAN_RELAY) && !anchor_busy(e, j);
                            let dj = length(residue_world(e, b, j) - ra);
                            if (ok && dj <= best_d) {
                                best_d = dj;
                                best_j = j;
                            }
                        }
                        if (best_j != BOND_NONE) {
                            bonds[prop_i] = vec4<u32>(e, b.id, k | (best_j << 8u), free);
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
// e uma do filho ligam-se logo (sem lugar livre no
// pai, ou sem âncoras compatíveis, separam-se).
// Âncora do pai onde o filho vai nascer: a primeira livre, se o pai tiver
// um lugar de ligação livre. BOND_NONE = nasce solto.
fn bud_anchor(parent: u32, pa: Agent) -> u32 {
    var found = BOND_NONE;
    var has_slot = false;
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(parent, i).x == BOND_NONE) { has_slot = true; }
    }
    if (!has_slot) { return BOND_NONE; }
    for (var k = 0u; k < pa.body_len; k++) {
        if (anchor_polarity(parent, k) != 0.0 && !anchor_busy(parent, k)) {
            found = k;
            break;
        }
    }
    return found;
}

// Liga o filho acabado de nascer à âncora `k` do pai (ver bud_anchor).
fn birth_bond(parent: u32, pa: Agent, child: u32, k: u32) {
    if (k == BOND_NONE || pa.body_len == 0u) { return; }
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
    // O filho agarra-se com uma âncora sua, se tiver; senão pela ponta.
    var j = 0u;
    var pb = anchor_break(parent, k);
    for (var q = 0u; q < ca.body_len; q++) {
        if (anchor_polarity(child, q) != 0.0) {
            j = q;
            pb = max(pb, anchor_break(child, q));
            break;
        }
    }
    bond_write(parent, slot_i, child, ca.id, k, j, BOND_KIND_BIRTH, pb);
    bond_write(child, 0u, parent, pa.id, j, k, BOND_KIND_BIRTH, pb);
}
