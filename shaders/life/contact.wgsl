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
// Limite do empurrão POR PAR (unidades do mundo por passo). Tem de ser por par
// e não sobre a soma de cada agente: os dois lados de um par veem a mesma
// sobreposição e cortam igual, por isso o que um anda o outro desanda. Com o
// corte na soma, um agente com dois vizinhos era cortado e os vizinhos não, e
// um grupo de três agarrados por âncoras andava sozinho sem nadar.
const CONTACT_MAX_STEP: f32 = 2.0;

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
    atomicStore(&prey_killer[slot], NO_ENTRY);
}

// PROTEASE (predação química): um órgão. Em cada passo de contacto, as
// proteases abertas de um atacante TIRAM à vítima uma quantidade certa de
// energia: PRED_DRAIN × força das proteases × o que a vítima tem EXPOSTO
// × (1 − defesa da prolina) × params.protease_power. É determinista e
// acumula. Quando a energia chega a zero a vítima morre.
// QUEM É VULNERÁVEL: toda a gente. Um corpo sem nenhum mecanismo de proteção
// é cortado por QUALQUER protease, com a força inteira. (Antes cada família
// só cortava certos aminoácidos, e um corpo pequeno sem esses aminoácidos era
// imune por composição: o replicador mínimo escapava a tudo.) As defesas são
// duas: o órgão INIBIDOR, que bloqueia uma FAMÍLIA de proteases, e a prolina.
// A FAMÍLIA de uma protease (decidida pelo aminoácido seguinte na cadeia, o
// bolso) já não diz o que ela corta: diz que inibidor a trava.
//   família 1: vizinho ácido (D, E);   família 2: vizinho básico (K, R);
//   família 3: vizinho hidrofóbico (F, L, W, Y, I, V);
//   sem um vizinho desses é GENERALISTA: um terço da força, mas nenhum
//   inibidor a trava.
// A regra é cega (nenhum genoma é comparado) e não há autodigestão: uma
// protease só corta OUTROS; os parentes só se poupam se tiverem o inibidor.
// Para onde vai a energia tirada: params.protease_direct vai direta para o
// atacante; do resto, params.lysis_yield reativa monómeros gastos na célula
// da vítima (um por cada food_power), que qualquer boca pode comer.
const PRED_DRAIN: f32 = 0.2;
// Força 1 contra uma vítima sem proteção = PRED_DRAIN por passo. (Era 10,
// quando só contava a fração de resíduos-alvo, tipicamente uns 10%.)
const PRED_SITE_SCALE: f32 = 1.0;
const PRED_PROLINE_DEFENSE: f32 = 0.9;
const AA_PROLINE: u32 = 12u;
const BITE_SCALE: f32 = 10000.0;
// TOQUE: distância (unidades do mundo) entre o centro de uma protease e o de
// um resíduo da vítima a que a protease lhe chega (o raio do órgão com os
// dentes mais a espessura do resíduo). E quantas proteases de um corpo contam.
const PROTEASE_TOUCH: f32 = 14.0;
const PROTEASE_SLOTS: u32 = 12u;
const S_DIGEST: u32 = 13u;
const S_SPILL: u32 = 15u;

// FAMÍLIA de uma protease = o que corta, decidido pelo aminoácido SEGUINTE
// da cadeia (o bolso de especificidade; a mesma regra da antena do sensor
// de corpos, ver pocket_family): 1 = lisina e arginina, 2 = aspartato e
// asparagina, 3 = aromáticos e leucina. Sem um vizinho desses é generalista:
// corta as três, com um terço da força em cada.
fn protease_family(slot: u32, k: u32, n: u32) -> u32 {
    var f = 0u;
    if (k + 1u < n) { f = pocket_family(body_get(slot, k + 1u)); }
    return f;
}

// Pesos (família 1, 2, 3, GENERALISTA). Uma especialista corta só os alvos
// da sua família, com força inteira. A generalista (sem vizinho que defina o
// bolso) é de largo espectro: corta QUALQUER resíduo, a um terço da força.
// Assim nenhum corpo é imune por composição, por mais pequeno que seja; a
// defesa geral que resta é a prolina.
const GENERALIST_STRENGTH: f32 = 1.0 / 3.0;
fn family_weights(f: u32) -> vec4<f32> {
    var w = vec4<f32>(0.0, 0.0, 0.0, GENERALIST_STRENGTH);
    if (f == 1u) { w = vec4<f32>(1.0, 0.0, 0.0, 0.0); }
    if (f == 2u) { w = vec4<f32>(0.0, 1.0, 0.0, 0.0); }
    if (f == 3u) { w = vec4<f32>(0.0, 0.0, 1.0, 0.0); }
    return w;
}

// Quanto a protease do resíduo k está LIGADA (0..1): sempre, ou pelo sinal
// interno do canal da variante (p2).
// LIMIAR: abaixo de PROTEASE_MIN_DRIVE de sinal a protease esta fechada (nao
// morde nem gasta); acima, a forca e proporcional ao sinal. (Como um
// zimogenio: a enzima so e ativada a partir de um certo estimulo.) O desenho
// usa o mesmo limiar (agents_view.wgsl).
const PROTEASE_MIN_DRIVE: f32 = 0.25;
// TRÉGUA DA REPRODUÇÃO: quando um agente dá à luz, as proteases dele e as do
// filho ficam fechadas PROTEASE_TRUCE passos. O filho nasce encostado ao
// progenitor e é a outra forma da linhagem: sem isto, o progenitor com
// protease comia o filho ao nascer (ou o filho o progenitor), e uma linhagem
// de predadores desfazia-se a cada geração. É o tempo de os dois se
// afastarem. (Como as proteases reais, que se fabricam inativas.)
const PROTEASE_TRUCE: u32 = 15u;
fn in_truce(slot: u32) -> bool {
    let t = truce_until[slot];
    // (Um valor muito à frente é lixo de outra corrida: não conta.)
    return t > params.epoch && t - params.epoch <= PROTEASE_TRUCE;
}

fn protease_drive(slot: u32, k: u32, v: OrganVariant) -> f32 {
    if (in_truce(slot)) { return 0.0; }
    var drive = 1.0;
    if (v.p2 >= 0.0) { drive = clamp(signals[slot * MAX_BODY + k][u32(clamp(v.p2, 0.0, 3.0))], 0.0, 1.0); }
    return select(0.0, drive, drive >= PROTEASE_MIN_DRIVE);
}

// DOIS MODOS DE CAÇAR, pela variante:
//  - de CONTACTO (alcance 0): a protease está encostada à presa e o que
//    digere entra no corpo do atacante (a energia tirada, × protease_direct,
//    e a matéria da presa quando ela morre);
//  - LÍTICA (as de alcance): é enzima largada na água. Desfaz a presa cá
//    fora e o atacante não recebe nada diretamente: a energia e a matéria da
//    presa ficam no meio, como monómeros ativados, para quem os comer (ele,
//    se tiver boca, os parentes e os oportunistas). Não é uma maneira de
//    captar: não entra no orçamento de captação (ver harvest_share).
// ALCANCE: a variante (p0) diz a que distância ALÉM do contacto a protease
// chega (0 = só a tocar; as de alcance são proteases segregadas para a água
// à volta). Três escalões: contacto, médio e longo, com a força ativa de
// cada família em cada um e o alcance do escalão.
const PROTEASE_MID_FROM: f32 = 10.0;
const PROTEASE_FAR_FROM: f32 = 70.0;
// DILUIÇÃO: uma protease de alcance é enzima largada na água, e dilui-se. A
// tocar morde com a força inteira; daí até ao fim do alcance cai em linha
// reta até REACH_EDGE da força. Quem tem alcance morde primeiro mas fraco;
// quem aguenta a aproximação e chega perto morde forte.
const REACH_EDGE: f32 = 0.5;
fn reach_falloff(dist: f32, reach: f32) -> f32 {
    return 1.0 - (1.0 - REACH_EDGE) * clamp(dist / max(reach, 1.0), 0.0, 1.0);
}

// CUSTO DE ESTAR ABERTA, em residuos de manutencao: forca x alcance /
// PROTEASE_REACH_REF x PROTEASE_ACTIVE_COST x abertura. Largar enzima forte
// num volume grande de agua e caro; uns picos compridos e fracos a espera de
// presa custam pouco (como os de um heliozoario), e as de contacto nada. O
// resto do custo dos espigoes e peso e arrasto (tabela).
const PROTEASE_ACTIVE_COST: f32 = 1.0;
const PROTEASE_REACH_REF: f32 = 40.0;
fn protease_active(slot: u32, n: u32) -> f32 {
    var c = 0.0;
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_PROTEASE) {
            let v = organ_var(o);
            // força × alcance: a de contacto não gasta; picos compridos e fracos
            // gastam pouco; só "longe E forte" sai caro (alcance 100, força 3 =
            // 7,5 resíduos de manutenção).
            c += protease_drive(slot, k, v) * max(v.p1, 0.0) * organ_gain(o) * max(v.p0, 0.0) / PROTEASE_REACH_REF;
        }
    }
    return c * PROTEASE_ACTIVE_COST;
}

// IMUNIDADE às próprias proteases: quem tem uma protease de uma família
// tem também o seu inibidor (senão digeria-se a si próprio), e por isso
// resiste às dessa família vindas de outros. É o que impede dois caçadores
// iguais de se desfazerem um ao outro ao mesmo tempo. Devolve, por família,
// quanto do alvo fica exposto (1 = tudo, 1 − PROTEASE_IMMUNITY = protegido).
const PROTEASE_IMMUNITY: f32 = 1.0;
fn protease_exposed(slot: u32, n: u32) -> vec4<f32> {
    // A imunidade vem do órgão INIBIDOR, não da protease: quem ataca com uma
    // família e quer estar a salvo dela (dos parentes, dos vizinhos) tem de
    // pagar os dois órgãos; quem só tem o inibidor resiste sem atacar. É o
    // que dá o ciclo das bactérias com toxinas: o armado mata o indefeso, o
    // resistente (mais barato) cresce mais do que o armado, o indefeso (que
    // não paga nada) cresce mais do que o resistente.
    var own = vec4<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_INHIBITOR) {
            // A família é a do vizinho, como no bolso de uma protease.
            let block = clamp(max(organ_var(o).p0, 0.0) * organ_gain(o), 0.0, 1.0);
            own = max(own, sign(family_weights(protease_family(slot, k, n))) * block);
        }
    }
    // ORÇAMENTO DE DEFESA: a proteção de um corpo soma no máximo 1, repartida
    // pelas famílias que tenta cobrir. Imune a uma família = 100% nessa; a
    // duas = 50% em cada; a tudo = 25%. Quem quer estar a salvo dos seus
    // (que atacam com uma família) consegue; quem quer ser imune a toda a
    // gente não, e fica sempre uma família que o apanha: pedra, papel, tesoura.
    // A GENERALISTA NÃO TEM DEFESA: o inibidor só bloqueia as três famílias
    // especializadas. A generalista morde qualquer corpo, a um terço da força,
    // e quem a usa não pode proteger os seus (morde parentes e vizinhos de
    // colónia). Contra ela só valem o tamanho, as reservas e a prolina.
    own.w = 0.0;
    let total = own.x + own.y + own.z;
    if (total > 1.0) { own /= total; }
    return vec4<f32>(1.0) - PROTEASE_IMMUNITY * own;
}

// DEFESA de um corpo, num só número (cabe exato num f32): alvo exposto das
// famílias 1, 2 e 3 e fração de prolina, cada um 0..1 em 5 bits, e no bit 20
// se tem uma protease generalista (fica imune às generalistas dos outros).
fn pack_defence(slot: u32, n: u32) -> f32 {
    // Só conta o que o PRÓPRIO corpo tem: nada é herdado do pai nem há
    // tréguas. Um filho (a outra forma da linhagem) só está a salvo das
    // proteases do pai e dos parentes se o seu corpo também tiver o inibidor.
    let ex = protease_exposed(slot, n);
    // (O que fica exposto a cada família: tudo, menos o que o inibidor dela bloqueia.)
    let t = ex.xyz;
    let q = vec4<u32>(round(clamp(vec4<f32>(t, proline_fraction(slot, n)), vec4<f32>(0.0), vec4<f32>(1.0)) * 31.0));
    // (A imunidade às generalistas em 4 bits, proporcional como as outras:
    // num só bit, o inibidor mais fraco dava imunidade total.)
    let gen = u32(round(clamp(1.0 - ex.w, 0.0, 1.0) * 15.0));
    return f32(q.x | (q.y << 5u) | (q.z << 10u) | (q.w << 15u) | (gen << 20u));
}

// (alvo exposto das famílias 1, 2, 3; de TODOS os resíduos para a generalista).
fn unpack_targets(w: f32) -> vec4<f32> {
    let u = u32(max(w, 0.0));
    let gen = 1.0 - PROTEASE_IMMUNITY * f32((u >> 20u) & 15u) / 15.0;
    return vec4<f32>(f32(u & 31u) / 31.0, f32((u >> 5u) & 31u) / 31.0, f32((u >> 10u) & 31u) / 31.0, gen);
}

fn unpack_proline(w: f32) -> f32 {
    return f32((u32(max(w, 0.0)) >> 15u) & 31u) / 31.0;
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
    // ONDE ESTÃO AS PROTEASES: a mordida é do ÓRGÃO, não do corpo. Cada
    // protease aberta guarda a sua posição no mundo, os pesos por família e o
    // alcance; morde quem tiver um RESÍDUO a tocar-lhe (ou dentro do alcance).
    // (Antes o teste era entre os discos dos dois corpos, no centro de massa:
    // uma presa passava por cima de uma protease na ponta de um corpo comprido
    // sem ser mordida, e era mordida ao tocar no disco longe de qualquer protease.)
    var ppos: array<vec2<f32>, PROTEASE_SLOTS>;
    var pwt: array<vec4<f32>, PROTEASE_SLOTS>;
    var preach: array<f32, PROTEASE_SLOTS>;
    var np = 0u;
    {
        let harvest = harvest_share(slot, a.body_len);
        for (var k = 0u; k < a.body_len; k++) {
            let o = organ_get(slot, k);
            if (organ_type(o) == ORGAN_PROTEASE && np < PROTEASE_SLOTS) {
                let v = organ_var(o);
                // (Só as de contacto repartem o orçamento de captação.)
                let f = family_weights(protease_family(slot, k, a.body_len)) * max(v.p1, 0.0) * organ_gain(o) * protease_drive(slot, k, v) * select(harvest, 1.0, v.p0 >= PROTEASE_MID_FROM);
                if (f.x + f.y + f.z + f.w > 0.0) {
                    ppos[np] = residue_world(slot, a, k);
                    pwt[np] = f;
                    preach[np] = select(0.0, v.p0, v.p0 >= PROTEASE_MID_FROM);
                    np += 1u;
                }
            }
        }
    }
    let armed = np > 0u;
    // As células de contacto têm 120 unidades e os agentes estão lá pelo
    // centro: quem está armado olha duas células em vez de uma, para apanhar
    // corpos compridos cuja ponta lhe toca na protease.
    let span = select(1, 2, armed);
    // CONTÍNUO: uma protease aberta corrói em todos os passos em que toca na
    // vítima (PRED_DRAIN por passo). (Houve um disparo único com recarga de 60
    // passos; via-se mal o que mordia o quê e foi retirado.)
    // Energia que este agente tira a outros neste passo.
    var gained = 0.0;
    for (var dy = -span; dy <= span; dy++) {
        for (var dx = -span; dx <= span; dx++) {
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
                        push += dir * min(overlap * CONTACT_RELAX, CONTACT_MAX_STEP);
                    }
                    // Ataque: cada protease morde se algum resíduo da vítima
                    // lhe toca (PROTEASE_TOUCH); as de alcance também até essa
                    // distância para lá do toque, mais fracas (reach_falloff).
                    var sites = vec4<f32>(0.0);
                    // (o mesmo, só das proteases líticas)
                    var lytic = vec4<f32>(0.0);
                    if (armed && b.energy > 0.0) {
                        let nb = max(b.body_len, 1u);
                        for (var i = 0u; i < np; i++) {
                            var d2 = 1e12;
                            for (var j = 0u; j < nb; j++) {
                                var q = vec2<f32>(b.pos_x, b.pos_y);
                                if (b.body_len > 0u) { q = residue_world(e, b, j); }
                                let dq = q - ppos[i];
                                d2 = min(d2, dot(dq, dq));
                            }
                            let gap = max(sqrt(d2) - PROTEASE_TOUCH, 0.0);
                            if (gap <= preach[i]) {
                                let hit = pwt[i] * reach_falloff(gap, preach[i]);
                                sites += hit;
                                if (preach[i] > 0.0) { lytic += hit; }
                            }
                        }
                    }
                    if (sites.x + sites.y + sites.z + sites.w > 0.0) {
                        let power = dot(sites, unpack_targets(contact_disp[e].w)) * PRED_SITE_SCALE;
                        let resist = 1.0 - PRED_PROLINE_DEFENSE * unpack_proline(contact_disp[e].w);
                        let bite = min(PRED_DRAIN * max(params.protease_power, 0.0) * power * resist, b.energy);
                        if (bite > 0.0) {
                            // A parte lítica da mordida não é de ninguém: fica
                            // no meio (bitten_free). Só a de contacto alimenta
                            // o atacante e lhe dá direito ao corpo da presa.
                            let targets = unpack_targets(contact_disp[e].w);
                            let part = clamp(dot(lytic, targets) / max(dot(sites, targets), 1e-9), 0.0, 1.0);
                            atomicAdd(&bitten[e], u32(bite * BITE_SCALE));
                            atomicAdd(&bitten_free[e], u32(bite * part * BITE_SCALE));
                            if (part < 1.0) { atomicMin(&prey_killer[e], slot); }
                            gained += bite * (1.0 - part);
                        }
                    }
                }
                e = contact_next[e];
            }
        }
    }
    let dp = push;
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
    // VENTOSA AGARRADA: o corpo está preso ao terreno, por isso os empurrões
    // dos vizinhos e os puxões das ligações quase não o movem (divididos por
    // 1 + força com que agarra; o terreno fica com a diferença).
    let pinned = 1.0 / (1.0 + holdfast_held(slot, a));
    let np = clamp(vec2<f32>(a.pos_x, a.pos_y) + (contact_disp[slot].xy + bd.xy) * pinned, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) {
        a.pos_x = np.x;
        a.pos_y = np.y;
    }
    a.rot += bd.z * pinned;
    // PARTILHA DE MATÉRIA (decidida em bond_maintain): quem recebeu fica
    // com mais um complemento; quem foi escolhido como dador fica sem o
    // último. A base é a mesma nos dois, por isso a matéria conserva-se
    // canal a canal.
    let mc = atomicLoad(&matter_claim[slot]);
    if (mc == MATTER_RECEIVED) {
        let at = pair_first_free(slot, a.gene_len);
        if (at < a.gene_len) {
            pair_set(slot, at);
            a.pair_count += 1u;
        }
    } else if (mc != BOND_NONE && mc != MATTER_LOCK && a.pair_count > 0u) {
        let at = pair_last_set(slot, a.gene_len);
        if (at < a.gene_len) {
            pair_unset(slot, at);
            a.pair_count -= 1u;
        }
    }
    // Fica limpo já aqui: um slot que morra e renasça a meio de um passo
    // não pode herdar a reserva do morto.
    atomicStore(&matter_claim[slot], BOND_NONE);
    // O que este agente tirou a outros: a parte direta entra-lhe na energia.
    let direct = clamp(params.protease_direct, 0.0, 1.0);
    let cap_a = energy_capacity(slot, a);
    let fed = max(a.energy + contact_disp[slot].z * direct + bd.w, 0.0);
    a.energy = min(fed, cap_a);
    // O que um predador de barriga cheia ganha A MAIS não se perde: reativa
    // gastos na célula onde está (um por cada food_power), como o excesso de
    // um fotossistema. É assim que a energia das presas vira monómeros para
    // a cópia dele, ali mesmo, onde ficaram os restos.
    let spill = min(fed - a.energy, max(contact_disp[slot].z * direct, 0.0)) / max(params.food_power, 1e-3);
    if (spill > 0.0) {
        let qs = rng_f4(a.id, params.epoch, S_SPILL);
        var ns = u32(spill);
        if (qs.x < fract(spill)) { ns += 1u; }
        let cell_a = world_to_cell(vec2<f32>(a.pos_x, a.pos_y));
        let c0 = min(u32(qs.y * 4.0), 3u);
        for (var i = 0u; i < min(ns, 8u); i++) {
            for (var t = 0u; t < 4u; t++) {
                if (chem_activate_one(cell_a * 4u + (c0 + i + t) % 4u)) { break; }
            }
        }
    }
    // O que lhe tiraram: sai da energia; a parte que não foi direta para os
    // atacantes reativa gastos da célula onde está (× lysis_yield, um por
    // cada food_power, arredondamento ao acaso). Sem gastos ali, perde-se.
    let bit_all = f32(atomicExchange(&bitten[slot], 0u)) / BITE_SCALE;
    let bit_free = f32(atomicExchange(&bitten_free[slot], 0u)) / BITE_SCALE;
    let lost = min(bit_all, max(a.energy, 0.0));
    if (lost > 0.0) {
        let q = rng_f4(a.id, params.epoch, S_DIGEST);
        // (As mordidas líticas ficam inteiras no meio; das outras, só a parte
        // que não foi direta para os atacantes.)
        let free = lost * clamp(bit_free / max(bit_all, 1e-9), 0.0, 1.0);
        let want = (free + (lost - free) * (1.0 - direct)) * clamp(params.lysis_yield, 0.0, 1.0) / max(params.food_power, 1e-3);
        var count = u32(want);
        if (q.x < fract(want)) { count += 1u; }
        let cell = world_to_cell(vec2<f32>(a.pos_x, a.pos_y));
        let ch0 = min(u32(q.y * 4.0), 3u);
        for (var i = 0u; i < min(count, 24u); i++) {
            for (var t = 0u; t < 4u; t++) {
                if (chem_activate_one(cell * 4u + (ch0 + i + t) % 4u)) { break; }
            }
        }
        a.energy -= lost;
        // Sem energia: morre aqui, e conta como morte por protease.
        if (a.energy <= 0.0) {
            atomicAdd(&life_counters[LC_BITES], 1u);
            // O PREDADOR COME O CORPO: a energia que a presa trazia no bolso
            // é pouca; o grosso do que ela vale é a matéria (o genoma e os
            // complementos que já tinha capturado, cada monómero com a
            // energia de uma refeição). Quem a matou digere a fração
            // protease_direct dessa matéria: recebe a energia (no passo
            // seguinte, por prey_inbox) e esses monómeros saem GASTOS.
            // Do resto, como antes: os complementos saem ativados e
            // lysis_matter do genoma também.
            let killer = atomicLoad(&prey_killer[slot]);
            let paired = min(a.pair_count, a.gene_len);
            var eat_genome = 0u;
            var eat_paired = 0u;
            if (killer != NO_ENTRY && killer < params.max_agents) {
                eat_genome = min(u32(f32(a.gene_len) * direct), a.gene_len);
                eat_paired = min(u32(f32(paired) * direct), paired);
                atomicAdd(&prey_inbox[killer], u32(f32(eat_genome + eat_paired) * max(params.food_power, 0.0) * BITE_SCALE));
            }
            die_release(slot, a, u32(f32(a.gene_len - eat_genome) * clamp(params.lysis_matter, 0.0, 1.0)), eat_paired);
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
