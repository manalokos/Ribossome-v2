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
    // Defesa contra proteases (fração de prolina), calculada uma vez por
    // passo: os atacantes leem-na em contact_disp.w.
    contact_disp[slot] = vec4<f32>(0.0, 0.0, 0.0, proline_fraction(slot, agents[slot].body_len));
}

// PROTEASE (predação química): ao tocar noutro agente, um agente com
// proteases parte-lhe proteína e tira-lhe energia (fica com PRED_EFFICIENCY
// dela). A matéria fica com a vítima e volta ao meio quando ela morre.
// Defesa química: corpos ricos em PROLINA resistem (como algumas proteínas
// reais). Nenhuma regra olha para o genoma.
const PRED_BITE: f32 = 0.01;
const PRED_EFFICIENCY: f32 = 0.5;
const PRED_PROLINE_DEFENSE: f32 = 0.9;
const AA_PROLINE: u32 = 12u;
const BITE_SCALE: f32 = 1000.0;

// (força total, alcance máximo) das proteases do agente (variantes).
fn protease_power(slot: u32, n: u32) -> vec2<f32> {
    var pw = vec2<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let o = organ_get(slot, k);
        if (organ_type(o) == ORGAN_PROTEASE) {
            let v = organ_var(o);
            pw.x += max(v.p0, 0.0) * organ_gain(o);
            pw.y = max(pw.y, v.p1);
        }
    }
    return pw;
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
    let prot = protease_power(slot, a.body_len);
    let power = prot.x;
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
                    // Mordida: em contacto, ou até ao alcance da protease.
                    if (overlap + prot.y > 0.0) {
                        if (power > 0.0 && b.energy > 0.0) {
                            let resist = 1.0 - PRED_PROLINE_DEFENSE * contact_disp[e].w;
                            let bite = min(PRED_BITE * max(params.protease_power, 0.0) * power * resist, b.energy);
                            if (bite > 0.0) {
                                atomicAdd(&bitten[e], u32(bite * BITE_SCALE));
                                gained += bite * PRED_EFFICIENCY;
                                atomicAdd(&life_counters[LC_BITES], 1u);
                            }
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
    // Predação: o que este agente mordeu e o que lhe morderam.
    let lost = f32(atomicExchange(&bitten[slot], 0u)) / BITE_SCALE;
    a.rot += bd.z;
    a.energy = min(max(a.energy + contact_disp[slot].z + bd.w, 0.0), energy_capacity(slot, a)) - lost;
    agents[slot] = a;
    // Para a vista (flash das proteases): .x = energia que lhe morderam
    // neste passo, .z = a que ganhou a morder. contact_build repõe tudo no
    // passo seguinte.
    contact_disp[slot] = vec4<f32>(lost, 0.0, contact_disp[slot].z, contact_disp[slot].w);
}
