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
                }
                e = contact_next[e];
            }
        }
    }
    var dp = push * CONTACT_RELAX;
    let dl = length(dp);
    if (dl > CONTACT_MAX_STEP) { dp *= CONTACT_MAX_STEP / dl; }
    contact_disp[slot] = vec4<f32>(dp, 0.0, 0.0);
}

// Aplica os deslocamentos (não entra em rocha sólida).
@compute @workgroup_size(64)
fn contact_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }
    let np = clamp(vec2<f32>(a.pos_x, a.pos_y) + contact_disp[slot].xy, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) {
        a.pos_x = np.x;
        a.pos_y = np.y;
    }
    agents[slot] = a;
}
