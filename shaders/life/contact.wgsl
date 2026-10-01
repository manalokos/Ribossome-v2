// CONTACTO ENTRE AGENTES: repulsão estérica entre resíduos (fase 3).
//
// Cada resíduo é um disco com o raio da cadeia lateral (v3: 4·√(volume/130)
// unidades do mundo); o RNA nu é um disco de RESIDUE_NAKED_R no centro.
// Uma grelha espacial à resolução do ambiente (células de 30 unidades, mais
// do que dois raios) guarda, por célula, uma lista ligada de resíduos.
// Cada agente soma os empurrões de sobreposição dos resíduos dos OUTROS
// agentes e responde como corpo rígido sobreamortecido (baixo Reynolds:
// sem inércia; desloca-se pela média dos empurrões e roda pelo binário).
// Os deslocamentos ficam em contact_disp e só são aplicados num passe
// seguinte (todos leem as posições do mesmo instante).

const RESIDUE_NAKED_R: f32 = 4.0;
const NO_ENTRY: u32 = 0xFFFFFFFFu;
// Fração da sobreposição corrigida por passo (cada lado faz metade).
const CONTACT_RELAX: f32 = 0.5;
// Limite de deslocamento por passo (unidades do mundo) e de rotação (rad).
const CONTACT_MAX_STEP: f32 = 4.0;
const CONTACT_MAX_ROT: f32 = 0.1;

fn residue_radius(aa: u32) -> f32 {
    var vol = AA_VOLUME;
    return 4.0 * sqrt(vol[aa] / 130.0);
}

// Posição e raio de um resíduo (RNA nu: k = 0 no centro).
fn contact_site(slot: u32, a: Agent, k: u32) -> vec3<f32> {
    if (a.body_len == 0u) {
        return vec3<f32>(a.pos_x, a.pos_y, RESIDUE_NAKED_R);
    }
    let p = residue_world(slot, a, k);
    return vec3<f32>(p.x, p.y, residue_radius(body_get(slot, k)));
}

@compute @workgroup_size(256)
fn contact_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= GRID_SIZE * GRID_SIZE) { return; }
    atomicStore(&contact_head[i], NO_ENTRY);
}

// Uma thread por (slot, resíduo).
@compute @workgroup_size(256)
fn contact_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.y * 65535u * 256u + gid.x;
    let slot = e / MAX_BODY;
    let k = e % MAX_BODY;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u || k >= max(a.body_len, 1u)) { return; }
    let s = contact_site(slot, a, k);
    let cell = world_to_cell(s.xy);
    contact_next[e] = atomicExchange(&contact_head[cell], e);
}

// Uma thread por agente: soma os empurrões e guarda (dx, dy, dθ).
@compute @workgroup_size(64)
fn contact_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    let centre = vec2<f32>(a.pos_x, a.pos_y);
    let n = max(a.body_len, 1u);
    var push = vec2<f32>(0.0);
    var torque = 0.0;
    var inertia = 0.0;
    for (var k = 0u; k < n; k++) {
        let s = contact_site(slot, a, k);
        let r = s.xy - centre;
        inertia += dot(r, r);
        let c = world_to_cell(s.xy);
        let cx = i32(c % GRID_SIZE);
        let cy = i32(c / GRID_SIZE);
        var pk = vec2<f32>(0.0);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let x = cx + dx;
                let y = cy + dy;
                if (x < 0 || y < 0 || x >= i32(GRID_SIZE) || y >= i32(GRID_SIZE)) { continue; }
                var e = atomicLoad(&contact_head[u32(y) * GRID_SIZE + u32(x)]);
                var guard = 0u;
                loop {
                    if (e == NO_ENTRY || guard >= 256u) { break; }
                    guard += 1u;
                    let other = e / MAX_BODY;
                    if (other != slot) {
                        let s2 = contact_site(other, agents[other], e % MAX_BODY);
                        let d = s.xy - s2.xy;
                        let dist = length(d);
                        let overlap = s.z + s2.z - dist;
                        if (overlap > 0.0) {
                            // Coincidentes: direção determinista pelo par.
                            var dir = vec2<f32>(1.0, 0.0);
                            if (dist > 1e-4) {
                                dir = d / dist;
                            } else if (slot < other) {
                                dir = vec2<f32>(-1.0, 0.0);
                            }
                            pk += dir * overlap;
                        }
                    }
                    e = contact_next[e];
                }
            }
        }
        push += pk;
        torque += r.x * pk.y - r.y * pk.x;
    }
    var dp = push / f32(n) * CONTACT_RELAX;
    let dl = length(dp);
    if (dl > CONTACT_MAX_STEP) { dp *= CONTACT_MAX_STEP / dl; }
    var dth = 0.0;
    if (inertia > 1e-3) {
        dth = clamp(torque / inertia * CONTACT_RELAX, -CONTACT_MAX_ROT, CONTACT_MAX_ROT);
    }
    contact_disp[slot] = vec4<f32>(dp, dth, 0.0);
}

// Aplica os deslocamentos (não entra em rocha sólida).
@compute @workgroup_size(64)
fn contact_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    var a = agents[slot];
    if (a.alive == 0u) { return; }
    let d = contact_disp[slot];
    let np = clamp(vec2<f32>(a.pos_x, a.pos_y) + d.xy, vec2<f32>(0.0), vec2<f32>(SIM_SIZE - 0.01));
    if (gamma_count(world_to_cell(np)) < GAMMA_SOLID_THRESHOLD) {
        a.pos_x = np.x;
        a.pos_y = np.y;
    }
    a.rot += d.z;
    agents[slot] = a;
}
