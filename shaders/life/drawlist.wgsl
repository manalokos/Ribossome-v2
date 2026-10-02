// Lista dos agentes vivos para o desenho (um draw indireto só com eles).
// A ordem da lista não importa: só serve para desenhar.
@compute @workgroup_size(64)
fn build_draw_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot == 0u) { atomicStore(&draw_args[0], 6u); }
    if (slot >= params.max_agents || agents[slot].alive == 0u) { return; }
    // Por agente: 64 tubos, 64 órgãos por cima e 64 bases de RNA nas pontas.
    let i = atomicAdd(&draw_args[1], 192u) / 192u;
    draw_list[i] = slot;
}
