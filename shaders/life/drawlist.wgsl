// Lista dos agentes vivos para o desenho (um draw indireto só com eles).
// A ordem da lista não importa: só serve para desenhar.
@compute @workgroup_size(64)
fn build_draw_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot == 0u) {
        atomicStore(&draw_args[0], 6u);
        atomicStore(&draw_args[4], 6u);
    }
    if (slot >= params.max_agents || agents[slot].alive == 0u) { return; }
    // Só os que estão À VISTA: draw_args[8..11] é o retângulo da câmara no
    // mundo (bits de f32, já com folga para o corpo), escrito pelo CPU.
    let lo = vec2<f32>(bitcast<f32>(atomicLoad(&draw_args[8])), bitcast<f32>(atomicLoad(&draw_args[9])));
    let hi = vec2<f32>(bitcast<f32>(atomicLoad(&draw_args[10])), bitcast<f32>(atomicLoad(&draw_args[11])));
    let pw = vec2<f32>(agents[slot].pos_x, agents[slot].pos_y);
    if (any(pw < lo) || any(pw > hi)) { return; }
    // Por agente: 64 tubos, 64 órgãos por cima e 64 bases de RNA nas pontas.
    // Instâncias por agente: 64 tubos, 64 órgãos, 64 bases de RNA, 4 ligações
    // e a bola do parentesco (AGENT_INSTANCES em agents_view.wgsl).
    let i = atomicAdd(&draw_args[1], 197u) / 197u;
    draw_list[i] = slot;
    // Segundo draw (argumentos 4..7), para a vista AFASTADA: 16 troços de 4
    // resíduos e a bola de marcação (LOD_INSTANCES em agents_view.wgsl).
    atomicAdd(&draw_args[5], 17u);
}
