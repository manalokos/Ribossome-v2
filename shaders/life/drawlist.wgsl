// Lista dos agentes vivos para o desenho (um draw indireto só com eles).
// A ordem da lista não importa: só serve para desenhar.
// TOM DA ESPÉCIE: um ângulo de cor (radianos) tirado da composição do genoma
// em pares de bases vizinhas. Cada par pesa o mesmo que o seu complementar
// invertido, por isso as duas formas de uma linhagem (o genoma e o seu
// complemento reverso) dão o MESMO tom; uma mutação muda dois pares em cem,
// por isso parentes próximos ficam com tons quase iguais e espécies
// diferentes com tons diferentes. É só para o desenho: nenhuma regra o lê.
const HUE_W1 = array<f32, 16>(0.013, -0.510, 1.000, -0.746, 0.653, 0.013, 0.139, -0.790, -0.790, -0.746, 0.567, -0.852, 0.139, 1.000, 0.344, 0.567);
const HUE_W2 = array<f32, 16>(-0.816, -0.773, -0.082, 0.750, -0.704, -0.816, -0.498, 0.338, 0.338, 0.750, 1.000, 0.233, -0.498, -0.082, -0.140, 1.000);
fn species_hue(slot: u32, gene_len: u32) -> f32 {
    var x = 0.0;
    var y = 0.0;
    var prev = genome_get(slot, 0u);
    for (var i = 1u; i < gene_len; i++) {
        let b = genome_get(slot, i);
        x += HUE_W1[prev * 4u + b];
        y += HUE_W2[prev * 4u + b];
        prev = b;
    }
    return atan2(y, x);
}

@compute @workgroup_size(64)
fn build_draw_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot == 0u) {
        atomicStore(&draw_args[0], 6u);
        atomicStore(&draw_args[4], 6u);
        atomicStore(&draw_args[12], 6u);
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
    tint_out[slot] = species_hue(slot, agents[slot].gene_len);
    // Segundo draw (argumentos 4..7), para a vista AFASTADA: 16 troços de 4
    // resíduos e a bola de marcação (LOD_INSTANCES em agents_view.wgsl).
    atomicAdd(&draw_args[5], 17u);
    // Terceiro draw (argumentos 12..15), para a MEIA distância: um troço por
    // resíduo e a bola de marcação (65 instâncias).
    atomicAdd(&draw_args[13], 65u);
}
