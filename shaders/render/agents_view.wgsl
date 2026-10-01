// Agentes: um quadrado instanciado por slot, desenhado como disco (fase 3a).
// Slots livres viram triângulos degenerados. O custo depende dos píxeis
// cobertos, não do número de agentes (nada de desenho por agente em compute).
// O desenho dos aminoácidos por SDF (v3 amino_render.wgsl) chega na fase 4.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> agents_view: array<Agent>;

struct AgentVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) slot: u32) -> AgentVsOut {
    var o: AgentVsOut;
    let a = agents_view[slot];
    if (a.alive == 0u) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    // Dois triângulos: cantos (-1,-1) (1,-1) (-1,1) (-1,1) (1,-1) (1,1).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    // Raio em unidades do mundo: cresce com o comprimento do genoma; nunca
    // menos de 3 píxeis, para se ver com o zoom afastado.
    let r_world = max(f32(WORLD_UNITS_PER_CELL) * (0.6 + 0.08 * sqrt(f32(a.gene_len))), 3.0 / view.zoom);
    let w = vec2<f32>(a.pos_x, a.pos_y) + c * r_world;
    // Mundo -> píxel -> NDC (cima no ecrã = +y no mundo).
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.local = c;
    // Cor: geração 0 a branco; descendentes a ciano. Mais escuro com pouca energia.
    let base = select(vec3<f32>(0.3, 1.0, 0.9), vec3<f32>(1.0, 1.0, 1.0), a.generation == 0u);
    o.color = base * mix(0.35, 1.0, clamp(a.energy / 5.0, 0.0, 1.0));
    return o;
}

@fragment
fn fs_agent(in: AgentVsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if (d > 1.0) { discard; }
    // Anel escuro na borda para se distinguir da sopa.
    let rim = smoothstep(0.7, 0.95, d);
    return vec4<f32>(mix(in.color, vec3<f32>(0.0), rim * 0.8), 1.0);
}
