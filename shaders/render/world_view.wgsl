// Vista do mundo: um triângulo de ecrã inteiro; cada píxel lê a célula da
// grelha por baixo dele. Cima no ecrã = +y no mundo.
// Paleta provisória da fase 1: a paleta do v3 (composite.wgsl) é portada na fase 2.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> chem_view: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// Cores provisórias dos 4 nucleótidos.
fn channel_color(ch: u32) -> vec3<f32> {
    switch ch {
        case 0u: { return vec3<f32>(0.95, 0.35, 0.30); } // A
        case 1u: { return vec3<f32>(0.95, 0.85, 0.30); } // U
        case 2u: { return vec3<f32>(0.35, 0.90, 0.40); } // G
        default: { return vec3<f32>(0.35, 0.55, 1.00); } // C
    }
}

@fragment
fn fs_world(in: VsOut) -> @location(0) vec4<f32> {
    // Píxel -> mundo (y invertido: o ecrã cresce para baixo, o mundo para cima).
    let px = in.pos.xy - 0.5 * vec2<f32>(view.screen_w, view.screen_h);
    let world = vec2<f32>(view.center_x + px.x / view.zoom, view.center_y - px.y / view.zoom);
    // Mundo -> célula, explicitamente.
    let cell_f = floor(world / f32(WORLD_UNITS_PER_CELL));
    if (cell_f.x < 0.0 || cell_f.y < 0.0 || cell_f.x >= f32(GRID_SIZE) || cell_f.y >= f32(GRID_SIZE)) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }
    let idx = u32(cell_f.y) * GRID_SIZE + u32(cell_f.x);

    var act = vec4<f32>(0.0);
    var spent = vec4<f32>(0.0);
    for (var ch = 0u; ch < 4u; ch++) {
        let v = chem_view[idx * 4u + ch];
        act[ch] = f32(v & 0xFFFFu);
        spent[ch] = f32(v >> 16u);
    }
    let water = vec3<f32>(0.03, 0.07, 0.12);

    if (view.view_mode >= 1u && view.view_mode <= 4u) {
        let ch = view.view_mode - 1u;
        let t = clamp(act[ch] / 12.0, 0.0, 1.0);
        return vec4<f32>(mix(water, channel_color(ch), t), 1.0);
    }
    if (view.view_mode == 5u) {
        let t = clamp(dot(spent, vec4<f32>(1.0)) / 24.0, 0.0, 1.0);
        return vec4<f32>(mix(water, vec3<f32>(0.75), t), 1.0);
    }

    // Normal: tom = mistura dos canais ativados; gastos puxam para cinzento.
    let act_sum = dot(act, vec4<f32>(1.0));
    let total = act_sum + dot(spent, vec4<f32>(1.0));
    if (total <= 0.0) { return vec4<f32>(water, 1.0); }
    var col = vec3<f32>(0.0);
    for (var ch = 0u; ch < 4u; ch++) {
        col += channel_color(ch) * act[ch];
    }
    let hue = select(vec3<f32>(0.5), col / max(act_sum, 1e-6), act_sum > 0.0);
    let base = mix(vec3<f32>(0.45), hue, act_sum / total);
    let density = clamp(total / f32(CHEM_CELL_CAP), 0.0, 1.0);
    return vec4<f32>(mix(water, base, sqrt(density)), 1.0);
}
