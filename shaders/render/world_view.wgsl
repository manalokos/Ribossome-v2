// Vista do mundo: um triângulo de ecrã inteiro; cada píxel lê a célula da
// grelha por baixo dele. Cima no ecrã = +y no mundo.
// Paleta provisória: a paleta do v3 (composite.wgsl) é portada no fim da fase 2.
// Vistas: 1–4 ativados por canal, 5 gastos, 6 terreno, 7 temperatura, 8 UV, 9 fluido.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> chem_view: array<u32>;
@group(0) @binding(2) var<storage, read> gamma_view: array<u32>;
@group(0) @binding(3) var<storage, read> light_view: array<f32>;
@group(0) @binding(4) var<storage, read> temp_view: array<f32>;
@group(0) @binding(5) var<storage, read> velocity_view: array<vec2<f32>>;

// Célula do fluido debaixo de uma posição do mundo.
fn fluid_index_at_world(w: vec2<f32>) -> u32 {
    let f = clamp(floor(w / SIM_SIZE * f32(FLUID_SIZE)), vec2<f32>(0.0), vec2<f32>(f32(FLUID_SIZE - 1u)));
    return u32(f.y) * FLUID_SIZE + u32(f.x);
}

// Rampa térmica: preto -> vermelho -> amarelo -> branco.
fn heat_ramp(t: f32) -> vec3<f32> {
    let u = clamp(t, 0.0, 1.0);
    return vec3<f32>(clamp(u * 3.0, 0.0, 1.0), clamp(u * 3.0 - 1.0, 0.0, 1.0), clamp(u * 3.0 - 2.0, 0.0, 1.0));
}

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
    // 6: terreno (quanta de gamma; rocha >= 3 a branco).
    if (view.view_mode == 6u) {
        let g = f32(gamma_view[idx]);
        return vec4<f32>(mix(vec3<f32>(0.02, 0.03, 0.05), vec3<f32>(0.85, 0.8, 0.7), clamp(g / 3.0, 0.0, 1.0)), 1.0);
    }
    // 7: temperatura (0..12), com a isotérmica de ativação (T = 2) marcada.
    if (view.view_mode == 7u) {
        let t = temp_view[fluid_index_at_world(world)];
        var c = heat_ramp(t / 12.0);
        if (abs(t - 2.0) < 0.08) { c = vec3<f32>(0.2, 0.9, 1.0); }
        return vec4<f32>(c, 1.0);
    }
    // 8: luz UV (raiz quadrada, para ver o fundo).
    if (view.view_mode == 8u) {
        let l = sqrt(clamp(light_view[idx], 0.0, 1.0));
        return vec4<f32>(vec3<f32>(0.75, 0.6, 1.0) * l, 1.0);
    }
    // 9: velocidade do fluido (|v| em células/s; escala 0..50) e direção no tom.
    if (view.view_mode == 9u) {
        let v = velocity_view[fluid_index_at_world(world)];
        let sp = clamp(length(v) / 50.0, 0.0, 1.0);
        let dir = select(vec2<f32>(0.0), normalize(v), length(v) > 1e-6);
        let hue = vec3<f32>(0.5 + 0.5 * dir.x, 0.5 + 0.5 * dir.y, 0.6);
        return vec4<f32>(hue * sqrt(sp), 1.0);
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
