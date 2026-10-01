// Agentes: um quadrado instanciado por RESÍDUO (slot·64 + k), desenhado
// como disco com a cor da classe química (classes do v3). Instâncias sem
// resíduo viram triângulos degenerados. O custo depende dos píxeis
// cobertos, não do número de agentes. RNA nu (sem corpo) = disco cinzento.
// O desenho fino dos aminoácidos por SDF (v3 amino_render.wgsl) chega na fase 4.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> agents_view: array<Agent>;
@group(0) @binding(2) var<storage, read> bodies_view: array<u32>;
@group(0) @binding(3) var<storage, read> body_pos_view: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> draw_list_view: array<u32>;
@group(0) @binding(5) var<storage, read> organs_view: array<u32>;

const MAX_BODY_V: u32 = 64u;

struct AgentVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    // 1 = órgão (desenhado maior, com anel branco).
    @location(2) @interpolate(flat) organ: u32,
};

// Classes (v3): alifáticos A I L M V, aromáticos F W Y, polares S T N Q,
// C à parte, positivos K R H, negativos D E, G e P especiais.
fn class_color(aa: u32) -> vec3<f32> {
    switch aa {
        case 0u, 7u, 9u, 10u, 17u: { return vec3<f32>(0.72, 0.72, 0.62); } // A I L M V
        case 4u, 18u, 19u: { return vec3<f32>(0.70, 0.45, 0.95); }        // F W Y
        case 15u, 16u, 11u, 13u: { return vec3<f32>(0.40, 0.85, 0.45); }  // S T N Q
        case 1u: { return vec3<f32>(0.95, 0.90, 0.30); }                  // C
        case 8u, 14u, 6u: { return vec3<f32>(0.35, 0.55, 1.00); }         // K R H
        case 2u, 3u: { return vec3<f32>(1.00, 0.35, 0.30); }              // D E
        case 5u: { return vec3<f32>(0.95, 0.95, 0.95); }                  // G
        default: { return vec3<f32>(1.00, 0.60, 0.20); }                  // P
    }
}

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> AgentVsOut {
    var o: AgentVsOut;
    let slot = draw_list_view[inst / MAX_BODY_V];
    let k = inst % MAX_BODY_V;
    let a = agents_view[slot];
    let naked = a.body_len == 0u;
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot;
    if (a.alive == 0u || hidden || (k >= a.body_len && !(naked && k == 0u))) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    var centre = vec2<f32>(a.pos_x, a.pos_y);
    var r_world = 6.0;
    var col = vec3<f32>(0.55, 0.55, 0.55);
    var is_organ = 0u;
    if (!naked) {
        let aa = (bodies_view[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
        let lp = body_pos_view[slot * MAX_BODY_V + k];
        let cr = cos(a.rot);
        let sr = sin(a.rot);
        centre += vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
        // Espessura (v3): 4·√(volume/130), mais folga para os discos se tocarem.
        var vol = AA_VOLUME;
        r_world = 4.0 * sqrt(vol[aa] / 130.0) + 2.0;
        col = class_color(aa);
        if (((organs_view[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu) != 0u) {
            is_organ = 1u;
            r_world *= 1.5;
        }
    }
    // Nunca menos de 1,5 píxeis, para se ver com o zoom afastado.
    let r = max(r_world, 1.5 / view.zoom);
    let w = centre + c * r;
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.local = c;
    o.organ = is_organ;
    // Pouca energia = mais escuro.
    o.color = col * mix(0.35, 1.0, clamp(a.energy / max(f32(a.body_len), 1.0), 0.0, 1.0));
    return o;
}

@fragment
fn fs_agent(in: AgentVsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if (d > 1.0) { discard; }
    let rim = smoothstep(0.75, 0.98, d);
    if (in.organ != 0u) {
        return vec4<f32>(mix(in.color, vec3<f32>(1.0), smoothstep(0.6, 0.8, d)), 1.0);
    }
    return vec4<f32>(mix(in.color, vec3<f32>(0.0), rim * 0.7), 1.0);
}
