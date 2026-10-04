// Agentes: DUAS instâncias por resíduo. Primeiro um TUBO (cápsula do
// resíduo k até ao k+1, com a cor da classe química do v3 e sombreado de
// cilindro); depois, só nos órgãos, a forma do órgão por cima (SDF no
// fragmento, orientada pela cadeia):
//   boca        disco com uma abertura escura virada para fora;
//   músculo     elipse ao longo da cadeia, com estrias;
//   sensores    TOTAIS: coroa de 6 antenas; DIRECIONAIS: 2 antenas, uma de
//               cada lado da cadeia (comida verde, luz amarela);
//   energia     disco com anel interior;
//   relógio     mostrador com um ponteiro que roda com o período;
//   relé        losango;
//   armazenamento disco grande com anéis.
// Vista de sinais (view.signal_view): a cor passa a ser o sinal α e/ou β.
// Instâncias sem resíduo viram triângulos degenerados. RNA nu = disco cinzento.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> agents_view: array<Agent>;
@group(0) @binding(2) var<storage, read> bodies_view: array<u32>;
@group(0) @binding(3) var<storage, read> body_pos_view: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> draw_list_view: array<u32>;
@group(0) @binding(5) var<storage, read> organs_view: array<u32>;
@group(0) @binding(6) var<storage, read> signals_view: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> aa_props_view: array<AaProps, 20>;
@group(0) @binding(8) var<storage, read> genomes_view: array<u32>;
@group(0) @binding(9) var<storage, read> rna_tail_view: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> organ_variants_view: array<OrganVariant>;
// Ligações entre agentes (shaders/life/bonds.wgsl): 5 vec4<u32> por slot.
@group(0) @binding(11) var<storage, read> bonds_view: array<vec4<u32>>;
// Semelhança genética de cada slot com o selecionado (−1 = sem dados).
@group(0) @binding(12) var<storage, read> kin_view: array<f32>;
// Mordidas do último passo por agente: .x = energia que lhe tiraram, .z = a
// que ganhou a morder (shaders/life/contact.wgsl).
@group(0) @binding(13) var<storage, read> bite_view: array<vec4<f32>>;

// FLASH DAS PROTEASES: quem está a morder fica com as proteases maiores e
// amarelo-claras; a vítima fica com o corpo vermelho vivo.
const BITE_ORGAN_GROW: f32 = 2.2;
const BITE_ATTACK_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.45);
const BITE_VICTIM_COLOR: vec3<f32> = vec3<f32>(1.0, 0.12, 0.08);

// Fios de RNA nas pontas (as zonas não traduzidas): bases desenhadas por
// agente (metade para cada ponta), distância entre bases e ondulação.
const RNA_PER_END: u32 = 32u;
const RNA_SPACING: f32 = 5.0;
const RNA_RADIUS: f32 = 1.6;
const RNA_WIGGLE: f32 = 0.35;

fn base_color(b: u32) -> vec3<f32> {
    switch b {
        case 0u: { return vec3<f32>(1.0, 0.25, 0.2); }   // A
        case 1u: { return vec3<f32>(1.0, 0.85, 0.2); }   // U
        case 2u: { return vec3<f32>(0.25, 0.9, 0.3); }   // G
        default: { return vec3<f32>(0.3, 0.55, 1.0); }   // C
    }
}

fn genome_base(slot: u32, i: u32) -> u32 {
    return (genomes_view[slot * 16u + i / 16u] >> ((i % 16u) * 2u)) & 3u;
}

const MAX_BODY_V: u32 = 64u;
// Ligações por agente e vec4 por slot (iguais a MAX_BONDS / BOND_STRIDE).
const BONDS_V: u32 = 4u;
const BOND_STRIDE_V: u32 = 5u;
// Instâncias por agente: tubos, órgãos, bases de RNA, ligações e a bola do
// parentesco.
const AGENT_INSTANCES: u32 = 3u * MAX_BODY_V + BONDS_V + 1u;
// Raio da bola do parentesco, em píxeis do ecrã (igual para todos).
const KIN_DOT_PX: f32 = 5.0;
const NO_ORGAN: u32 = 0xFFu;
// Tamanho de um órgão em relação a um resíduo estrutural.
const ORGAN_SCALE: f32 = 2.2;

struct AgentVsOut {
    @builtin(position) pos: vec4<f32>,
    // 0 = tubo (segmento do resíduo k até ao k+1), 1 = órgão por cima.
    @location(5) @interpolate(flat) mode: u32,
    // Coordenadas no quadrado (−1..1), eixos do mundo.
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    // Tipo do órgão (NO_ORGAN se não houver).
    @location(2) @interpolate(flat) organ: u32,
    // Tangente da cadeia (mundo, unitária).
    @location(3) @interpolate(flat) tangent: vec2<f32>,
    // Raio do disco do resíduo em fração do quadrado; ângulo do relógio.
    @location(4) @interpolate(flat) core_phase: vec2<f32>,
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

// Cor de um sinal com sinal: positivo -> `pos`, negativo -> `neg`, zero -> cinzento escuro.
fn signed_color(v: f32, pos: vec3<f32>, neg: vec3<f32>) -> vec3<f32> {
    let t = tanh(abs(v));
    return mix(vec3<f32>(0.18), select(neg, pos, v >= 0.0), t);
}

// Tamanho do quadrado em múltiplos do raio do disco, por tipo de órgão.
fn organ_extent(t: u32) -> f32 {
    switch t {
        case ORGAN_FOOD_SENSOR, ORGAN_LIGHT_SENSOR: { return 2.6; }
        case ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: { return 3.4; }
        case ORGAN_MUSCLE: { return 1.6; }
        case ORGAN_STORAGE: { return 1.9; }
        case ORGAN_PHOTOSYSTEM: { return 1.8; }
        case ORGAN_PROTEASE: { return 1.7; }
        case ORGAN_ANCHOR: { return 1.5; }
        case ORGAN_BIAS: { return 1.0; }
        case NO_ORGAN: { return 1.0; }
        default: { return 1.3; }
    }
}

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> AgentVsOut {
    var o: AgentVsOut;
    // Instâncias por agente: 0..63 tubos, 64..127 órgãos (por cima),
    // 128..191 bases de RNA não traduzidas nas pontas.
    let slot = draw_list_view[inst / AGENT_INSTANCES];
    let local_i = inst % AGENT_INSTANCES;
    let a = agents_view[slot];
    if (local_i == 3u * MAX_BODY_V + BONDS_V) {
        return kin_vertex(vi, slot, a);
    }
    if (local_i >= 3u * MAX_BODY_V) {
        return bond_vertex(vi, slot, a, local_i - 3u * MAX_BODY_V);
    }
    if (local_i >= 2u * MAX_BODY_V) {
        return rna_vertex(vi, slot, a, local_i - 2u * MAX_BODY_V);
    }
    let glyph = local_i >= MAX_BODY_V;
    let k = local_i % MAX_BODY_V;
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
    var organ = NO_ORGAN;
    var tangent = vec2<f32>(1.0, 0.0);
    var phase = 0.0;
    let bite = bite_view[slot];
    var flash = vec3<f32>(-1.0);
    if (bite.x > 0.0) { flash = BITE_VICTIM_COLOR; }
    if (!naked) {
        let base = slot * MAX_BODY_V;
        let aa = (bodies_view[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
        let lp = body_pos_view[base + k];
        let cr = cos(a.rot);
        let sr = sin(a.rot);
        centre += vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
        // Tangente da cadeia (vizinhos), rodada para o mundo.
        let ka = select(k - 1u, k, k == 0u);
        let kb = select(k + 1u, k, k + 1u >= a.body_len);
        let tl = body_pos_view[base + kb] - body_pos_view[base + ka];
        let tn = select(vec2<f32>(1.0, 0.0), tl / length(tl), length(tl) > 1e-5);
        tangent = vec2<f32>(cr * tn.x - sr * tn.y, sr * tn.x + cr * tn.y);
        // Espessura (v3): 4·√(volume/130), mais folga para os discos se tocarem.
        // Estruturais finos (a cadeia); os órgãos destacam-se (ORGAN_SCALE).
        r_world = 2.6 * sqrt(aa_props_view[aa].volume / 130.0) + 1.3;
        col = class_color(aa);
        let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
        if (oc != 0u) {
            organ = (oc & 0x1Fu) - 1u;
            r_world *= ORGAN_SCALE;
            if (organ == ORGAN_PROTEASE) {
                col = vec3<f32>(0.9, 0.2, 0.2);
                if (bite.z > 0.0 && glyph) {
                    flash = BITE_ATTACK_COLOR;
                    r_world *= BITE_ORGAN_GROW;
                }
            }
            if (organ == ORGAN_BIAS) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                let beta = organ_variants_view[ORGAN_BIAS * ORGAN_VARIANTS + p].p0 >= 0.5;
                col = select(vec3<f32>(1.0, 0.55, 0.15), vec3<f32>(0.35, 0.95, 0.35), beta);
            }
            if (organ == ORGAN_ANCHOR) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                let plus = organ_variants_view[ORGAN_ANCHOR * ORGAN_VARIANTS + p].p0 >= 0.0;
                col = select(vec3<f32>(0.25, 0.5, 1.0), vec3<f32>(1.0, 0.3, 0.25), plus);
            }
            if (organ == ORGAN_CLOCK) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                // Período da variante (o ponteiro ignora a modulação por α/β).
                let period = max(organ_variants_view[ORGAN_CLOCK * ORGAN_VARIANTS + p].p1, 2.0);
                phase = 6.2831853 * f32(a.age) / period;
            }
        }
        let s = signals_view[base + k];
        switch view.signal_view {
            case 1u: { col = signed_color(s.x, vec3<f32>(1.0, 0.45, 0.1), vec3<f32>(0.1, 0.6, 1.0)); }
            case 2u: { col = signed_color(s.y, vec3<f32>(0.3, 1.0, 0.3), vec3<f32>(0.95, 0.3, 0.9)); }
            case 3u: { col = vec3<f32>(0.5 + 0.5 * tanh(s.x), 0.5 + 0.5 * tanh(s.y), 0.35); }
            case 5u: { col = vec3<f32>(0.5 + 0.5 * tanh(s.z), 0.5 + 0.5 * tanh(s.w), 0.35); }
            default: {}
        }
    }
    // Pouca energia = mais escuro (só na vista química).
    let dim = mix(0.35, 1.0, clamp(a.energy / max(f32(a.body_len), 1.0), 0.0, 1.0));
    o.color = select(col, col * dim, view.signal_view == 0u || view.signal_view == 4u);
    if (flash.x >= 0.0 && view.signal_view == 0u) { o.color = flash; }
    if (!glyph && !naked) {
        // TUBO: cápsula do resíduo k até ao k+1 (o último só tem a ponta).
        // A espessura é a do resíduo k sem o aumento dos órgãos.
        var r_tube = r_world;
        if (organ != NO_ORGAN) { r_tube /= ORGAN_SCALE; }
        r_tube = max(r_tube, 1.0 / view.zoom);
        var b = centre;
        if (k + 1u < a.body_len) {
            let lp1 = body_pos_view[slot * MAX_BODY_V + k + 1u];
            let cr = cos(a.rot);
            let sr = sin(a.rot);
            b = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * lp1.x - sr * lp1.y, sr * lp1.x + cr * lp1.y);
        }
        let seg = b - centre;
        let l = length(seg);
        let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
        let nn = vec2<f32>(-e.y, e.x);
        // Quadrado orientado que cobre a cápsula.
        let along = select(-r_tube, l + r_tube, c.x > 0.0);
        let w = centre + e * along + nn * (c.y * r_tube);
        let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
        o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
        o.mode = 0u;
        o.local = w - centre;
        o.tangent = seg;
        o.core_phase = vec2<f32>(r_tube, 0.0);
        o.organ = organ;
        return o;
    }
    if (glyph && !naked && organ == NO_ORGAN) {
        // Resíduo estrutural: só o tubo.
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    if (glyph && naked) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    let ext = organ_extent(organ);
    // Nunca menos de 1,5 píxeis, para se ver com o zoom afastado.
    let r = max(r_world * ext, 1.5 / view.zoom);
    let w = centre + c * r;
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.local = c;
    o.organ = organ;
    o.tangent = tangent;
    o.core_phase = vec2<f32>(1.0 / ext, phase);
    o.mode = 1u;
    return o;
}

// Quadrado (vi) que cobre a cápsula a–b de raio r, em modo tubo.
fn capsule_vertex(vi: u32, a_w: vec2<f32>, b_w: vec2<f32>, r: f32, col: vec3<f32>) -> AgentVsOut {
    var o: AgentVsOut;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let seg = b_w - a_w;
    let l = length(seg);
    let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
    let nn = vec2<f32>(-e.y, e.x);
    let along = select(-r, l + r, c.x > 0.0);
    let w = a_w + e * along + nn * (c.y * r);
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.mode = 0u;
    o.local = w - a_w;
    o.tangent = seg;
    o.core_phase = vec2<f32>(r, 0.0);
    o.organ = NO_ORGAN;
    o.color = col;
    return o;
}

// Base j das pontas: j < RNA_PER_END = 5' UTR (antes do AUG, a sair da
// ponta N); senão 3' UTR (depois do stop, a sair da ponta C). RNA nu: o
// genoma todo, a partir do centro. O fio ondula devagar com a idade.
fn residue_world_v(slot: u32, a: Agent, k: u32) -> vec2<f32> {
    let lp = body_pos_view[slot * MAX_BODY_V + k];
    let cr = cos(a.rot);
    let sr = sin(a.rot);
    return vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
}

// LIGAÇÃO i do agente: tubo fino entre os dois resíduos. Só a
// desenha o lado de slot menor (o outro tem a mesma ligação ao contrário).
fn bond_vertex(vi: u32, slot: u32, a: Agent, i: u32) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let b = bonds_view[slot * BOND_STRIDE_V + i];
    if (a.alive == 0u || b.x == 0xFFFFFFFFu || b.x <= slot) { return o; }
    let other = agents_view[b.x];
    if (other.alive == 0u || other.id != b.y) { return o; }
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot && b.x != view.focus_slot;
    let mine = b.z & 0xFFu;
    let theirs = (b.z >> 8u) & 0xFFu;
    if (hidden || mine >= a.body_len || theirs >= other.body_len) { return o; }
    let p0 = residue_world_v(slot, a, mine);
    let p1 = residue_world_v(b.x, other, theirs);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let r_tube = max(1.2, 1.0 / view.zoom);
    let seg = p1 - p0;
    let l = length(seg);
    let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
    let nn = vec2<f32>(-e.y, e.x);
    let along = select(-r_tube, l + r_tube, c.x > 0.0);
    let w = p0 + e * along + nn * (c.y * r_tube);
    let px = vec2<f32>((w.x - view.center_x) * view.zoom, (w.y - view.center_y) * view.zoom);
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.mode = 0u;
    o.local = w - p0;
    o.tangent = seg;
    o.core_phase = vec2<f32>(r_tube, 0.0);
    o.organ = NO_ORGAN;
    // Dourado = ligada ao tocar; azul-claro = de nascimento (pai e filho).
    o.color = select(vec3<f32>(1.0, 0.82, 0.35), vec3<f32>(0.45, 0.85, 1.0), (b.z >> 16u) != 0u);
    return o;
}

// PARENTESCO (vista 4): bola por cima do agente, do mesmo tamanho no ecrã
// para todos. Verde = genoma próximo do selecionado, amarelo, vermelho = distante.
fn kin_vertex(vi: u32, slot: u32, a: Agent) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    // MARCAR ÓRGÃO: bola ciano por cima de quem tem o tipo escolhido (a mesma
    // bola do parentesco; tem prioridade sobre ele).
    if (view.mark_organ != 0u) {
        if (a.alive == 0u) { return o; }
        var has = false;
        for (var k = 0u; k < min(a.body_len, MAX_BODY_V); k++) {
            let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
            if (oc != 0u && (oc & 0x1Fu) == view.mark_organ) {
                has = true;
                break;
            }
        }
        if (!has) { return o; }
        let cm = vec2<f32>(a.pos_x, a.pos_y);
        return capsule_vertex(vi, cm, cm, KIN_DOT_PX / view.zoom, vec3<f32>(0.1, 0.95, 1.0));
    }
    let q = kin_view[slot];
    if (view.signal_view != 4u || a.alive == 0u || q < 0.0) { return o; }
    // Raiz quadrada: mais resolução nos parentescos fracos (os clãs
    // distantes distinguem-se uns dos outros): 25% de 8-meros -> meio da escala.
    let t = sqrt(clamp(q, 0.0, 1.0));
    let col = select(mix(vec3<f32>(1.0, 0.85, 0.1), vec3<f32>(0.15, 1.0, 0.25), (t - 0.5) * 2.0),
                     mix(vec3<f32>(1.0, 0.12, 0.08), vec3<f32>(1.0, 0.85, 0.1), t * 2.0), t < 0.5);
    let c = vec2<f32>(a.pos_x, a.pos_y);
    return capsule_vertex(vi, c, c, KIN_DOT_PX / view.zoom, col);
}

fn rna_vertex(vi: u32, slot: u32, a: Agent, j: u32) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot;
    if (a.alive == 0u || hidden) { return o; }
    let trailer = j >= RNA_PER_END;
    let m = j % RNA_PER_END;
    let start = a.coding_span & 0xFFFFu;
    let after = a.coding_span >> 16u;
    let n = a.body_len;
    var base_i = 0u;
    var anchor = vec2<f32>(0.0);
    var dir = vec2<f32>(select(-1.0, 1.0, trailer), 0.0);
    if (n == 0u) {
        // RNA nu: as duas metades do genoma, em sentidos opostos.
        base_i = select(m, RNA_PER_END + m, trailer);
        if (base_i >= a.gene_len) { return o; }
    } else if (!trailer) {
        if (m >= start) { return o; }
        base_i = start - 1u - m;
        anchor = body_pos_view[slot * MAX_BODY_V];
        if (n > 1u) { dir = normalize(anchor - body_pos_view[slot * MAX_BODY_V + 1u] + vec2<f32>(1e-6, 0.0)); }
    } else {
        base_i = after + m;
        if (base_i >= a.gene_len) { return o; }
        anchor = body_pos_view[slot * MAX_BODY_V + n - 1u];
        if (n > 1u) { dir = normalize(anchor - body_pos_view[slot * MAX_BODY_V + n - 2u] + vec2<f32>(1e-6, 0.0)); }
    }
    // Caminha base a base até m: ondula devagar (a flutuar) e curva com a
    // curvatura do fio, que vem do movimento da ponta (fica para trás).
    var ang = atan2(dir.y, dir.x);
    let bend_state = rna_tail_view[slot * 2u + 1u];
    let bend = select(bend_state.x, bend_state.y, trailer) / f32(RNA_PER_END);
    var p = anchor;
    var prev = anchor;
    let seed = f32(a.id % 977u) * 0.37 + select(0.0, 2.1, trailer);
    for (var t = 0u; t <= m; t++) {
        ang += RNA_WIGGLE * sin(f32(t) * 0.7 + f32(a.age) * 0.03 + seed) + bend;
        prev = p;
        p += vec2<f32>(cos(ang), sin(ang)) * RNA_SPACING;
    }
    // Para o mundo.
    let cr = cos(a.rot);
    let sr = sin(a.rot);
    let c0 = vec2<f32>(a.pos_x, a.pos_y);
    let pw = c0 + vec2<f32>(cr * p.x - sr * p.y, sr * p.x + cr * p.y);
    let qw = c0 + vec2<f32>(cr * prev.x - sr * prev.y, sr * prev.x + cr * prev.y);
    return capsule_vertex(vi, qw, pw, max(RNA_RADIUS, 1.0 / view.zoom), base_color(genome_base(slot, base_i)) * 0.85);
}

// Distância de p ao segmento a–b.
fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * h);
}

// Antena do centro até `tip` (com um botão na ponta): 1 dentro, 0 fora.
fn antenna(p: vec2<f32>, tip: vec2<f32>, core: f32) -> f32 {
    let stalk = seg_dist(p, vec2<f32>(0.0), tip) < 0.05;
    let knob = length(p - tip) < 0.11;
    return select(0.0, 1.0, (stalk && length(p) > core * 0.8) || knob);
}

@fragment
fn fs_agent(in: AgentVsOut) -> @location(0) vec4<f32> {
    if (in.mode == 0u) {
        // TUBO: cápsula com sombreado de cilindro (centro claro, bordas escuras).
        let r_t = in.core_phase.x;
        let dd = seg_dist(in.local, vec2<f32>(0.0), in.tangent);
        if (dd > r_t) { discard; }
        let x = dd / r_t;
        let shade = sqrt(max(1.0 - x * x, 0.0));
        let c = in.color * (0.35 + 0.65 * shade) + vec3<f32>(0.18) * pow(shade, 8.0);
        return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    let p = in.local;
    let d = length(p);
    let core = in.core_phase.x;
    let t = in.tangent;
    let nrm = vec2<f32>(-t.y, t.x);
    // Coordenadas ao longo (u) e através (v) da cadeia.
    let u = dot(p, t);
    let v = dot(p, nrm);
    let white = vec3<f32>(1.0);
    let rim_mix = smoothstep(core * 0.6, core * 0.8, d);

    switch in.organ {
        case NO_ORGAN: {
            if (d > 1.0) { discard; }
            let rim = smoothstep(0.75, 0.98, d);
            return vec4<f32>(mix(in.color, vec3<f32>(0.0), rim * 0.7), 1.0);
        }
        case ORGAN_MOUTH: {
            if (d > core) { discard; }
            // Abertura em cunha, virada para o lado esquerdo da cadeia.
            if (v > 0.15 * core && abs(u) < v * 0.9) { return vec4<f32>(0.05, 0.02, 0.02, 1.0); }
            return vec4<f32>(mix(in.color, white, rim_mix), 1.0);
        }
        case ORGAN_MUSCLE: {
            let e = (u / (core * 1.5)) * (u / (core * 1.5)) + (v / (core * 0.85)) * (v / (core * 0.85));
            if (e > 1.0) { discard; }
            let stripe = step(0.0, sin(u / core * 12.0));
            let red = mix(in.color, vec3<f32>(0.85, 0.25, 0.25), 0.55);
            return vec4<f32>(mix(red, red * 0.6, stripe) * mix(1.0, 0.7, smoothstep(0.7, 1.0, e)), 1.0);
        }
        case ORGAN_FOOD_SENSOR, ORGAN_LIGHT_SENSOR, ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: {
            let is_light = in.organ == ORGAN_LIGHT_SENSOR || in.organ == ORGAN_LIGHT_SENSOR_DIR;
            let ant_col = select(vec3<f32>(0.45, 1.0, 0.45), vec3<f32>(1.0, 0.95, 0.4), is_light);
            if (d < core) { return vec4<f32>(mix(in.color, ant_col, rim_mix), 1.0); }
            var hit = 0.0;
            if (in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR) {
                // Duas antenas, uma de cada lado, ligeiramente inclinadas para a frente.
                hit = max(antenna(p, nrm * 0.85 + t * 0.3, core), antenna(p, -nrm * 0.85 + t * 0.3, core));
            } else {
                // Coroa de 6 antenas curtas (amostra à volta toda).
                for (var i = 0u; i < 6u; i++) {
                    let ang = f32(i) * 1.0471976;
                    let dir = t * cos(ang) + nrm * sin(ang);
                    hit = max(hit, antenna(p, dir * 0.82, core));
                }
            }
            if (hit < 0.5) { discard; }
            return vec4<f32>(ant_col, 1.0);
        }
        case ORGAN_ENERGY_SENSOR: {
            if (d > core) { discard; }
            let ring = abs(d - core * 0.45) < core * 0.12;
            return vec4<f32>(select(mix(in.color, white, rim_mix), vec3<f32>(1.0, 0.85, 0.2), ring), 1.0);
        }
        case ORGAN_CLOCK: {
            if (d > core) { discard; }
            let hand_dir = t * cos(in.core_phase.y) + nrm * sin(in.core_phase.y);
            if (seg_dist(p, vec2<f32>(0.0), hand_dir * core * 0.8) < core * 0.12) {
                return vec4<f32>(0.05, 0.05, 0.1, 1.0);
            }
            return vec4<f32>(mix(vec3<f32>(0.85, 0.9, 1.0), white, rim_mix), 1.0);
        }
        case ORGAN_RELAY: {
            if (abs(u) + abs(v) > core * 1.25) { discard; }
            let edge = smoothstep(core * 0.85, core * 1.1, abs(u) + abs(v));
            return vec4<f32>(mix(in.color, vec3<f32>(0.6, 0.9, 1.0), edge), 1.0);
        }
        case ORGAN_PHOTOSYSTEM: {
            // Disco verde com 8 raios curtos (capta luz).
            let ang = atan2(v, u);
            let ray = d < 0.95 && d > core && abs(sin(ang * 4.0)) < 0.25;
            if (d > core && !ray) { discard; }
            let leaf = vec3<f32>(0.35, 0.95, 0.35);
            return vec4<f32>(select(mix(leaf * 0.7, leaf, 1.0 - d / core), vec3<f32>(0.85, 1.0, 0.4), ray), 1.0);
        }
        case ORGAN_CHEMO: {
            // Disco amarelo-enxofre com pintas escuras (consome o redutor).
            if (d > core) { discard; }
            let spot = step(0.6, fract(u * 3.7 / core) * fract(v * 3.1 / core) * 2.5);
            return vec4<f32>(mix(vec3<f32>(0.9, 0.78, 0.15), vec3<f32>(0.35, 0.28, 0.05), spot), 1.0);
        }
        case ORGAN_BIAS: {
            // Ponto cheio (laranja = α, verde = β) com um contorno claro.
            if (d > core) { discard; }
            return vec4<f32>(mix(in.color, vec3<f32>(1.0), smoothstep(core * 0.75, core, d)), 1.0);
        }
        case ORGAN_ANCHOR: {
            // Anel grosso (vermelho = +, azul = −) com o centro escuro.
            if (d > core * 1.15 || d < core * 0.5) { discard; }
            let edge = smoothstep(core * 0.95, core * 1.15, d);
            return vec4<f32>(mix(in.color, in.color * 0.4, edge), 1.0);
        }
        case ORGAN_PROTEASE: {
            // Disco com dentes (6 triângulos à volta).
            let ang = atan2(v, u);
            let tooth = core + (0.95 - core) * max(0.0, 1.0 - abs(fract(ang / 1.0471976) - 0.5) * 4.0);
            if (d > tooth) { discard; }
            return vec4<f32>(select(in.color, mix(in.color, vec3<f32>(1.0), 0.8), d > core), 1.0);
        }
        default: {
            // Armazenamento: disco com anéis concêntricos.
            if (d > core) { discard; }
            let rings = step(0.5, fract(d / core * 3.0));
            return vec4<f32>(mix(in.color, vec3<f32>(0.15, 0.1, 0.0), rings * 0.7), 1.0);
        }
    }
}
