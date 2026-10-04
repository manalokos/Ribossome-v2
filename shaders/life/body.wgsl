// TRADUÇÃO E CORPO.
// O corpo é a cadeia de aminoácidos traduzida a partir da primeira base (ou
// do primeiro AUG, se params.require_start), com o código genético padrão,
// até ao primeiro stop ou 64 resíduos. A tabela
// CODON_TABLE e as propriedades AA_* vêm de src/life/amino.rs (dados reais).
//
// Geometria: cadeia principal de comprimento constante; cada junta tem um
// ângulo de repouso COM SINAL, próprio do aminoácido (tabela do v3; a
// homoquiralidade, com todas as dobras para o mesmo lado, foi abandonada a
// pedido do Filipe).
// As posições locais (centradas no centro de massa) ficam em body_pos.

const MAX_BODY: u32 = 64u;
// Comprimento de referência de um segmento (o do v3, unidades do mundo); cada
// resíduo tem o seu (residue_len).
const SEGMENT_LEN: f32 = 11.0;

fn body_get(slot: u32, i: u32) -> u32 {
    return (bodies[slot * 16u + i / 4u] >> ((i % 4u) * 8u)) & 0xFFu;
}

// Massa de um resíduo nas unidades do v3 (0,02 por 118 Da).
// Massa e comprimento do resíduo k: os do aminoácido (tabela) × os do órgão,
// se o resíduo for um órgão.
fn residue_mass(slot: u32, k: u32) -> f32 {
    var m = 0.02 * aa_props[body_get(slot, k)].mass / 118.0;
    let o = organ_get(slot, k);
    if (o != 0u) { m *= organ_cost(o).mass_mult; }
    return m;
}

// Massa de um resíduo médio (118 Da) e de um corpo médio (16 resíduos).
const MASS_RESIDUE_REF: f32 = 0.02;
const MASS_BODY_REF: f32 = 0.32;

// Massa do corpo (resíduos × órgãos).
fn body_mass(slot: u32, n: u32) -> f32 {
    var m = 0.0;
    for (var k = 0u; k < n; k++) { m += residue_mass(slot, k); }
    return max(m, MASS_RESIDUE_REF);
}

fn residue_len(slot: u32, k: u32) -> f32 {
    var l = aa_props[body_get(slot, k)].seg_len;
    let o = organ_get(slot, k);
    if (o != 0u) { l *= organ_cost(o).len_mult; }
    return max(l, 1.0);
}

// Ângulo de repouso da junta (com sinal; tabela do v3 em src/life/amino.rs).
fn residue_bend(aa: u32) -> f32 {
    return aa_props[aa].rest_angle;
}

// Traduz o genoma do slot, escreve bodies e body_pos e devolve o nº de
// resíduos. Em `span` escreve a zona traduzida: início (AUG) | (primeira
// base depois do stop) << 16.
fn translate_agent(slot: u32, gene_len: u32, span: ptr<function, u32>) -> u32 {
    // Sem AUG obrigatório (por omissão), lê-se a partir da primeira base.
    var start = select(0xFFFFFFFFu, 0u, params.require_start == 0u);
    for (var i = 0u; i + 2u < gene_len && start == 0xFFFFFFFFu; i++) {
        if (genome_get(slot, i) == 0u && genome_get(slot, i + 1u) == 1u && genome_get(slot, i + 2u) == 2u) {
            start = i;
            break;
        }
    }
    for (var w = 0u; w < 16u; w++) { bodies[slot * 16u + w] = 0u; }
    for (var w = 0u; w < 32u; w++) { organs[slot * 32u + w] = 0u; }
    if (start == 0xFFFFFFFFu) {
        // Sem AUG: o genoma todo fica por traduzir.
        *span = gene_len | (gene_len << 16u);
        return 0u;
    }
    var n = 0u;
    var i = start;
    loop {
        if (i + 2u >= gene_len || n >= MAX_BODY) { break; }
        let c = genome_get(slot, i) * 16u + genome_get(slot, i + 1u) * 4u + genome_get(slot, i + 2u);
        let aa = CODON_TABLE[c];
        if (aa == AA_STOP) {
            i += 3u; // o stop faz parte da zona lida
            break;
        }
        bodies[slot * 16u + n / 4u] |= aa << ((n % 4u) * 8u);
        // ÓRGÃO: promotor + modificador (aminoácidos) com entrada no código
        // dos órgãos (assets/codigo_orgaos.json): 6 bases, ou 9 com a intensidade.
        var step = 3u;
        if (i + 5u < gene_len) {
            let m = genome_get(slot, i + 3u) * 16u + genome_get(slot, i + 4u) * 4u + genome_get(slot, i + 5u);
            let maa = CODON_TABLE[m];
            var c = 0u;
            if (maa != AA_STOP) { c = organ_code[aa * 20u + maa]; }
            if (c != 0u) {
                // Segundo modificador: intensidade (senão 32 = ganho 1, 6 bases).
                var gain = 32u;
                step = 6u;
                if (i + 8u < gene_len) {
                    let m2 = genome_get(slot, i + 6u) * 16u + genome_get(slot, i + 7u) * 4u + genome_get(slot, i + 8u);
                    if (CODON_TABLE[m2] != AA_STOP) {
                        gain = m2;
                        step = 9u;
                    }
                }
                organs[slot * 32u + n / 2u] |= (c | (gain << 8u)) << ((n % 2u) * 16u);
            }
        }
        n += 1u;
        i += step;
    }
    *span = start | (min(i, gene_len) << 16u);
    // Geometria inicial: dobras homoquirais pela tendência local; a
    // dobragem (fold.wgsl) parte daqui nos primeiros passos de vida.
    for (var k = 0u; k < n; k++) {
        // O órgão pode ter o seu ângulo de repouso (tabela dos órgãos, v3).
        var b = residue_bend(body_get(slot, k));
        let ob = organ_get(slot, k);
        if (ob != 0u) {
            let oa = organ_cost(ob).rest_angle;
            if (oa < 1e8) { b = oa; }
        }
        joint_angle[slot * MAX_BODY + k] = b;
        joint_base[slot * MAX_BODY + k] = b;
        joint_state[slot * MAX_BODY + k] = 0u;
        joint_active[slot * MAX_BODY + k] = 0.0;
        signals[slot * MAX_BODY + k] = vec4<f32>(0.0);
        sensor_mem[slot * MAX_BODY + k] = 0.0;
        sensor_avg[slot * MAX_BODY + k] = SENSOR_UNSET;
    }
    rebuild_body(slot, n);
    return n;
}

// Raio de contacto: raio de giração dos resíduos (mais a espessura de um
// resíduo), limitado a CONTACT_R_MAX; RNA nu = CONTACT_R_NAKED.
fn contact_radius(slot: u32, n: u32) -> f32 {
    if (n == 0u) { return CONTACT_R_NAKED; }
    var s = 0.0;
    for (var k = 0u; k < n; k++) {
        let q = body_pos[slot * MAX_BODY + k];
        s += dot(q, q);
    }
    return min(sqrt(s / f32(n)) + 4.0, CONTACT_R_MAX);
}

fn rotate(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Posição no MUNDO do resíduo k.
fn residue_world(slot: u32, a: Agent, k: u32) -> vec2<f32> {
    return vec2<f32>(a.pos_x, a.pos_y) + rotate(body_pos[slot * MAX_BODY + k], a.rot);
}

// SOMBRA DOS AGENTES: cada resíduo conta na célula da luz onde está (antes
// de cada cálculo da luz; clear_shade limpa a grelha).
@compute @workgroup_size(64)
fn agents_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    let lw = f32(WORLD_UNITS_PER_CELL * LIGHT_DIV);
    for (var k = 0u; k < a.body_len; k++) {
        // Só absorvem UV os aromáticos (absortividade a 280 nm relativa ao
        // triptofano: W 1, Y 0,27, F 0,04) e os fotossistemas (captam luz).
        // (absorcao_uv da tabela dos aminoácidos.)
        var w = u32(max(aa_props[body_get(slot, k)].uv_absorb, 0.0) * f32(SHADE_ONE) + 0.5);
        if (organ_type(organ_get(slot, k)) == ORGAN_PHOTOSYSTEM) { w = max(w, SHADE_ONE); }
        if (w == 0u) { continue; }
        let p = residue_world(slot, a, k);
        let c = clamp(vec2<i32>(floor(p / lw)), vec2<i32>(0), vec2<i32>(i32(LIGHT_SIZE) - 1));
        atomicAdd(&shade_grid[u32(c.y) * LIGHT_SIZE + u32(c.x)], w);
    }
}

// Ponto fixo da sombra dos agentes (um "triptofano equivalente").
const SHADE_ONE: u32 = 100u;
