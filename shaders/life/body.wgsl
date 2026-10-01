// TRADUÇÃO E CORPO (fase 3: corpo rígido).
// O corpo é a cadeia de aminoácidos traduzida a partir do primeiro AUG, com
// o código genético padrão, até ao primeiro stop ou 64 resíduos. A tabela
// CODON_TABLE e as propriedades AA_* vêm de src/life/amino.rs (dados reais).
//
// Geometria (v3): cadeia principal de comprimento constante; em cada junta
// a cadeia dobra sempre para o MESMO lado (homoquiralidade em 2D) com um
// ângulo da tendência de volta/hélice de Chou-Fasman:
//   dobra = clamp(0,55·(Pt − 0,6) + 0,25·max(Pa − 1, 0), 0, 0,6) rad.
// As posições locais (centradas no centro de massa) ficam em body_pos.

const MAX_BODY: u32 = 64u;
const SEGMENT_LEN: f32 = 11.0;      // unidades do mundo (v3)

fn body_get(slot: u32, i: u32) -> u32 {
    return (bodies[slot * 16u + i / 4u] >> ((i % 4u) * 8u)) & 0xFFu;
}

// Massa de um resíduo nas unidades do v3 (0,02 por 118 Da).
fn residue_mass(aa: u32) -> f32 {
    var m = AA_MASS;
    return 0.02 * m[aa] / 118.0;
}

fn residue_bend(aa: u32) -> f32 {
    var pt = AA_P_TURN;
    var pa = AA_P_HELIX;
    return clamp(0.55 * (pt[aa] - 0.6) + 0.25 * max(pa[aa] - 1.0, 0.0), 0.0, 0.6);
}

// Traduz o genoma do slot, escreve bodies e body_pos e devolve o nº de resíduos.
fn translate_agent(slot: u32, gene_len: u32) -> u32 {
    var start = 0xFFFFFFFFu;
    for (var i = 0u; i + 2u < gene_len; i++) {
        if (genome_get(slot, i) == 0u && genome_get(slot, i + 1u) == 1u && genome_get(slot, i + 2u) == 2u) {
            start = i;
            break;
        }
    }
    for (var w = 0u; w < 16u; w++) { bodies[slot * 16u + w] = 0u; }
    if (start == 0xFFFFFFFFu) { return 0u; }
    var codons = CODON_TABLE;
    var n = 0u;
    var i = start;
    loop {
        if (i + 2u >= gene_len || n >= MAX_BODY) { break; }
        let c = genome_get(slot, i) * 16u + genome_get(slot, i + 1u) * 4u + genome_get(slot, i + 2u);
        let aa = codons[c];
        if (aa == AA_STOP) { break; }
        bodies[slot * 16u + n / 4u] |= aa << ((n % 4u) * 8u);
        n += 1u;
        i += 3u;
    }
    // Geometria: anda a cadeia com dobras homoquirais e centra no centro de massa.
    var p = vec2<f32>(0.0);
    var ang = 0.0;
    var com = vec2<f32>(0.0);
    var mass = 0.0;
    for (var k = 0u; k < n; k++) {
        let aa = body_get(slot, k);
        body_pos[slot * MAX_BODY + k] = p;
        let m = residue_mass(aa);
        com += p * m;
        mass += m;
        ang += residue_bend(aa);
        p += vec2<f32>(cos(ang), sin(ang)) * SEGMENT_LEN;
    }
    com /= max(mass, 1e-6);
    for (var k = 0u; k < n; k++) {
        body_pos[slot * MAX_BODY + k] -= com;
    }
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
