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
const SEGMENT_LEN: f32 = 11.0;      // unidades do mundo (v3)

fn body_get(slot: u32, i: u32) -> u32 {
    return (bodies[slot * 16u + i / 4u] >> ((i % 4u) * 8u)) & 0xFFu;
}

// Massa de um resíduo nas unidades do v3 (0,02 por 118 Da).
fn residue_mass(aa: u32) -> f32 {
    return 0.02 * AA_MASS[aa] / 118.0;
}

// Ângulo de repouso da junta (com sinal; tabela do v3 em src/life/amino.rs).
fn residue_bend(aa: u32) -> f32 {
    return AA_REST_ANGLE[aa];
}

// Traduz o genoma do slot, escreve bodies e body_pos e devolve o nº de resíduos.
fn translate_agent(slot: u32, gene_len: u32) -> u32 {
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
    if (start == 0xFFFFFFFFu) { return 0u; }
    var n = 0u;
    var i = start;
    loop {
        if (i + 2u >= gene_len || n >= MAX_BODY) { break; }
        let c = genome_get(slot, i) * 16u + genome_get(slot, i + 1u) * 4u + genome_get(slot, i + 2u);
        let aa = CODON_TABLE[c];
        if (aa == AA_STOP) { break; }
        bodies[slot * 16u + n / 4u] |= aa << ((n % 4u) * 8u);
        // ÓRGÃO: promotor seguido de um modificador que não é stop (6 bases).
        var step = 3u;
        if (AA_IS_PROMOTER[aa] != 0u && i + 5u < gene_len) {
            let m = genome_get(slot, i + 3u) * 16u + genome_get(slot, i + 4u) * 4u + genome_get(slot, i + 5u);
            if (CODON_TABLE[m] != AA_STOP) {
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
                let ob = ((m % ORGAN_TYPES) + 1u) | ((m / ORGAN_TYPES) << 4u) | (gain << 8u);
                organs[slot * 32u + n / 2u] |= ob << ((n % 2u) * 16u);
            }
        }
        n += 1u;
        i += step;
    }
    // Geometria inicial: dobras homoquirais pela tendência local; a
    // dobragem (fold.wgsl) parte daqui nos primeiros passos de vida.
    for (var k = 0u; k < n; k++) {
        let b = residue_bend(body_get(slot, k));
        joint_angle[slot * MAX_BODY + k] = b;
        joint_base[slot * MAX_BODY + k] = b;
        joint_state[slot * MAX_BODY + k] = 0u;
        joint_active[slot * MAX_BODY + k] = 0.0;
        signals[slot * MAX_BODY + k] = vec2<f32>(0.0);
        sensor_mem[slot * MAX_BODY + k] = 0.0;
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
