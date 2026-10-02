// OBSERVAÇÃO (só para quem vê; a simulação não usa nada disto).
//
// stats_reduce: estatísticas da população numa redução atómica (lidas de
// forma assíncrona de N em N epochs).
// kinship: semelhança genética de cada agente com um genoma escolhido (o do
// agente selecionado), pela fração de 8-meros partilhados. Os 8-meros são
// CANÓNICOS (o menor entre o k-mero e o seu complementar invertido): o filho
// é o complementar invertido do pai e tem de contar como parente.

// Palavras de stats_out (ver src/stats.rs, STAT_*):
const ST_ALIVE: u32 = 0u;
const ST_ENERGY10: u32 = 1u;   // Σ energia × 10
const ST_BODY: u32 = 2u;       // Σ resíduos do corpo
const ST_GENE: u32 = 3u;       // Σ bases do genoma
const ST_GEN_MAX: u32 = 4u;
const ST_GEN_SUM: u32 = 5u;
const ST_BONDED: u32 = 6u;     // agentes com pelo menos uma ligação
const ST_ORGANS: u32 = 7u;     // Σ órgãos
const ST_HAS_ORGAN: u32 = 8u;  // + tipo: agentes com pelo menos um órgão desse tipo

@compute @workgroup_size(64)
fn stats_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    if (a.alive == 0u) { return; }
    atomicAdd(&stats_out[ST_ALIVE], 1u);
    atomicAdd(&stats_out[ST_ENERGY10], u32(max(a.energy, 0.0) * 10.0));
    atomicAdd(&stats_out[ST_BODY], a.body_len);
    atomicAdd(&stats_out[ST_GENE], a.gene_len);
    atomicMax(&stats_out[ST_GEN_MAX], a.generation);
    atomicAdd(&stats_out[ST_GEN_SUM], a.generation);
    var has = 0u;
    var count = 0u;
    for (var k = 0u; k < a.body_len; k++) {
        let t = organ_type(organ_get(slot, k));
        if (t != 0xFFu) {
            has |= 1u << t;
            count += 1u;
        }
    }
    atomicAdd(&stats_out[ST_ORGANS], count);
    for (var t = 0u; t < ORGAN_TYPES; t++) {
        if ((has & (1u << t)) != 0u) { atomicAdd(&stats_out[ST_HAS_ORGAN + t], 1u); }
    }
    for (var i = 0u; i < MAX_BONDS; i++) {
        if (bond_at(slot, i).x != BOND_NONE) {
            atomicAdd(&stats_out[ST_BONDED], 1u);
            break;
        }
    }
}

const KMER: u32 = 8u;
const KMER_MASK: u32 = 0xFFFFu;

// Complementar invertido de um 8-mero (2 bits por base, A=0 U=1 G=2 C=3:
// o complementar é base ^ 1).
fn kmer_revcomp(k: u32) -> u32 {
    var r = 0u;
    var x = k ^ 0x5555u;
    for (var i = 0u; i < KMER; i++) {
        r = (r << 2u) | (x & 3u);
        x >>= 2u;
    }
    return r;
}

// kin_target: [n, k-meros canónicos ORDENADOS e únicos do genoma escolhido...]
fn kin_has(k: u32) -> bool {
    var lo = 1u;
    var hi = kin_target[0] + 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let v = kin_target[mid];
        if (v == k) { return true; }
        if (v < k) { lo = mid + 1u; } else { hi = mid; }
    }
    return false;
}

@compute @workgroup_size(64)
fn kinship(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.max_agents) { return; }
    let a = agents[slot];
    let n_t = kin_target[0];
    if (a.alive == 0u || a.gene_len < KMER || n_t == 0u) {
        kin_out[slot] = -1.0;
        return;
    }
    var k = 0u;
    var hits = 0u;
    for (var i = 0u; i < a.gene_len; i++) {
        k = ((k << 2u) | genome_get(slot, i)) & KMER_MASK;
        if (i + 1u < KMER) { continue; }
        if (kin_has(min(k, kmer_revcomp(k)))) { hits += 1u; }
    }
    let n_a = a.gene_len - KMER + 1u;
    kin_out[slot] = f32(hits) / f32(max(n_a, n_t));
}
