// ---- Grupo 3: organismos ----
// Agentes em SLOTS FIXOS (nunca há compactação): o despacho cobre sempre a
// capacidade inteira e um slot livre tem alive = 0. Slots livres numa pilha.
@group(3) @binding(0) var<storage, read_write> agents: array<Agent>;
// Genoma: 16 u32 por slot, 2 bits por base (A=0 U=1 G=2 C=3), base 0 primeiro.
@group(3) @binding(1) var<storage, read_write> genomes: array<u32>;
// Pilha de slots livres; o topo está em life_counters[LC_FREE_TOP].
@group(3) @binding(2) var<storage, read_write> free_slots: array<u32>;
@group(3) @binding(3) var<storage, read_write> life_counters: array<atomic<u32>, 8>;
// Pedidos de sementes (geração 0) escritos pelo CPU.
@group(3) @binding(4) var<storage, read> spawn_requests: array<SpawnRequest>;
// Corpo traduzido: 16 u32 por slot, 4 resíduos (índices em AMINO) por u32.
@group(3) @binding(5) var<storage, read_write> bodies: array<u32>;

const LC_FREE_TOP: u32 = 0u;
const LC_NEXT_ID: u32 = 1u;
const LC_SPAWNED: u32 = 2u;
const LC_SPAWN_FAILED: u32 = 3u;
const LC_DEATHS: u32 = 4u;
const LC_BIRTHS: u32 = 5u;

const GENOME_WORDS: u32 = 16u;
const MAX_GENE_LEN: u32 = 256u;
