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
// Posições locais dos resíduos (centradas no centro de massa), MAX_BODY por slot.
@group(3) @binding(6) var<storage, read_write> body_pos: array<vec2<f32>>;
// Lista de slots vivos para o desenho e argumentos do draw indireto
// [vertex_count, instance_count, first_vertex, first_instance].
@group(3) @binding(7) var<storage, read_write> draw_list: array<u32>;
@group(3) @binding(8) var<storage, read_write> draw_args: array<atomic<u32>, 4>;
// Grelha de contacto: cabeça da lista por célula do ambiente, ligação por
// resíduo (slot·64 + k) e deslocamento calculado por agente (dx, dy, dθ, _).
@group(3) @binding(9) var<storage, read_write> contact_head: array<atomic<u32>>;
@group(3) @binding(10) var<storage, read_write> contact_next: array<u32>;
@group(3) @binding(11) var<storage, read_write> contact_disp: array<vec4<f32>>;
// Ângulos das juntas (atual e base), MAX_BODY por slot.
@group(3) @binding(12) var<storage, read_write> joint_angle: array<f32>;
@group(3) @binding(13) var<storage, read_write> joint_base: array<f32>;
// Estado do ciclo catalítico de cada resíduo: 0 livre, 1 ligado, 2 produto.
@group(3) @binding(14) var<storage, read_write> joint_state: array<u32>;
// Deformação ATIVA de cada junta (a que vem do ciclo catalítico, propagada).
@group(3) @binding(15) var<storage, read_write> joint_active: array<f32>;
// Órgãos: 16 bits por resíduo (0 = nenhum; (tipo + 1) | (parâmetro << 4) |
// (intensidade << 8)), 32 u32 por slot.
@group(3) @binding(16) var<storage, read_write> organs: array<u32>;
// Sinais internos (α, β) por resíduo.
@group(3) @binding(17) var<storage, read_write> signals: array<vec2<f32>>;
// Memória dos sensores (o valor sentido no passo anterior), por resíduo.
@group(3) @binding(18) var<storage, read_write> sensor_mem: array<f32>;
// Energia tirada a cada agente por proteases de outros neste passo
// (milésimos, somas atómicas: ordem indiferente).
@group(3) @binding(19) var<storage, read_write> bitten: array<atomic<u32>>;

const LC_FREE_TOP: u32 = 0u;
const LC_NEXT_ID: u32 = 1u;
const LC_SPAWNED: u32 = 2u;
const LC_SPAWN_FAILED: u32 = 3u;
const LC_DEATHS: u32 = 4u;
const LC_BIRTHS: u32 = 5u;
// Mortes com energia < 1 (fome).
const LC_STARVED: u32 = 6u;
// Mordidas de protease (contactos em que um agente tirou energia a outro).
const LC_BITES: u32 = 7u;

const GENOME_WORDS: u32 = 16u;
const MAX_GENE_LEN: u32 = 256u;
