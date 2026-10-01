// Grupo 0: frame. Grupo 1: mundo.
@group(0) @binding(0) var<uniform> params: SimParams;

// Grelha de monómeros: 4 u32 atómicos por célula (idx*4 + canal; 0=A 1=U 2=G 3=C).
// 16 bits baixos = ATIVADOS, 16 bits altos = GASTOS.
@group(1) @binding(0) var<storage, read_write> chem_grid: array<atomic<u32>>;

// Livro-razão: [ativados A U G C, gastos A U G C].
@group(1) @binding(1) var<storage, read_write> ledger: array<atomic<u32>, 8>;
