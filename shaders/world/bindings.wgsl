// Grupo 0: frame. Grupo 1: mundo (resolução do ambiente). Grupo 2: fluido.
@group(0) @binding(0) var<uniform> params: SimParams;

// ---- Grupo 1: mundo ----
// Grelha de monómeros: 4 u32 atómicos por célula (idx*4 + canal; 0=A 1=U 2=G 3=C).
// 16 bits baixos = ATIVADOS, 16 bits altos = GASTOS.
@group(1) @binding(0) var<storage, read_write> chem_grid: array<atomic<u32>>;
// Livro-razão: [ativados A U G C, gastos A U G C, presos em agentes A U G C].
@group(1) @binding(1) var<storage, read_write> ledger: array<atomic<u32>, 12>;
// Terreno: quanta inteiros de gamma por célula (rocha >= GAMMA_SOLID_THRESHOLD).
@group(1) @binding(2) var<storage, read_write> gamma_grid: array<atomic<u32>>;
// Luz UV por célula (0..1), recalculada de tempos a tempos.
// Luz UV a LIGHT_SIZE² (1/LIGHT_DIV da grelha): ler com uv_light_at_cell.
@group(1) @binding(3) var<storage, read_write> light_grid: array<f32>;
// Declive do terreno por unidade do mundo (recalculado a cada passo).
@group(1) @binding(4) var<storage, read_write> slope_grid: array<vec2<f32>>;
// Grelha de destino do transporte (mesmo formato que chem_grid; zero entre passos).
@group(1) @binding(5) var<storage, read_write> chem_next: array<atomic<u32>>;

// ---- Grupo 2: fluido (resolução FLUID_SIZE) ----
// Velocidade em células do fluido por segundo. Pares in/out alternam por bind group.
@group(2) @binding(0) var<storage, read> velocity_in: array<vec2<f32>>;
@group(2) @binding(1) var<storage, read_write> velocity_out: array<vec2<f32>>;
@group(2) @binding(2) var<storage, read> pressure_in: array<f32>;
@group(2) @binding(3) var<storage, read_write> pressure_out: array<f32>;
@group(2) @binding(4) var<storage, read_write> divergence: array<f32>;
@group(2) @binding(5) var<storage, read_write> temp_in: array<f32>;
@group(2) @binding(6) var<storage, read_write> temp_out: array<f32>;
// Forças acumuladas (f32 como bits, atómico: muitos escritores por célula).
@group(2) @binding(7) var<storage, read_write> force_vectors: array<atomic<u32>>;
@group(2) @binding(8) var<storage, read_write> fluid_forces: array<vec2<f32>>;
// Fonte de calor por célula do fluido (fumarolas pontuais + píxeis da imagem).
@group(2) @binding(9) var<storage, read> heat_src: array<f32>;
// Velocidade final SUAVIZADA (média de 5 células), calculada uma vez por passo
// do fluido; os resíduos dos agentes leem-na (uma leitura em vez de 5).
@group(2) @binding(10) var<storage, read_write> velocity_smooth: array<vec2<f32>>;
// Máscara das paredes do fluido (1 = sólido), refeita no início de cada passo
// do fluido: a pergunta "é parede?" passa a ser UMA leitura (antes eram 4 do
// terreno, e o realce de vorticidade fazia ~700 por célula).
@group(2) @binding(11) var<storage, read_write> solid_mask: array<u32>;
// REDUTOR das fumarolas (H₂S, H₂…): largado onde há calor de fumarola,
// levado e difundido como a temperatura, oxida-se devagar; os órgãos de
// quimiossíntese consomem-no (redox_eaten, ponto fixo, somas atómicas).
@group(2) @binding(12) var<storage, read_write> redox_in: array<f32>;
@group(2) @binding(13) var<storage, read_write> redox_out: array<f32>;
@group(2) @binding(14) var<storage, read_write> redox_eaten: array<atomic<u32>>;

// Sombra dos agentes: resíduos por célula da luz (LIGHT_SIZE²), refeita antes
// de cada cálculo da luz.
@group(1) @binding(6) var<storage, read_write> shade_grid: array<atomic<u32>>;
// Luz do passo seguinte (propagação; light_commit copia para light_grid).
@group(1) @binding(7) var<storage, read_write> light_next: array<f32>;
// AGREGAÇÃO (só com params.aggregation > 0): ativados por célula (todos os
// canais) e a soma dos 8 vizinhos, calculados antes do transporte (assim o
// transporte lê 5 valores por célula em vez de uma janela 5×5 × 4 canais).
@group(1) @binding(8) var<storage, read_write> agg_act: array<u32>;
@group(1) @binding(9) var<storage, read_write> agg_nb: array<u32>;
