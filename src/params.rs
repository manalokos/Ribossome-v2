//! Tipos partilhados CPU↔GPU, definidos UMA vez.
//!
//! A macro `gpu_struct!` gera a struct Rust (Pod, `repr(C)`) e o texto WGSL
//! equivalente. O teste `tests/shaders.rs` confirma que o layout que o naga
//! calcula para o WGSL tem o mesmo tamanho que a struct Rust.

/// Mapeia um tipo escalar Rust para o nome WGSL.
pub trait WgslScalar {
    const WGSL: &'static str;
}
impl WgslScalar for f32 {
    const WGSL: &'static str = "f32";
}
impl WgslScalar for u32 {
    const WGSL: &'static str = "u32";
}
impl WgslScalar for i32 {
    const WGSL: &'static str = "i32";
}

/// Struct uniforme partilhada. Só escalares de 4 bytes, para o layout WGSL
/// ser trivialmente igual ao `repr(C)`; o tamanho tem de ser múltiplo de 16.
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        pub struct $name:ident { $( $(#[$fmeta:meta])* pub $field:ident : $ty:ty ),* $(,)? }
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name { $( $(#[$fmeta])* pub $field: $ty ),* }

        const _: () = assert!(
            std::mem::size_of::<$name>() % 16 == 0,
            concat!(stringify!($name), ": o tamanho tem de ser múltiplo de 16 (acrescenta _pad)")
        );

        impl $name {
            pub const WGSL_NAME: &'static str = stringify!($name);

            pub fn wgsl() -> String {
                let mut s = format!("struct {} {{\n", stringify!($name));
                $( s += &format!("    {}: {},\n", stringify!($field), <$ty as $crate::params::WgslScalar>::WGSL); )*
                s += "}\n";
                s
            }
        }
    };
}

gpu_struct! {
    /// Parâmetros por passo (grupo 0, binding 0). Uma cópia por passo.
    /// Os valores por omissão são os da última corrida do v3
    /// (`simulation_settings.json`, que se sobrepunha aos defaults do código).
    pub struct SimParams {
        /// Contador de passos da simulação (semente temporal do RNG).
        pub epoch: u32,
        /// Semente do mundo.
        pub seed: u32,
        /// Tempo por passo (s). O fluido e os monómeros usam o mesmo.
        pub dt: f32,
        /// Multiplicador da difusão dos monómeros (slider "monomer_diffusion" do v3).
        pub diffusion: f32,
        /// Multiplicador do assentamento dos monómeros (slider "gravity_monomer" do v3).
        pub settle: f32,
        /// dt de uma resolução do fluido = dt × fluid_substep.
        pub fluid_dt: f32,
        /// Amortecimento da velocidade por frame a 60 fps.
        pub fluid_decay: f32,
        /// Força do confinamento de vorticidade (limitada a 10).
        pub fluid_vorticity: f32,
        /// Viscosidade (células²/s).
        pub fluid_viscosity: f32,
        /// Número de fumarolas no buffer.
        pub fumarole_count: u32,
        /// Força da fotoativação UV (slider "uv_strength" do v3).
        pub uv_strength: f32,
        /// Atenuação da UV pela água, do topo ao fundo (slider "uv_depth" do v3).
        pub uv_depth: f32,
        /// 1 = fluido ligado (com 0 os monómeros e grãos não são advectados).
        pub fluid_enabled: u32,
        /// Permeabilidade pelo declive: perm = 1/(1 + k·|declive|).
        pub fluid_obstacle_strength: f32,
        /// Rapidez com que o escoamento roda para "declive abaixo" (1/s).
        pub slope_steer_rate: f32,
        /// Coesão: enviesamento da difusão dos ativados para vizinhos do mesmo tipo.
        pub cohesion: f32,
        // ---- vida ----
        /// Pedidos de sementes (geração 0) a processar neste passo.
        pub spawn_count: u32,
        /// Capacidade de agentes (slots).
        pub max_agents: u32,
        /// Mortalidade base por passo (÷ energia, como no v3; "death_probability").
        pub death_probability: f32,
        /// Energia inicial de uma semente.
        pub spawn_energy: f32,
        /// Energia por monómero hidrolisado ("food_power" do v3).
        pub food_power: f32,
        /// Custo de manutenção por resíduo e por passo ("amino_maintenance_cost").
        pub maintenance_cost: f32,
        /// Bases emparelhadas por passo, em média ("spawn_probability" do v3).
        pub pairing_rate: f32,
        /// Taxa de mutação por base na cópia.
        pub mutation_rate: f32,
        /// Multiplicador do dano UV à superfície ("uv_damage").
        pub uv_damage: f32,
        /// Probabilidade de hidrólise por monómero ativado, por unidade de
        /// propensão catalítica e por passo (v3: 0,01 × massa mínima 0,1).
        pub uptake_rate: f32,
        pub _pad0: u32,
        pub _pad1: u32,
    }
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            epoch: 0,
            seed: 1,
            dt: 0.017,
            diffusion: 20.0,
            settle: 0.0,
            fluid_dt: 0.017 * 2.0,
            fluid_decay: 0.999,
            fluid_vorticity: 7.0,
            fluid_viscosity: 3.7,
            fumarole_count: 0,
            uv_strength: 3.0,
            uv_depth: 11.0,
            fluid_enabled: 1,
            fluid_obstacle_strength: 1000.0,
            slope_steer_rate: 210.0,
            // 0 (o v3 tinha 0.6): a coesão separava os nucleótidos por tipo em fios
            // e condensava-os em camadas presas junto ao fundo e às rochas.
            cohesion: 0.0,
            spawn_count: 0,
            max_agents: 0,
            death_probability: 0.02,
            spawn_energy: 5.0,
            food_power: 6.0,
            maintenance_cost: 0.0001,
            pairing_rate: 3.0,
            mutation_rate: 0.003,
            uv_damage: 10.0,
            uptake_rate: 0.001,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

gpu_struct! {
    /// Agente: dados "quentes" (lidos todos os passos). O genoma vive num
    /// buffer à parte (16 u32 por slot, 2 bits por base, a partir da base 0).
    pub struct Agent {
        /// Posição e velocidade em unidades do MUNDO (e por segundo).
        pub pos_x: f32,
        pub pos_y: f32,
        pub vel_x: f32,
        pub vel_y: f32,
        pub rot: f32,
        /// Energia = ativação colhida (não é matéria; evapora na morte).
        pub energy: f32,
        /// 1 = vivo, 0 = slot livre.
        pub alive: u32,
        /// Bases do genoma (cada uma é um monómero real preso no agente).
        pub gene_len: u32,
        /// Complementos já capturados (também matéria presa).
        pub pair_count: u32,
        /// Resíduos do corpo traduzido.
        pub body_len: u32,
        pub generation: u32,
        pub age: u32,
        /// Identificador único (para o RNG; não muda com o slot).
        pub id: u32,
        pub _pad0: u32,
        pub _pad1: u32,
        pub _pad2: u32,
    }
}

gpu_struct! {
    /// Pedido de semente (geração 0), escrito pelo CPU.
    pub struct SpawnRequest {
        /// Posição em unidades do mundo.
        pub pos_x: f32,
        pub pos_y: f32,
        /// Número de bases a montar.
        pub gene_len: u32,
        /// bit 0: começar por AUG (bases também tiradas da vizinhança).
        pub flags: u32,
    }
}

gpu_struct! {
    /// Fumarola: fonte de calor no fundo. A flutuação vem só da temperatura.
    /// (No v3 havia também direção, variação e taxas de dye: já não eram usadas.)
    pub struct Fumarole {
        /// Posição em fração do mundo (0..1).
        pub x_frac: f32,
        pub y_frac: f32,
        /// Intensidade do aquecimento.
        pub strength: f32,
        /// Raio, em unidades do MUNDO.
        pub spread: f32,
        pub enabled: u32,
        pub _pad0: u32,
        pub _pad1: u32,
        pub _pad2: u32,
    }
}

impl Fumarole {
    pub fn new(x_frac: f32, y_frac: f32, strength: f32, spread_world: f32) -> Self {
        Self { x_frac, y_frac, strength, spread: spread_world, enabled: 1, _pad0: 0, _pad1: 0, _pad2: 0 }
    }

    /// A fumarola ativa da última corrida do v3. O raio era 13,65 células de
    /// um fluido de 1024² num mundo de 61440 → 13,65 × 60 = 819 unidades.
    pub fn v3_default() -> Self {
        Self::new(0.359, 0.0425, 5000.0, 819.0)
    }
}

gpu_struct! {
    /// Câmara e vista (grupo próprio do render).
    pub struct ViewParams {
        /// Centro da câmara, em unidades do mundo.
        pub center_x: f32,
        pub center_y: f32,
        /// Píxeis do ecrã por unidade do mundo.
        pub zoom: f32,
        /// Vista de debug (0 = normal, 1–4 = ativados por canal, 5 = gastos).
        pub view_mode: u32,
        pub screen_w: f32,
        pub screen_h: f32,
        /// Brilho da camada de monómeros (0..1; "monomer_brightness" do v3).
        pub monomer_brightness: f32,
        pub _pad0: u32,
    }
}

/// Dimensões do mundo, escolhidas ao arrancar e injetadas como `const` nos
/// shaders. Unidades: `sim_size` (mundo) ≠ `grid_size` (células) ≠
/// `fluid_size` (células do fluido). Converte sempre explicitamente.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldConfig {
    pub grid_size: u32,
    pub fluid_size: u32,
    /// Unidades do mundo por célula do ambiente (61440/2048 = 30 no v3).
    pub world_units_per_cell: u32,
    /// Capacidade de agentes (slots fixos).
    pub max_agents: u32,
}

impl WorldConfig {
    /// Fluido a 1024²: era o que o v3 corria (as constantes do fluido estão
    /// afinadas em células do fluido).
    pub const DEFAULT: Self = Self { grid_size: 2048, fluid_size: 1024, world_units_per_cell: 30, max_agents: 60_000 };
    /// Mundo pequeno para testes.
    pub const TEST: Self = Self { grid_size: 256, fluid_size: 128, world_units_per_cell: 30, max_agents: 4096 };

    pub fn sim_size(&self) -> f32 {
        (self.grid_size * self.world_units_per_cell) as f32
    }

    pub fn cells(&self) -> u64 {
        self.grid_size as u64 * self.grid_size as u64
    }

    pub fn fluid_cells(&self) -> u64 {
        self.fluid_size as u64 * self.fluid_size as u64
    }
}
