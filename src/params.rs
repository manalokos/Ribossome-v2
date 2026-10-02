//! Tipos partilhados CPU↔GPU, definidos UMA vez.
//!
//! A macro `gpu_struct!` gera a struct Rust (Pod, `repr(C)`) e o texto WGSL
//! equivalente. O teste `tests/shaders.rs` confirma que o layout que o naga
//! calcula para o WGSL tem o mesmo tamanho que a struct Rust.

/// Mapeia um tipo escalar Rust para o nome WGSL.
pub trait WgslScalar: Copy {
    const WGSL: &'static str;
    /// Conversões exatas para gravar os campos por nome (cenas).
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}
impl WgslScalar for f32 {
    const WGSL: &'static str = "f32";
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl WgslScalar for u32 {
    const WGSL: &'static str = "u32";
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as u32
    }
}
impl WgslScalar for i32 {
    const WGSL: &'static str = "i32";
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as i32
    }
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

            /// Os campos por nome (para gravar cenas: um campo novo ou
            /// removido não estraga uma gravação antiga).
            pub fn to_named(&self) -> Vec<(&'static str, f64)> {
                vec![$( (stringify!($field), <$ty as $crate::params::WgslScalar>::to_f64(self.$field)) ),*]
            }

            /// Muda um campo pelo nome; false se não existir.
            pub fn set_named(&mut self, name: &str, v: f64) -> bool {
                match name {
                    $( stringify!($field) => { self.$field = <$ty as $crate::params::WgslScalar>::from_f64(v); true } )*
                    _ => false,
                }
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
        /// Movimento browniano: desvio por passo (unidades do mundo) de um
        /// corpo com o raio de UM resíduo; corpos maiores andam menos
        /// (Stokes-Einstein: D ∝ 1/raio).
        pub brownian: f32,
        /// Difusioforese (v3): unidades do mundo por passo por unidade de
        /// fluxo de consumo não compensado.
        pub phoretic_gain: f32,
        /// Rigidez das juntas (RT/rad² para flexibilidade 1; ÷ flexibilidade²).
        pub chain_stiffness: f32,
        /// Agitação térmica das juntas (kT em RT à temperatura ambiente).
        pub thermal_kt: f32,
        /// Desvio de ângulo (rad) de uma junta catalítica com o ligando
        /// ligado (+) e com o produto (−): o curso do motor.
        pub motor_amplitude: f32,
        /// 1 = natação por forças resistivas (RFT) a partir da mudança de forma.
        pub rft_enabled: u32,
        /// 1 = a tradução começa no primeiro AUG; 0 = na primeira base
        /// (por omissão: o AUG é uma convenção da maquinaria moderna).
        pub require_start: u32,
        /// Acoplamento mecânico: fração do desvio das juntas vizinhas que
        /// passa a cada junta (propagação de mudanças de forma ao longo da cadeia).
        pub joint_coupling: f32,
        /// 1 = regulação pela fome (v3): um organismo cheio deixa de ligar comida.
        pub hunger_regulation: u32,
        /// Energia gasta por base emparelhada (polimerizar consome energia).
        pub pairing_cost: f32,
        /// Reativação UNIFORME: probabilidade por passo de um monómero gasto
        /// voltar a ativado, em qualquer lado (modo laboratório; 0 = só luz/calor).
        pub reactivation_rate: f32,
        /// Ganho da natação: escala o deslocamento (translação e rotação) que o
        /// RFT calcula por mudança de forma, i.e. a rapidez do corpo face ao mundo.
        pub swim_gain: f32,
        /// BIOTURBAÇÃO: probabilidade, por célula percorrida e por resíduo,
        /// de um resíduo em movimento empurrar um grão de ENTULHO para a
        /// célula seguinte (a rocha não se mexe). 0 = desligada.
        pub bioturbation: f32,
        /// Energia gasta por grão empurrado (o trabalho não é de graça).
        pub bioturbation_cost: f32,
        /// Absorção da UV pelos MONÓMEROS (ativados e gastos: os nucleótidos
        /// absorvem UV). Profundidade ótica do topo ao fundo por monómero
        /// médio por célula (independente da resolução).
        pub monomer_uv_absorb: f32,
        /// SEDIMENTAÇÃO dos agentes (Stokes): afundam sedimentation·√n
        /// unidades do mundo por passo (n = resíduos; peso ∝ n, arrasto ∝ √n).
        pub sedimentation: f32,
        /// Fotoativação DIRETA dos gastos pela luz (multiplica a força UV).
        /// 0 por omissão: a luz só vira comida pelos FOTOSSISTEMAS dos agentes.
        pub direct_photoactivation: f32,
        /// Decaimento espontâneo da ativação (hidrólise), por monómero
        /// ativado e por passo. 0 = os ativados não decaem sozinhos.
        pub activation_decay: f32,
        /// PRESSÃO dos monómeros (osmótica): os saltos de difusão preferem a
        /// vizinha menos cheia, ∝ à diferença de enchimento. 0 = desligada.
        pub monomer_pressure: f32,
        /// Os agentes EMPURRAM a água: cada resíduo devolve ao fluido o seu
        /// arrasto (soma zero para um nadador: um dipolo). 0 = desligado.
        pub agent_fluid_push: f32,
        /// Fração do VAIVÉM de cada batida que se aplica (1 = físico, 0 = só
        /// o avanço médio, sem balanço).
        pub swim_wobble: f32,
        /// EXPERIÊNCIA: natação só pelo fluido. Sem o movimento do RFT; cada
        /// resíduo empurra a água com a sua velocidade de mudança de forma e o
        /// agente é levado pela água no centro (sem média). 0 = desligada.
        pub fluid_swim_only: u32,
        /// Custo de DISSIPAÇÃO do movimento: energia por passo = isto ×
        /// Σ √arrasto·dθ² das juntas (potência viscosa ξ·ω²; só a parte ativa,
        /// o ruído térmico vem do banho). Mover depressa custa ao quadrado.
        pub motion_cost: f32,
        /// LIGAÇÕES (pontes salinas): probabilidade por passo de um agente
        /// tentar ligar um resíduo carregado a um vizinho de carga oposta.
        pub bond_rate: f32,
        /// Probabilidade por passo de uma ligação se soltar (agitação).
        pub bond_break: f32,
        /// Fração da diferença de energia que passa por cada ligação, por passo.
        pub bond_energy_share: f32,
        /// Condutância dos sinais α/β pela ligação.
        pub bond_signal: f32,
        pub _pad_b0: u32,
        pub _pad_b1: u32,
        pub _pad_b2: u32,
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
            // 2 (era 11 do v3): o resto da atenuação vem dos monómeros.
            uv_depth: 2.0,
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
            // Mais cara do que no v3 (0,0001): com ela a energia nunca faltava e
            // não havia seleção. 0,002: a fome passa a ser a principal causa de morte.
            maintenance_cost: 0.002,
            pairing_rate: 3.0,
            mutation_rate: 0.003,
            uv_damage: 10.0,
            uptake_rate: 0.0005,
            // 0: a agitação térmica já entra pelo tremor das juntas (RFT); somar
            // este browniano contava-a duas vezes e afogava a natação.
            brownian: 0.0,
            phoretic_gain: 100.0,
            chain_stiffness: 20.0,
            // 0 por omissão (pedido do Filipe): sem tremor térmico, o movimento
            // próprio vem só dos sinais.
            thermal_kt: 0.0,
            // 0: com órgãos, quem move as juntas são os músculos (o motor
            // catalítico "puro" fica como opção).
            motor_amplitude: 0.0,
            rft_enabled: 1,
            require_start: 1,
            joint_coupling: 0.9,
            // 1: inibição pela carga energética (um agente cheio não come).
            hunger_regulation: 1,
            pairing_cost: 0.3,
            reactivation_rate: 0.0,
            // 10: com 1 a natação era lenta demais para dar vantagem visível.
            swim_gain: 10.0,
            bioturbation: 0.05,
            bioturbation_cost: 0.05,
            // 1,1: calibrado para a luz a meio do mundo ficar como antes só
            // com a água (uv_depth 11 do v3): ~0,4% da luz do topo.
            monomer_uv_absorb: 1.1,
            // ~1/20 da natação de um agente médio: afundar é lento, quem nada
            // para cima vence-o.
            sedimentation: 0.02,
            direct_photoactivation: 0.0,
            // 0 (pedido do Filipe; o valor antigo fixo era 0,0002).
            activation_decay: 0.0,
            monomer_pressure: 20.0,
            agent_fluid_push: 0.25,
            swim_wobble: 1.0,
            fluid_swim_only: 0,
            motion_cost: 0.1,
            bond_rate: 0.05,
            bond_break: 0.0002,
            bond_energy_share: 0.01,
            bond_signal: 0.5,
            _pad_b0: 0,
            _pad_b1: 0,
            _pad_b2: 0,
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
        /// Raio de contacto (unidades do mundo): raio de giração do corpo.
        pub radius: f32,
        /// `id` do pai (0xFFFFFFFF na geração 0).
        pub parent: u32,
        /// Zona traduzida do genoma: base do AUG (16 bits baixos) e primeira
        /// base depois do codão stop (16 bits altos). Antes = 5' UTR, depois =
        /// 3' UTR (desenhadas como fios de RNA nas pontas).
        pub coding_span: u32,
    }
}

gpu_struct! {
    /// Um nível do multigrid da pressão: tamanho e offset do nível e do de baixo.
    pub struct MgLevel {
        pub n: u32,
        pub off: u32,
        pub n_c: u32,
        pub off_c: u32,
        /// Termo de ancoragem na diagonal (escala com o espaçamento²).
        pub eps: f32,
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
        /// bit 1: genoma ESCOLHIDO (em `genome`); cada base é tirada da sopa,
        /// a mais próxima do tipo pedido (a matéria continua exata).
        pub flags: u32,
        pub genome0: u32,
        pub genome1: u32,
        pub genome2: u32,
        pub genome3: u32,
        pub genome4: u32,
        pub genome5: u32,
        pub genome6: u32,
        pub genome7: u32,
        pub genome8: u32,
        pub genome9: u32,
        pub genome10: u32,
        pub genome11: u32,
        pub genome12: u32,
        pub genome13: u32,
        pub genome14: u32,
        pub genome15: u32,
    }
}

impl SpawnRequest {
    /// Pedido com posição, comprimento e flags (genoma vazio).
    pub fn new(pos_x: f32, pos_y: f32, gene_len: u32, flags: u32) -> Self {
        let mut r: Self = bytemuck::Zeroable::zeroed();
        r.pos_x = pos_x;
        r.pos_y = pos_y;
        r.gene_len = gene_len;
        r.flags = flags;
        r
    }

    /// Pedido com um genoma escolhido (bases 0 = A, 1 = U, 2 = G, 3 = C).
    pub fn with_genome(pos_x: f32, pos_y: f32, genome: &[u8]) -> Self {
        let n = genome.len().min(256);
        let mut r = Self::new(pos_x, pos_y, n as u32, 2);
        let words: &mut [u32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut r));
        for (i, &b) in genome[..n].iter().enumerate() {
            words[4 + i / 16] |= (b as u32 & 3) << ((i % 16) * 2);
        }
        r
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
    /// Propriedades de um aminoácido na GPU (de `life::table`, a tabela
    /// `assets/aminoacidos.json`). 20 entradas, ordem de `AMINO`.
    pub struct AaProps {
        pub mass: f32,
        pub volume: f32,
        pub catalytic: f32,
        pub flex: f32,
        pub rest_angle: f32,
        pub max_bend: f32,
        pub sens_alpha: f32,
        pub sens_beta: f32,
        pub cond_alpha_n: f32,
        pub cond_alpha_c: f32,
        pub cond_beta_n: f32,
        pub cond_beta_c: f32,
        pub sub_a: f32,
        pub sub_u: f32,
        pub sub_g: f32,
        pub sub_c: f32,
        pub uv_absorb: f32,
        /// Comprimento do segmento (unidades do mundo).
        pub seg_len: f32,
        /// Carga a pH ~7 (K, R +1; D, E −1): pontes salinas entre agentes.
        pub charge: f32,
        pub _pad2: u32,
    }
}

gpu_struct! {
    /// Multiplicadores físicos de um tipo de órgão (assets/orgaos.json).
    pub struct OrganProps {
        pub len_mult: f32,
        pub mass_mult: f32,
        /// Manutenção por passo (múltiplos da de um resíduo).
        pub upkeep: f32,
        /// Arrasto do segmento × isto.
        pub drag_mult: f32,
        /// 1 = a manutenção multiplica pela intensidade (órgãos que fazem
        /// trabalho); 0 = não (sinais: a intensidade é só um peso).
        pub gain_pays: f32,
        pub _pad0: f32,
        pub _pad1: f32,
        pub _pad2: f32,
    }
}

gpu_struct! {
    /// Propriedades de uma variante de órgão (tipo·6 + parâmetro), pela
    /// ordem de `organs::ORGAN_PROPS` do tipo.
    pub struct OrganVariant {
        pub p0: f32,
        pub p1: f32,
        pub p2: f32,
        pub p3: f32,
        pub p4: f32,
        pub p5: f32,
        pub p6: f32,
        pub p7: f32,
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
        /// Slot a desenhar sozinho (0xFFFFFFFF = todos).
        pub focus_slot: u32,
        /// Cor dos agentes: 0 química, 1 sinal α, 2 sinal β, 3 α e β.
        pub signal_view: u32,
        /// Profundidade ótica da água (= SimParams::uv_depth), para a vista
        /// normal separar a sombra do terreno do escurecer com a profundidade.
        pub uv_depth: f32,
        pub _vpad1: u32,
        pub _vpad2: u32,
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
    pub const DEFAULT: Self = Self { grid_size: 2048, fluid_size: 1024, world_units_per_cell: 30, max_agents: 400_000 };
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
