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
    /// Parâmetros por frame (grupo 0, binding 0).
    pub struct SimParams {
        /// Contador de passos da simulação (semente temporal do RNG).
        pub epoch: u32,
        /// Semente do mundo.
        pub seed: u32,
        /// Multiplicador da difusão (slider; 1 = valor afinado do v3).
        pub diffusion: f32,
        pub _pad0: u32,
    }
}

impl Default for SimParams {
    fn default() -> Self {
        Self { epoch: 0, seed: 1, diffusion: 1.0, _pad0: 0 }
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
        pub _pad0: u32,
        pub _pad1: u32,
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
}

impl WorldConfig {
    pub const DEFAULT: Self = Self { grid_size: 2048, fluid_size: 512, world_units_per_cell: 30 };
    /// Mundo pequeno para testes.
    pub const TEST: Self = Self { grid_size: 256, fluid_size: 64, world_units_per_cell: 30 };

    pub fn sim_size(&self) -> f32 {
        (self.grid_size * self.world_units_per_cell) as f32
    }

    pub fn cells(&self) -> u64 {
        self.grid_size as u64 * self.grid_size as u64
    }
}
