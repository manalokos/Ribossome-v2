//! Montagem dos módulos WGSL.
//!
//! Cada módulo = preâmbulo gerado (constantes do mundo + structs de
//! `params.rs`) + uma lista fixa de ficheiros. O registo `MODULES` declara
//! também os entry points que o Rust usa: o teste `tests/shaders.rs` valida
//! cada módulo com o naga do próprio wgpu e confirma que todos existem.

use crate::params::{Agent, Fumarole, SimParams, SpawnRequest, ViewParams, WorldConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

pub struct ModuleDef {
    pub name: &'static str,
    /// (caminho relativo a `shaders/`, conteúdo)
    pub files: &'static [(&'static str, &'static str)],
    pub entries: &'static [(&'static str, Stage)],
}

macro_rules! wgsl_files {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../shaders/", $path)))),*]
    };
}

pub const WORLD: ModuleDef = ModuleDef {
    name: "world",
    files: wgsl_files![
        "world/bindings.wgsl",
        "common/rng.wgsl",
        "common/terrain.wgsl",
        "common/chem.wgsl",
        "world/fluid.wgsl",
        "world/light.wgsl",
        "world/transport.wgsl",
        "world/terrain.wgsl",
        "world/ledger.wgsl",
        "life/bindings.wgsl",
        "life/lifecycle.wgsl",
        "life/body.wgsl",
    ],
    entries: &[
        ("transport_scatter", Stage::Compute),
        ("transport_commit", Stage::Compute),
        ("thermal_activation", Stage::Compute),
        ("ledger_reduce", Stage::Compute),
        ("compute_uv_light", Stage::Compute),
        ("clear_force_vectors", Stage::Compute),
        ("update_temperature", Stage::Compute),
        ("copy_temperature", Stage::Compute),
        ("buoyancy", Stage::Compute),
        ("gather_forces", Stage::Compute),
        ("add_forces", Stage::Compute),
        ("clear_forces", Stage::Compute),
        ("diffuse_velocity", Stage::Compute),
        ("advect_velocity", Stage::Compute),
        ("vorticity_confinement", Stage::Compute),
        ("compute_divergence", Stage::Compute),
        ("jacobi_pressure", Stage::Compute),
        ("subtract_gradient", Stage::Compute),
        ("enforce_boundaries", Stage::Compute),
        ("compute_gamma_slope", Stage::Compute),
        ("relax_gamma_a", Stage::Compute),
        ("relax_gamma_b", Stage::Compute),
        ("spawn_seeds", Stage::Compute),
        ("agents_step", Stage::Compute),
        ("agents_ledger", Stage::Compute),
        ("agents_birth", Stage::Compute),
    ],
};

pub const WORLD_VIEW: ModuleDef = ModuleDef {
    name: "world_view",
    files: wgsl_files!["render/world_view.wgsl"],
    entries: &[("vs_fullscreen", Stage::Vertex), ("fs_world", Stage::Fragment)],
};

pub const AGENTS_VIEW: ModuleDef = ModuleDef {
    name: "agents_view",
    files: wgsl_files!["render/agents_view.wgsl"],
    entries: &[("vs_agent", Stage::Vertex), ("fs_agent", Stage::Fragment)],
};

pub const MODULES: &[&ModuleDef] = &[&WORLD, &WORLD_VIEW, &AGENTS_VIEW];

/// Constantes do mundo e da química partilhadas por todos os módulos.
pub const CHEM_CELL_CAP: u32 = 48;

pub fn preamble(cfg: &WorldConfig) -> String {
    let mut s = String::new();
    s += "// ---- preâmbulo gerado (src/shaders.rs) ----\n";
    s += &format!("const GRID_SIZE: u32 = {}u;\n", cfg.grid_size);
    s += &format!("const FLUID_SIZE: u32 = {}u;\n", cfg.fluid_size);
    s += &format!("const WORLD_UNITS_PER_CELL: u32 = {}u;\n", cfg.world_units_per_cell);
    s += &format!("const SIM_SIZE: f32 = {:.1};\n", cfg.sim_size());
    s += &format!("const CHEM_CELL_CAP: u32 = {}u;\n", CHEM_CELL_CAP);
    s += &SimParams::wgsl();
    s += &ViewParams::wgsl();
    s += &Fumarole::wgsl();
    s += &Agent::wgsl();
    s += &SpawnRequest::wgsl();
    s += &crate::life::amino::wgsl();
    s
}

/// Fonte completa de um módulo, tal como é entregue ao wgpu.
pub fn source(def: &ModuleDef, cfg: &WorldConfig) -> String {
    let mut s = preamble(cfg);
    for (path, text) in def.files {
        s += &format!("\n// ---- {path} ----\n");
        s += text;
    }
    s
}

pub fn create(device: &wgpu::Device, def: &ModuleDef, cfg: &WorldConfig) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(def.name),
        source: wgpu::ShaderSource::Wgsl(source(def, cfg).into()),
    })
}

/// Devolve o nome do entry point depois de confirmar que está registado.
/// Um entry point usado no Rust e ausente do registo falha aqui (e no teste),
/// e não como um pipeline órfão a crashar no arranque.
pub fn entry(def: &ModuleDef, name: &'static str) -> &'static str {
    assert!(
        def.entries.iter().any(|(e, _)| *e == name),
        "entry point '{name}' não está registado no módulo '{}'",
        def.name
    );
    name
}
