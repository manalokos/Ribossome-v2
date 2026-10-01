//! Valida todos os módulos WGSL já montados (preâmbulo + ficheiros), com o
//! naga EXATO do wgpu em uso (`wgpu::naga`), para várias resoluções. Confirma
//! também que os entry points registados existem com o estágio certo e que
//! as structs geradas têm o mesmo tamanho no WGSL e no Rust.

use ribossome::params::{Fumarole, SimParams, ViewParams, WorldConfig};
use ribossome::shaders::{self, MODULES, Stage};
use wgpu::naga;

const CONFIGS: [WorldConfig; 3] = [
    WorldConfig::DEFAULT,
    WorldConfig::TEST,
    WorldConfig { grid_size: 1024, fluid_size: 256, world_units_per_cell: 30, max_agents: 1000 },
];

fn parse_and_validate(name: &str, src: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("[{name}] erro de parse:\n{}", e.emit_to_string(src)),
    };
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default());
    let info = match v.validate(&module) {
        Ok(i) => i,
        Err(e) => panic!("[{name}] erro de validação:\n{}", e.emit_to_string(src)),
    };
    (module, info)
}

#[test]
fn all_modules_validate() {
    for cfg in CONFIGS {
        for def in MODULES {
            let src = shaders::source(def, &cfg);
            parse_and_validate(&format!("{} @ {}²", def.name, cfg.grid_size), &src);
        }
    }
}

#[test]
fn registered_entry_points_exist() {
    for def in MODULES {
        let (module, _) = parse_and_validate(def.name, &shaders::source(def, &WorldConfig::TEST));
        for (name, stage) in def.entries {
            let ep = module
                .entry_points
                .iter()
                .find(|e| e.name == *name)
                .unwrap_or_else(|| panic!("[{}] entry point '{name}' não existe no WGSL", def.name));
            let want = match stage {
                Stage::Compute => naga::ShaderStage::Compute,
                Stage::Vertex => naga::ShaderStage::Vertex,
                Stage::Fragment => naga::ShaderStage::Fragment,
            };
            assert_eq!(ep.stage, want, "[{}] '{name}' tem o estágio errado", def.name);
        }
    }
}

#[test]
fn generated_structs_match_rust_layout() {
    let src = shaders::preamble(&WorldConfig::TEST);
    let (module, _) = parse_and_validate("preâmbulo", &src);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).expect("layout");
    for (name, rust_size) in [
        (SimParams::WGSL_NAME, size_of::<SimParams>()),
        (ViewParams::WGSL_NAME, size_of::<ViewParams>()),
        (Fumarole::WGSL_NAME, size_of::<Fumarole>()),
    ] {
        let (handle, _) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name} não encontrada no preâmbulo"));
        assert_eq!(layouter[handle].size as usize, rust_size, "{name}: tamanho WGSL ≠ Rust");
    }
}
