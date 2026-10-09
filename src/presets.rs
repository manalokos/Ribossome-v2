//! PRESETS DE LANÇAMENTO: um ficheiro JSON por preset em `assets/presets/`.
//! Carregar num preset põe os parâmetros todos (os valores por omissão mais
//! os que o ficheiro muda), liga ou desliga o fluido, escolhe o terreno,
//! semeia o mundo de novo e lança a população. O tamanho do mundo NÃO faz
//! parte do preset (escolhe-se à parte, ao arrancar).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::world::terrain::NoiseTerrain;

/// Pasta dos presets (e das imagens de terreno que eles referem).
pub const DIR: &str = "assets/presets";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum PresetTerrain {
    /// O terreno do projeto (assets/terreno.png).
    Default,
    /// Só água, sem fumarolas (para pintar à mão).
    Empty,
    /// Uma imagem de terreno (caminho relativo à pasta do projeto).
    Image { path: String },
    /// Gerado por ruído, com fumarolas automáticas.
    Noise(NoiseTerrain),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Preset {
    pub name: String,
    pub description: String,
    pub fluid: bool,
    pub terrain: PresetTerrain,
    /// Força das fumarolas (o multiplicador do separador World).
    pub fumarole_gain: f32,
    /// Parâmetros que diferem dos valores por omissão (nome -> valor).
    pub params: BTreeMap<String, f64>,
    /// População de arranque: quantas sementes (0 = nenhuma) e o intervalo
    /// de comprimentos do genoma.
    pub seeds: u32,
    pub seed_len: [u32; 2],
    /// Passos por frame (0 = não mexe).
    pub steps_per_frame: u32,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            fluid: true,
            terrain: PresetTerrain::Default,
            fumarole_gain: 1.0,
            params: BTreeMap::new(),
            seeds: 2000,
            seed_len: [12, 120],
            steps_per_frame: 0,
        }
    }
}

/// Os presets da pasta, por ordem do nome do ficheiro. Um ficheiro que não
/// se consiga ler fica de fora (com um aviso no registo).
pub fn list() -> Vec<(PathBuf, Preset)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(DIR)
        .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect())
        .unwrap_or_default();
    files.sort();
    files
        .into_iter()
        .filter_map(|path| match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Preset>(&t).map_err(|e| e.to_string())) {
            Ok(mut p) => {
                if p.name.is_empty() {
                    p.name = path.file_stem().map(|s| s.to_string_lossy().replace('_', " ")).unwrap_or_default();
                }
                Some((path, p))
            }
            Err(e) => {
                log::warn!("preset {}: {e}", path.display());
                None
            }
        })
        .collect()
}

/// Nome de ficheiro (sem extensão) para um preset com este nome.
pub fn file_stem(name: &str) -> String {
    let s: String = name.trim().chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    let s = s.trim_matches('_').to_string();
    if s.is_empty() { "preset".into() } else { s }
}

/// Grava o preset em `assets/presets/<nome>.json`.
pub fn save(preset: &Preset) -> Result<PathBuf, String> {
    std::fs::create_dir_all(DIR).map_err(|e| e.to_string())?;
    let path = Path::new(DIR).join(format!("{}.json", file_stem(&preset.name)));
    let text = serde_json::to_string_pretty(preset).map_err(|e| e.to_string())?;
    std::fs::write(&path, text + "\n").map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_round_trips_and_fills_defaults() {
        let p: Preset = serde_json::from_str(r#"{"name":"x","terrain":{"kind":"noise","seed":7},"params":{"dt":0.01}}"#).unwrap();
        assert!(p.fluid && p.seeds == 2000);
        let PresetTerrain::Noise(n) = &p.terrain else { panic!() };
        assert_eq!(n.seed, 7);
        assert_eq!(n.octaves, NoiseTerrain::default().octaves);
        let back: Preset = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back.terrain, p.terrain);
        assert_eq!(file_stem("  My Run #2 "), "my_run__2");
    }

    /// Os presets que vêm com o projeto leem-se todos e só mexem em
    /// parâmetros que existem.
    #[test]
    fn shipped_presets_are_valid() {
        let all = list();
        assert!(all.len() >= 5, "presets em falta em {DIR}");
        for (path, p) in all {
            let mut params = crate::params::SimParams::default();
            for (k, v) in &p.params {
                assert!(params.set_named(k, *v), "{}: parâmetro desconhecido {k}", path.display());
            }
            assert!(!p.name.is_empty() && p.seed_len[0] <= p.seed_len[1]);
        }
    }
}
