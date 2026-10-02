//! Tabela dos aminoácidos: ÚNICA fonte dos valores que a simulação usa
//! (massa, volume, catálise, flexibilidade, ângulo de repouso, dobra máxima,
//! sensibilidades e condutividades dos sinais, afinidade de substrato,
//! absorção UV). Vive em `assets/aminoacidos.json`; é lida ao arrancar (se o
//! ficheiro não existir, usa-se a cópia embutida no programa, que é o mesmo
//! ficheiro na altura da compilação). O editor local (`crate::editor`) muda-a
//! ao vivo e grava-a no mesmo ficheiro.

use serde::{Deserialize, Serialize};

use super::amino::AA_LETTERS;
use crate::params::AaProps;

pub const TABLE_PATH: &str = "assets/aminoacidos.json";
const EMBEDDED: &str = include_str!("../../assets/aminoacidos.json");

/// Uma linha da tabela (nomes em português, como no ficheiro).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AminoRow {
    pub letra: String,
    pub massa: f32,
    pub volume: f32,
    pub catalise: f32,
    pub flexibilidade: f32,
    pub angulo_repouso: f32,
    pub dobra_max: f32,
    pub sens_alfa: f32,
    pub sens_beta: f32,
    #[serde(rename = "cond_alfa_N")]
    pub cond_alfa_n: f32,
    #[serde(rename = "cond_alfa_C")]
    pub cond_alfa_c: f32,
    #[serde(rename = "cond_beta_N")]
    pub cond_beta_n: f32,
    #[serde(rename = "cond_beta_C")]
    pub cond_beta_c: f32,
    #[serde(rename = "substrato_A")]
    pub substrato_a: f32,
    #[serde(rename = "substrato_U")]
    pub substrato_u: f32,
    #[serde(rename = "substrato_G")]
    pub substrato_g: f32,
    #[serde(rename = "substrato_C")]
    pub substrato_c: f32,
    pub absorcao_uv: f32,
    /// Comprimento do segmento (unidades do mundo; o v3 usava 11 para todos).
    pub comprimento: f32,
}

/// Uma linha da tabela dos órgãos (assets/orgaos.json).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrganRow {
    pub tipo: u32,
    pub nome: String,
    /// Comprimento do segmento = o do aminoácido promotor × isto.
    pub comprimento_mult: f32,
    /// Massa = a do aminoácido promotor × isto.
    pub massa_mult: f32,
}

pub const ORGANS_PATH: &str = "assets/orgaos.json";
const EMBEDDED_ORGANS: &str = include_str!("../../assets/orgaos.json");

/// Lê e valida a tabela dos órgãos: uma linha por tipo, ordenada.
pub fn parse_organs(text: &str) -> Result<Vec<OrganRow>, String> {
    let mut rows: Vec<OrganRow> = serde_json::from_str(text).map_err(|e| format!("JSON inválido: {e}"))?;
    rows.sort_by_key(|r| r.tipo);
    let n = super::organs::ORGAN_TYPES;
    if rows.len() != n || rows.iter().enumerate().any(|(i, r)| r.tipo as usize != i) {
        return Err(format!("são precisas {n} linhas, tipos 0..{}", n - 1));
    }
    Ok(rows)
}

pub fn load_organs() -> (Vec<OrganRow>, String) {
    match std::fs::read_to_string(ORGANS_PATH) {
        Ok(text) => match parse_organs(&text) {
            Ok(rows) => (rows, ORGANS_PATH.to_string()),
            Err(e) => {
                log::error!("{ORGANS_PATH}: {e}; uso a tabela embutida");
                (embedded_organs(), format!("embutida ({ORGANS_PATH} inválido)"))
            }
        },
        Err(_) => (embedded_organs(), "embutida".to_string()),
    }
}

pub fn embedded_organs() -> Vec<OrganRow> {
    parse_organs(EMBEDDED_ORGANS).expect("assets/orgaos.json embutido inválido")
}

pub fn save_organs(rows: &[OrganRow]) -> Result<(), String> {
    let lines: Result<Vec<String>, _> = rows.iter().map(serde_json::to_string).collect();
    let text = format!("[\n  {}\n]\n", lines.map_err(|e| e.to_string())?.join(",\n  "));
    std::fs::write(ORGANS_PATH, text).map_err(|e| format!("{ORGANS_PATH}: {e}"))
}

pub fn organs_to_gpu(rows: &[OrganRow]) -> Vec<crate::params::OrganProps> {
    rows.iter()
        .map(|r| crate::params::OrganProps { len_mult: r.comprimento_mult, mass_mult: r.massa_mult, _pad0: 0, _pad1: 0 })
        .collect()
}

/// Lê e valida: 20 linhas, uma por aminoácido; devolve-as na ordem de `AMINO`.
pub fn parse(text: &str) -> Result<Vec<AminoRow>, String> {
    let rows: Vec<AminoRow> = serde_json::from_str(text).map_err(|e| format!("JSON inválido: {e}"))?;
    let mut ordered = Vec::with_capacity(20);
    for l in AA_LETTERS {
        let found: Vec<&AminoRow> = rows.iter().filter(|r| r.letra == l.to_string()).collect();
        match found.len() {
            1 => ordered.push(found[0].clone()),
            0 => return Err(format!("falta o aminoácido {l}")),
            _ => return Err(format!("o aminoácido {l} aparece {} vezes", found.len())),
        }
    }
    if rows.len() != 20 {
        return Err(format!("{} linhas (devem ser 20)", rows.len()));
    }
    Ok(ordered)
}

/// A tabela do ficheiro, ou a embutida se o ficheiro não existir. Devolve
/// também de onde veio (para o log).
pub fn load() -> (Vec<AminoRow>, String) {
    match std::fs::read_to_string(TABLE_PATH) {
        Ok(text) => match parse(&text) {
            Ok(rows) => (rows, TABLE_PATH.to_string()),
            Err(e) => {
                log::error!("{TABLE_PATH}: {e}; uso a tabela embutida");
                (embedded(), format!("embutida ({TABLE_PATH} inválido)"))
            }
        },
        Err(_) => (embedded(), "embutida".to_string()),
    }
}

/// A tabela embutida no programa (o ficheiro na altura da compilação).
pub fn embedded() -> Vec<AminoRow> {
    parse(EMBEDDED).expect("assets/aminoacidos.json embutido inválido")
}

/// Grava a tabela no ficheiro (formatado, uma linha por aminoácido).
pub fn save(rows: &[AminoRow]) -> Result<(), String> {
    let lines: Result<Vec<String>, _> = rows.iter().map(serde_json::to_string).collect();
    let text = format!("[\n  {}\n]\n", lines.map_err(|e| e.to_string())?.join(",\n  "));
    std::fs::write(TABLE_PATH, text).map_err(|e| format!("{TABLE_PATH}: {e}"))
}

/// Para a GPU (ordem de `AMINO`).
pub fn to_gpu(rows: &[AminoRow]) -> Vec<AaProps> {
    rows.iter()
        .map(|r| AaProps {
            mass: r.massa,
            volume: r.volume,
            catalytic: r.catalise,
            flex: r.flexibilidade,
            rest_angle: r.angulo_repouso,
            max_bend: r.dobra_max,
            sens_alpha: r.sens_alfa,
            sens_beta: r.sens_beta,
            cond_alpha_n: r.cond_alfa_n,
            cond_alpha_c: r.cond_alfa_c,
            cond_beta_n: r.cond_beta_n,
            cond_beta_c: r.cond_beta_c,
            sub_a: r.substrato_a,
            sub_u: r.substrato_u,
            sub_g: r.substrato_g,
            sub_c: r.substrato_c,
            uv_absorb: r.absorcao_uv,
            seg_len: r.comprimento,
            _pad1: 0,
            _pad2: 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table_is_complete_and_substrates_sum_to_one() {
        let rows = embedded();
        assert_eq!(rows.len(), 20);
        for r in &rows {
            let s = r.substrato_a + r.substrato_u + r.substrato_g + r.substrato_c;
            assert!((s - 1.0).abs() < 1e-4, "{}: substrato soma {s}", r.letra);
        }
    }

    #[test]
    fn embedded_organ_table_is_complete() {
        assert_eq!(embedded_organs().len(), crate::life::organs::ORGAN_TYPES);
    }

    #[test]
    fn save_format_round_trips() {
        let rows = embedded();
        let lines: Vec<String> = rows.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
        let text = format!("[\n  {}\n]\n", lines.join(",\n  "));
        assert_eq!(parse(&text).unwrap(), rows);
    }
}
