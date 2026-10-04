//! Tabela dos aminoácidos: ÚNICA fonte dos valores que a simulação usa
//! (massa, volume, catálise, flexibilidade, ângulo de repouso, dobra máxima,
//! sensibilidades e condutividades dos sinais, afinidade de substrato,
//! absorção UV). Vive em `assets/aminoacidos.json`; é lida ao arrancar (se o
//! ficheiro não existir, usa-se a cópia embutida no programa, que é o mesmo
//! ficheiro na altura da compilação). O editor local (`crate::editor`) muda-a
//! ao vivo e grava-a no mesmo ficheiro.

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

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
    /// Canais γ e δ (sem eles no ficheiro: junta insensível e condução
    /// igual para os dois lados).
    #[serde(default)]
    pub sens_gama: f32,
    #[serde(default)]
    pub sens_delta: f32,
    #[serde(rename = "cond_gama_N", default = "half")]
    pub cond_gama_n: f32,
    #[serde(rename = "cond_gama_C", default = "half")]
    pub cond_gama_c: f32,
    #[serde(rename = "cond_delta_N", default = "half")]
    pub cond_delta_n: f32,
    #[serde(rename = "cond_delta_C", default = "half")]
    pub cond_delta_c: f32,
    /// Emissão por contacto: canal (0 α, 1 β, 2 γ, 3 δ; −1 = não emite), a
    /// classe de parceiro que procura e a classe deste aminoácido (0 =
    /// nenhuma; 1 ácidos, 2 tiol, 3 aromáticos).
    #[serde(default = "minus_one")]
    pub contacto_canal: f32,
    #[serde(default)]
    pub contacto_quer: f32,
    #[serde(default)]
    pub contacto_classe: f32,
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
    /// Termoestabilidade (0..1): o risco de desnaturação pelo calor é ×
    /// (1 − média do corpo). Os termófilos reais têm mais aminoácidos com
    /// carga (E, K, R) e prolina, e menos glutamina e asparagina.
    #[serde(default)]
    pub termoestabilidade: f32,
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
    /// Custo de manutenção por passo, em múltiplos do de um resíduo.
    pub manutencao: f32,
    /// Arrasto do segmento (na água e no entulho) × isto: órgãos volumosos
    /// (armazenamento) são mais "pesados" a nadar.
    #[serde(default = "one")]
    pub arrasto_mult: f32,
    /// 1 = a manutenção multiplica pela intensidade do órgão (boca, músculo,
    /// armazenamento, fotossistema, protease: mais intensidade = mais
    /// proteína a manter). 0 = sinais (sensores, relógio, relé).
    #[serde(default)]
    pub intensidade_paga: f32,
    /// Ângulo de repouso próprio (rad); sem ele usa o do aminoácido promotor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angulo: Option<f32>,
    /// Condução dos sinais própria [α lado N, α lado C, β lado N, β lado C];
    /// sem ela usa a do aminoácido promotor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conducao: Option<[f32; 4]>,
    /// Condução própria nos canais γ e δ [γ N, γ C, δ N, δ C]; sem ela, γ
    /// usa a de α e δ a de β.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conducao_gd: Option<[f32; 4]>,
    /// As 6 variantes (parâmetro do modificador 0..5): propriedade -> valor
    /// (as propriedades de cada tipo estão em `organs::ORGAN_PROPS`).
    pub variantes: Vec<std::collections::BTreeMap<String, f32>>,
}

fn one() -> f32 {
    1.0
}

fn half() -> f32 {
    0.5
}

fn minus_one() -> f32 {
    -1.0
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
    for r in &rows {
        if r.variantes.len() != super::organs::VARIANTS {
            return Err(format!("{}: {} variantes (devem ser {})", r.nome, r.variantes.len(), super::organs::VARIANTS));
        }
        for (vi, v) in r.variantes.iter().enumerate() {
            for p in super::organs::ORGAN_PROPS[r.tipo as usize] {
                if !v.contains_key(p.name) {
                    return Err(format!("{} variante {vi}: falta \"{}\"", r.nome, p.name));
                }
            }
        }
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

/// Custos de cada variante, em múltiplos dos do tipo (1 se faltarem): uma
/// variante mais forte paga em massa, arrasto, comprimento ou manutenção.
pub const VARIANT_COSTS: [(&str, &str); 4] = [
    ("custo_comprimento", "× comprimento do tipo"),
    ("custo_massa", "× massa do tipo"),
    ("custo_arrasto", "× arrasto do tipo"),
    ("custo_manutencao", "× manutenção do tipo"),
];

/// Propriedades físicas por variante (tipo·VARIANTS + parâmetro): as do tipo
/// × os custos da variante.
/// "Sem valor próprio" nas propriedades de órgão enviadas à GPU.
pub const ORGAN_UNSET: f32 = 1e9;

pub fn organs_to_gpu(rows: &[OrganRow]) -> Vec<crate::params::OrganProps> {
    let mut out = Vec::new();
    for r in rows {
        for v in &r.variantes {
            let c = |k: &str| v.get(k).copied().unwrap_or(1.0);
            out.push(crate::params::OrganProps {
                len_mult: r.comprimento_mult * c("custo_comprimento"),
                mass_mult: r.massa_mult * c("custo_massa"),
                upkeep: r.manutencao * c("custo_manutencao"),
                drag_mult: r.arrasto_mult * c("custo_arrasto"),
                gain_pays: r.intensidade_paga,
                rest_angle: r.angulo.unwrap_or(ORGAN_UNSET),
                cond_alpha_n: r.conducao.map_or(ORGAN_UNSET, |c| c[0]),
                cond_alpha_c: r.conducao.map_or(0.0, |c| c[1]),
                cond_beta_n: r.conducao.map_or(0.0, |c| c[2]),
                cond_beta_c: r.conducao.map_or(0.0, |c| c[3]),
                cond_gamma_n: r.conducao_gd.or(r.conducao).map_or(0.0, |c| c[0]),
                cond_gamma_c: r.conducao_gd.or(r.conducao).map_or(0.0, |c| c[1]),
                cond_delta_n: r.conducao_gd.or(r.conducao).map_or(0.0, |c| c[2]),
                cond_delta_c: r.conducao_gd.or(r.conducao).map_or(0.0, |c| c[3]),
                _pad0: 0.0,
                _pad1: 0.0,
            });
        }
    }
    out
}

/// Variantes para a GPU: tipo·VARIANTS + parâmetro, propriedades pela ordem
/// de `ORGAN_PROPS` (as que faltam ficam a 0).
pub fn variants_to_gpu(rows: &[OrganRow]) -> Vec<crate::params::OrganVariant> {
    let mut out = Vec::new();
    for r in rows {
        for v in &r.variantes {
            let mut p = [0f32; super::organs::MAX_PROPS];
            for (i, d) in super::organs::ORGAN_PROPS[r.tipo as usize].iter().enumerate() {
                p[i] = v.get(d.name).copied().unwrap_or(0.0);
            }
            out.push(crate::params::OrganVariant { p0: p[0], p1: p[1], p2: p[2], p3: p[3], p4: p[4], p5: p[5], p6: p[6], p7: p[7] });
        }
    }
    out
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
            thermo: r.termoestabilidade,
            sens_gamma: r.sens_gama,
            sens_delta: r.sens_delta,
            cond_gamma_n: r.cond_gama_n,
            cond_gamma_c: r.cond_gama_c,
            cond_delta_n: r.cond_delta_n,
            cond_delta_c: r.cond_delta_c,
            contact_channel: r.contacto_canal,
            contact_want: r.contacto_quer,
            contact_class: r.contacto_classe,
        })
        .collect()
}

/// CÓDIGO DOS ÓRGÃOS: promotor (aminoácido) × modificador (aminoácido
/// seguinte) -> (tipo, variante). Um aminoácido é promotor se tiver alguma
/// entrada; uma combinação sem entrada é só o aminoácido (sem órgão).
/// Codões sinónimos dão o mesmo órgão, como na biologia (conta a proteína).
pub type OrganCode = BTreeMap<String, BTreeMap<String, [u32; 2]>>;

pub const CODE_PATH: &str = "assets/codigo_orgaos.json";
const EMBEDDED_CODE: &str = include_str!("../../assets/codigo_orgaos.json");

pub fn parse_code(text: &str) -> Result<OrganCode, String> {
    let code: OrganCode = serde_json::from_str(text).map_err(|e| format!("JSON inválido: {e}"))?;
    let letter = |l: &str| AA_LETTERS.iter().any(|c| c.to_string() == l);
    for (p, row) in &code {
        if !letter(p) {
            return Err(format!("promotor desconhecido: {p}"));
        }
        for (m, [t, v]) in row {
            if !letter(m) {
                return Err(format!("{p}: modificador desconhecido: {m}"));
            }
            if *t as usize >= super::organs::ORGAN_TYPES || *v as usize >= super::organs::VARIANTS {
                return Err(format!("{p}{m}: órgão {t} variante {v} fora da tabela"));
            }
        }
    }
    Ok(code)
}

pub fn load_code() -> (OrganCode, String) {
    match std::fs::read_to_string(CODE_PATH) {
        Ok(text) => match parse_code(&text) {
            Ok(c) => (c, CODE_PATH.to_string()),
            Err(e) => {
                log::error!("{CODE_PATH}: {e}; uso o código embutido");
                (embedded_code(), format!("embutido ({CODE_PATH} inválido)"))
            }
        },
        Err(_) => (embedded_code(), "embutido".to_string()),
    }
}

pub fn embedded_code() -> OrganCode {
    parse_code(EMBEDDED_CODE).expect("assets/codigo_orgaos.json embutido inválido")
}

pub fn save_code(code: &OrganCode) -> Result<(), String> {
    let lines: Vec<String> = code
        .iter()
        .map(|(p, row)| format!("  {}: {}", serde_json::to_string(p).unwrap(), serde_json::to_string(row).unwrap()))
        .collect();
    std::fs::write(CODE_PATH, format!("{{\n{}\n}}\n", lines.join(",\n"))).map_err(|e| format!("{CODE_PATH}: {e}"))
}

/// Para a GPU: 20 × 20 (promotor·20 + modificador, índices de AA_LETTERS);
/// 0 = sem órgão, senão (tipo + 1) | (variante << 4).
pub fn code_to_gpu(code: &OrganCode) -> Vec<u32> {
    let idx = |l: &str| AA_LETTERS.iter().position(|c| c.to_string() == l);
    let mut out = vec![0u32; 400];
    for (p, row) in code {
        for (m, [t, v]) in row {
            if let (Some(pi), Some(mi)) = (idx(p), idx(m)) {
                out[pi * 20 + mi] = (t + 1) | (v << 4);
            }
        }
    }
    out
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
