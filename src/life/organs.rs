//! Órgãos: "super-aminoácidos" com uma função simples.
//!
//! Codificação: um PROMOTOR (aminoácidos C, H ou W) seguido de um codão
//! MODIFICADOR que não seja stop forma um órgão, que ocupa UMA posição do
//! corpo e gasta 6 bases. O modificador (índice do codão 0..63, ordem A U G C)
//! define o tipo (modificador % 8) e o parâmetro (modificador / 8, 0..7).
//! Fisicamente (massa, dobragem MJ, catálise) o órgão continua a ser o
//! aminoácido promotor; o órgão acrescenta-lhe uma função.
//!
//! Os sinais internos são dois canais (α, β) por resíduo, conduzidos N->C.
//! Todas as juntas dobram conforme α e β (sensibilidade por aminoácido); o
//! "músculo" amplifica a resposta local; a natação faz-se pelo RFT.
//!
//! Parâmetro (3 bits):
//! - sensores (comida, luz, energia): bit 0 canal α/β, bit 1 sinal +/−,
//!   bit 2 nível ou VARIAÇÃO desde o passo anterior;
//! - relógio: bit 0 canal, bits 1–2 período (20, 40, 80, 160 passos);
//! - relé: bits 0–1 modo (α->β, β->α, inverte α, inverte β), bit 2 ganho ×2;
//! - boca: catálise ×(2 + p); músculo: resposta ×(2 + p/2);
//!   armazenamento: +4·(p + 1) de capacidade.
//!
//! Esta é a única fonte de verdade: o shader recebe as constantes geradas.

use super::amino::{AA_LETTERS, STOP, codon};

pub const ORGAN_TYPES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Organ {
    Mouth = 0,
    Muscle = 1,
    FoodSensor = 2,
    LightSensor = 3,
    EnergySensor = 4,
    Clock = 5,
    Relay = 6,
    Storage = 7,
}

/// Nota: TODAS as juntas respondem aos sinais α/β (sensibilidade por
/// aminoácido); o "músculo" é um amplificador dessa resposta local.
pub const ORGAN_NAMES: [&str; ORGAN_TYPES] = [
    "boca",
    "músculo (amplificador)",
    "sensor de comida",
    "sensor de luz",
    "sensor de energia",
    "relógio",
    "relé",
    "armazenamento",
];

/// Letras curtas para o inspetor.
pub const ORGAN_SYMBOLS: [char; ORGAN_TYPES] = ['B', 'μ', 'f', 'l', 'e', '◷', 'r', 's'];

/// Custo de manutenção por passo de um órgão, em múltiplos do custo de um resíduo.
pub const ORGAN_UPKEEP: [f32; ORGAN_TYPES] = [3.0, 4.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0];

/// Aminoácidos promotores (pouco frequentes num genoma ao acaso).
pub const PROMOTERS: [char; 3] = ['C', 'H', 'W'];

fn aa_index(l: char) -> u8 {
    AA_LETTERS.iter().position(|&x| x == l).unwrap() as u8
}

pub fn is_promoter(aa: u8) -> bool {
    PROMOTERS.iter().any(|&l| aa_index(l) == aa)
}

/// Um resíduo do corpo: aminoácido e, opcionalmente, órgão (tipo, parâmetro).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residue {
    pub aa: u8,
    pub organ: Option<(u8, u8)>,
}

/// Byte de órgão guardado na GPU: 0 = nenhum; senão (tipo + 1) | (param << 4).
pub fn organ_byte(r: &Residue) -> u8 {
    r.organ.map_or(0, |(t, p)| (t + 1) | (p << 4))
}

/// Tradução com órgãos (espelho exato do shader): a partir do primeiro AUG
/// (`require_start`) ou da base 0, até ao primeiro stop ou 64 resíduos.
pub fn translate_organs(genome: &[u8], require_start: bool) -> Vec<Residue> {
    let mut i = if require_start {
        let Some(s) = genome.windows(3).position(|w| w == [0, 1, 2]) else { return Vec::new() };
        s
    } else {
        0
    };
    let mut body = Vec::new();
    while i + 3 <= genome.len() && body.len() < super::amino::MAX_BODY {
        let aa = codon(genome[i], genome[i + 1], genome[i + 2]);
        if aa == STOP {
            break;
        }
        // Promotor seguido de um modificador que não é stop: órgão.
        if is_promoter(aa) && i + 6 <= genome.len() {
            let m = (genome[i + 3], genome[i + 4], genome[i + 5]);
            if codon(m.0, m.1, m.2) != STOP {
                let idx = m.0 * 16 + m.1 * 4 + m.2;
                body.push(Residue { aa, organ: Some((idx % ORGAN_TYPES as u8, idx / ORGAN_TYPES as u8)) });
                i += 6;
                continue;
            }
        }
        body.push(Residue { aa, organ: None });
        i += 3;
    }
    body
}

/// Constantes WGSL dos órgãos.
pub fn wgsl() -> String {
    let mut s = String::from("// ---- órgãos (gerado de src/life/organs.rs) ----\n");
    for (i, name) in ["MOUTH", "MUSCLE", "FOOD_SENSOR", "LIGHT_SENSOR", "ENERGY_SENSOR", "CLOCK", "RELAY", "STORAGE"]
        .iter()
        .enumerate()
    {
        s += &format!("const ORGAN_{name}: u32 = {i}u;\n");
    }
    let promo: Vec<String> = (0..20u8).map(|a| format!("{}u", is_promoter(a) as u32)).collect();
    s += &format!("const AA_IS_PROMOTER = array<u32, 20>({});\n", promo.join(", "));
    let up: Vec<String> = ORGAN_UPKEEP.iter().map(|v| format!("{v:.3}")).collect();
    s += &format!("const ORGAN_UPKEEP = array<f32, {ORGAN_TYPES}>({});\n", up.join(", "));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bases(s: &str) -> Vec<u8> {
        s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
    }

    #[test]
    fn promoter_plus_modifier_makes_an_organ() {
        // AUG (M) | UGU (C, promotor) + GCA (modificador) | UUU (F) | UAA (stop)
        let g = bases("AUGUGUGCAUUUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body.len(), 3);
        assert_eq!(body[0].organ, None);
        // GCA: G=2, C=3, A=0 -> 2*16 + 3*4 + 0 = 44 -> tipo 44 % 8 = 4, param 44 / 8 = 5.
        assert_eq!(body[1], Residue { aa: aa_index('C'), organ: Some((4, 5)) });
        assert_eq!(body[2].organ, None);
    }

    #[test]
    fn promoter_before_stop_is_a_plain_residue() {
        // AUG | CAU (H, promotor) | UAA (stop): sem órgão; o stop termina a cadeia.
        let g = bases("AUGCAUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body, vec![Residue { aa: aa_index('M'), organ: None }, Residue { aa: aa_index('H'), organ: None }]);
    }
}
