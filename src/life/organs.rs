//! Órgãos: "super-aminoácidos" com uma função simples.
//!
//! Codificação: um PROMOTOR (aminoácidos C, H ou W) seguido de um codão
//! MODIFICADOR que não seja stop forma um órgão, que ocupa UMA posição do
//! corpo. O modificador (índice do codão 0..63, ordem A U G C) define o tipo
//! (modificador % 12) e o parâmetro (modificador / 12, 0..5). Um SEGUNDO
//! modificador (se não for stop) dá a INTENSIDADE: 64 níveis logarítmicos,
//! ganho = 2^((índice − 32)/8), de ×0,06 a ×15 (9 bases no total); sem ele o
//! ganho é 1 (6 bases). A intensidade multiplica a emissão dos sensores,
//! relógios e relés e a amplificação do músculo.
//! Fisicamente (massa, ângulo de repouso, catálise) o órgão continua a ser o
//! aminoácido promotor; o órgão acrescenta-lhe uma função.
//!
//! Os sinais internos são dois canais (α, β) por resíduo, conduzidos entre
//! vizinhos com a condutividade de cada aminoácido (tabela `assets/aminoacidos.json`).
//! Todas as juntas dobram conforme α e β (sensibilidade por aminoácido); o
//! "músculo" amplifica a resposta local; a natação faz-se pelo RFT.
//!
//! Parâmetro (3 bits):
//! - sensores (comida, luz, energia): bit 0 canal α/β, bit 1 sinal +/−,
//!   bit 2 nível ou VARIAÇÃO desde o passo anterior. Os de comida e luz
//!   amostram as células num raio: os TOTAIS somam o disco todo; os
//!   DIRECIONAIS dão (lado esquerdo − lado direito) da cadeia;
//! - relógio: bit 0 canal, bits 1–2 período (20, 40, 80, 160 passos);
//! - relé: bits 0–1 modo (α->β, β->α, inverte α, inverte β), bit 2 ganho ×2;
//! - boca: catálise ×(2 + p); músculo: resposta ×(2 + p/2);
//!   armazenamento: +4·(p + 1) de capacidade;
//! - fotossistema: bit 0 = energia da luz (0) ou reativar gastos (1);
//! - protease: mordida ×(1 + p/2).
//!
//! Esta é a única fonte de verdade: o shader recebe as constantes geradas.

use super::amino::{AA_LETTERS, STOP, codon};

pub const ORGAN_TYPES: usize = 12;

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
    FoodSensorDirectional = 8,
    LightSensorDirectional = 9,
    Photosystem = 10,
    Protease = 11,
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
    "sensor de comida direcional",
    "sensor de luz direcional",
    "fotossistema",
    "protease",
];

/// Letras curtas para o inspetor.
pub const ORGAN_SYMBOLS: [char; ORGAN_TYPES] = ['B', 'μ', 'f', 'l', 'e', '◷', 'r', 's', 'ψ', 'Ψ', 'φ', 'ξ'];

/// Descrição em linguagem corrente de um órgão (tipo, parâmetro, índice de
/// intensidade), com o aspeto no ecrã. Espelha a semântica do shader.
pub fn describe(t: u8, p: u8, gain_idx: u8) -> String {
    let g = organ_gain(gain_idx);
    let canal = if p & 1 == 0 { "α" } else { "β" };
    let sensor = |o_que: &str, aspeto: &str| {
        format!(
            "{o_que} [{aspeto}]: emite em {canal}, {}, {}, força ×{g:.2}",
            if p & 2 == 0 { "positivo" } else { "invertido (negativo)" },
            if p & 4 == 0 { "pelo NÍVEL" } else { "pela VARIAÇÃO desde o passo anterior" },
        )
    };
    match t {
        0 => format!("boca [disco com abertura escura]: come monómeros ativados (só as bocas comem), força ×{}", 2 + p as u32),
        1 => format!(
            "músculo [elipse vermelha às riscas]: a junta dobra ×{:.2} mais com os sinais",
            (2.0 + 0.5 * p as f32) * g
        ),
        2 => sensor("sensor de comida TOTAL, mede os ativados num raio à volta", "coroa de 6 antenas verdes"),
        3 => sensor("sensor de luz TOTAL, mede a luz num raio à volta", "coroa de 6 antenas amarelas"),
        4 => sensor("sensor de energia, mede a energia interna", "disco com anel dourado"),
        5 => format!(
            "relógio [mostrador com ponteiro]: oscila em {canal} com período {} passos, força ×{g:.2}",
            CLOCK_PERIOD_BASE as u32 * (1 << (p >> 1))
        ),
        6 => {
            let modo = ["converte α em β", "converte β em α", "inverte α", "inverte β"][(p & 3) as usize];
            let k = if p & 4 != 0 { 2.0 } else { 1.0 };
            format!("relé [losango]: {modo}, força ×{:.2}", k * g)
        }
        7 => format!("armazenamento [disco com anéis]: +{} de capacidade de energia", 4 * (p as u32 + 1)),
        10 => format!(
            "fotossistema [disco verde com raios]: {}, força ×{g:.2}",
            if p & 1 == 0 {
                "dá energia com a luz (produtor)"
            } else {
                "usa a luz para reativar os gastos à volta (faz comida; a mesma energia que o modo produtor)"
            }
        ),
        11 => format!(
            "protease [disco com dentes]: ao tocar noutro agente tira-lhe energia (fica com metade), força ×{:.2}; corpos ricos em prolina resistem",
            (1.0 + 0.5 * p as f32) * g
        ),
        8 => sensor(
            "sensor de comida DIRECIONAL, compara o lado esquerdo com o direito",
            "2 antenas verdes, uma de cada lado",
        ),
        _ => sensor(
            "sensor de luz DIRECIONAL, compara o lado esquerdo com o direito",
            "2 antenas amarelas, uma de cada lado",
        ),
    }
}

/// Custo de manutenção por passo de um órgão, em múltiplos do custo de um resíduo.
pub const ORGAN_UPKEEP: [f32; ORGAN_TYPES] = [3.0, 4.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 3.0, 2.0, 3.0];

/// Período base do relógio (passos); o parâmetro multiplica-o por 2^(bits 1–2).
pub const CLOCK_PERIOD_BASE: f32 = 20.0;

/// Aminoácidos promotores (pouco frequentes num genoma ao acaso).
pub const PROMOTERS: [char; 3] = ['C', 'H', 'W'];

fn aa_index(l: char) -> u8 {
    AA_LETTERS.iter().position(|&x| x == l).unwrap() as u8
}

pub fn is_promoter(aa: u8) -> bool {
    PROMOTERS.iter().any(|&l| aa_index(l) == aa)
}

/// Índice de intensidade por omissão (ganho 1).
pub const GAIN_DEFAULT: u8 = 32;

/// Ganho de um índice de intensidade (0..63).
pub fn organ_gain(idx: u8) -> f32 {
    2f32.powf((idx as f32 - 32.0) / 8.0)
}

/// Um resíduo do corpo: aminoácido e, opcionalmente, órgão (tipo, parâmetro, intensidade).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residue {
    pub aa: u8,
    pub organ: Option<(u8, u8, u8)>,
}

/// Código de órgão guardado na GPU (16 bits): 0 = nenhum; senão
/// (tipo + 1) | (parâmetro << 4) | (intensidade << 8).
pub fn organ_code(r: &Residue) -> u16 {
    r.organ.map_or(0, |(t, p, g)| (t as u16 + 1) | ((p as u16) << 4) | ((g as u16) << 8))
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
                // Segundo modificador (intensidade), se existir e não for stop.
                let (gain, used) =
                    if i + 9 <= genome.len() && codon(genome[i + 6], genome[i + 7], genome[i + 8]) != STOP {
                        (genome[i + 6] * 16 + genome[i + 7] * 4 + genome[i + 8], 9)
                    } else {
                        (GAIN_DEFAULT, 6)
                    };
                body.push(Residue { aa, organ: Some((idx % ORGAN_TYPES as u8, idx / ORGAN_TYPES as u8, gain)) });
                i += used;
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
    let names = [
        "MOUTH",
        "MUSCLE",
        "FOOD_SENSOR",
        "LIGHT_SENSOR",
        "ENERGY_SENSOR",
        "CLOCK",
        "RELAY",
        "STORAGE",
        "FOOD_SENSOR_DIR",
        "LIGHT_SENSOR_DIR",
        "PHOTOSYSTEM",
        "PROTEASE",
    ];
    for (i, name) in names.iter().enumerate() {
        s += &format!("const ORGAN_{name}: u32 = {i}u;\n");
    }
    s += &format!("const ORGAN_TYPES: u32 = {ORGAN_TYPES}u;\n");
    s += &format!("const CLOCK_PERIOD_BASE: f32 = {CLOCK_PERIOD_BASE:.1};\n");
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
        // AUG (M) | UGU (C, promotor) + GCA (modificador) + UUU (intensidade) | GGU (G) | UAA
        let g = bases("AUGUGUGCAUUUGGUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body.len(), 3);
        assert_eq!(body[0].organ, None);
        // GCA: 2*16 + 3*4 + 0 = 44 -> tipo 44 % 12 = 8, param 3; UUU: 1*16 + 1*4 + 1 = 21 -> intensidade 21.
        assert_eq!(body[1], Residue { aa: aa_index('C'), organ: Some((8, 3, 21)) });
        assert_eq!(body[2].organ, None);
        // Sem segundo modificador (stop a seguir): intensidade por omissão, 6 bases.
        let g2 = bases("AUGUGUGCAUAA");
        let b2 = translate_organs(&g2, true);
        assert_eq!(b2[1].organ, Some((8, 3, GAIN_DEFAULT)));
        assert!((organ_gain(GAIN_DEFAULT) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn promoter_before_stop_is_a_plain_residue() {
        // AUG | CAU (H, promotor) | UAA (stop): sem órgão; o stop termina a cadeia.
        let g = bases("AUGCAUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body, vec![Residue { aa: aa_index('M'), organ: None }, Residue { aa: aa_index('H'), organ: None }]);
    }
}
