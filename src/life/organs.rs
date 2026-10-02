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
//! As PROPRIEDADES de cada variante (o parâmetro 0..5) vêm de
//! assets/orgaos.json (ver `ORGAN_PROPS`); o texto abaixo descreve os
//! valores iniciais. Parâmetro (antigo, 3 bits):
//! - sensores (comida, luz, energia): bit 0 canal α/β, bit 1 sinal +/−,
//!   bit 2 nível ou VARIAÇÃO desde o passo anterior. Os de comida e luz
//!   amostram as células num raio: os TOTAIS somam o disco todo; os
//!   DIRECIONAIS dão (lado esquerdo − lado direito) da cadeia;
//! - relógio: canal, período e modulação por α/β vêm das variantes (assets/orgaos.json);
//! - relé: bits 0–1 modo (α->β, β->α, inverte α, inverte β), bit 2 ganho ×2;
//! - boca: catálise ×(2 + p); músculo: resposta ×(2 + p/2);
//!   armazenamento: +4·(p + 1) de capacidade;
//! - fotossistema: bit 0 = energia da luz (0) ou reativar gastos (1);
//! - protease: mordida ×(1 + p/2).
//!
//! Esta é a única fonte de verdade: o shader recebe as constantes geradas.

use super::amino::{AA_LETTERS, STOP, codon};

pub const ORGAN_TYPES: usize = 14;
/// Tipos escolhidos pelo modificador dos promotores C, H, W (índice % 12);
/// a âncora (Y) e o bias (Q) têm promotores próprios.
pub const CODED_ORGAN_TYPES: usize = 12;

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
    Anchor = 12,
    Bias = 13,
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
    "âncora",
    "bias",
];

/// Letras curtas para o inspetor.
pub const ORGAN_SYMBOLS: [char; ORGAN_TYPES] = ['B', 'μ', 'f', 'l', 'e', '◷', 'r', 's', 'ψ', 'Ψ', 'φ', 'ξ', '⚓', 'b'];

/// Descrição em linguagem corrente de um órgão (tipo, parâmetro, índice de
/// intensidade), com o aspeto no ecrã. Espelha a semântica do shader.
/// Propriedade de uma variante de órgão: nome no ficheiro e significado.
pub struct PropDef {
    pub name: &'static str,
    pub desc: &'static str,
}

const fn pd(name: &'static str, desc: &'static str) -> PropDef {
    PropDef { name, desc }
}

/// Variantes por tipo de órgão (o parâmetro do modificador, 0..5).
pub const VARIANTS: usize = 6;
/// Máximo de propriedades por variante (o que cabe na GPU).
pub const MAX_PROPS: usize = 8;

const SENSOR_PROPS: &[PropDef] = &[
    pd("canal", "0 = emite em α, 1 = em β"),
    pd("ganho", "multiplica o que sente (negativo inverte)"),
    pd("modo", "0 = pelo NÍVEL, 1 = pela VARIAÇÃO (quimiotaxia)"),
    pd("memoria", "0..1: na variação, quanto a referência demora a seguir o sentido (0 = passo anterior)"),
];

/// Propriedades de cada tipo de órgão, por ordem (a mesma na GPU).
pub const ORGAN_PROPS: [&[PropDef]; ORGAN_TYPES] = [
    &[pd("forca", "multiplica a catálise do promotor"), pd("vies_AU", "-1..1: prefere A/U (+) ou G/C (−)")],
    &[pd("amplificacao", "multiplica a dobra da junta pelos sinais"), pd("canal", "0 = só α, 1 = só β, 2 = ambos")],
    SENSOR_PROPS,
    SENSOR_PROPS,
    SENSOR_PROPS,
    &[
        pd("canal", "0 = emite em α, 1 = em β"),
        pd("periodo", "passos por ciclo"),
        pd("mod_alfa", "o relógio acelera (+) ou abranda (−) com o nível de α"),
        pd("mod_beta", "o mesmo com β"),
    ],
    &[
        pd("entrada", "0 = lê α, 1 = lê β"),
        pd("saida", "0 = emite em α, 1 = em β"),
        pd("ganho", "multiplica (negativo inverte)"),
        pd("limiar", "só passa o que estiver acima deste nível (porta)"),
    ],
    &[pd("capacidade", "energia extra que guarda")],
    SENSOR_PROPS,
    SENSOR_PROPS,
    &[pd("reciclar", "0..1: fração da luz usada para reativar gastos (o resto dá energia)"), pd("eficiencia", "multiplica o rendimento")],
    &[pd("forca", "multiplica a mordida"), pd("alcance", "unidades do mundo além do contacto")],
    &[
        pd("polaridade", "+1 ou −1: liga-se a âncoras de polaridade oposta de outros agentes"),
        pd("quebra", "probabilidade por passo de se soltar (0 = permanente)"),
    ],
    &[pd("canal", "0 = emite em α, 1 = em β"), pd("valor", "sinal constante emitido (× intensidade)")],
];

fn fmt_canal(v: f32) -> &'static str {
    if v < 0.5 { "α" } else { "β" }
}

/// Descrição em linguagem corrente de um órgão (tipo, variante, índice de
/// intensidade) com os valores da tabela.
pub fn describe(t: u8, p: u8, gain_idx: u8, table: &[super::table::OrganRow]) -> String {
    let g = organ_gain(gain_idx);
    let t = t as usize;
    let Some(row) = table.get(t) else { return format!("órgão {t}") };
    let v = |name: &str| row.variantes.get(p as usize).and_then(|m| m.get(name)).copied().unwrap_or(0.0);
    let sensor = |o_que: &str, aspeto: &str| {
        format!(
            "{o_que} [{aspeto}]: emite em {}, ganho ×{:.2}, {}",
            fmt_canal(v("canal")),
            v("ganho") * g,
            if v("modo") < 0.5 {
                "pelo NÍVEL".to_string()
            } else {
                format!("pela VARIAÇÃO (memória {:.2})", v("memoria"))
            }
        )
    };
    match t {
        0 => format!(
            "boca [disco com abertura escura]: come monómeros ativados, força ×{:.2}, {}",
            v("forca") * g,
            match v("vies_AU") {
                x if x > 0.05 => format!("prefere A/U ({x:+.2})"),
                x if x < -0.05 => format!("prefere G/C ({x:+.2})"),
                _ => "sem preferência extra".into(),
            }
        ),
        1 => format!(
            "músculo [elipse às riscas]: dobra ×{:.2} com {}",
            v("amplificacao") * g,
            ["α", "β", "α e β"][(v("canal").round().clamp(0.0, 2.0)) as usize]
        ),
        2 => sensor("sensor de comida TOTAL", "coroa de antenas verdes"),
        3 => sensor("sensor de luz TOTAL", "coroa de antenas amarelas"),
        4 => sensor("sensor de energia interna", "disco com anel dourado"),
        5 => format!(
            "relógio [mostrador]: em {}, período {:.0} passos, força ×{g:.2}{}",
            fmt_canal(v("canal")),
            v("periodo"),
            if v("mod_alfa").abs() + v("mod_beta").abs() > 0.0 {
                format!("; acelera com α ×{:+.2} e com β ×{:+.2}", v("mod_alfa"), v("mod_beta"))
            } else {
                String::new()
            }
        ),
        6 => format!(
            "relé [losango]: lê {} e emite em {}, ganho ×{:.2}{}",
            fmt_canal(v("entrada")),
            fmt_canal(v("saida")),
            v("ganho") * g,
            if v("limiar") > 0.0 { format!(", só acima de {:.2}", v("limiar")) } else { String::new() }
        ),
        7 => format!("armazenamento [disco com anéis]: +{:.1} de capacidade de energia", v("capacidade") * g),
        8 => sensor("sensor de comida DIRECIONAL (esquerda − direita)", "2 antenas verdes"),
        9 => sensor("sensor de luz DIRECIONAL (esquerda − direita)", "2 antenas amarelas"),
        10 => format!(
            "fotossistema [disco verde com raios]: {:.0}% da luz para reativar gastos, {:.0}% para energia, eficiência ×{:.2}",
            v("reciclar") * 100.0,
            (1.0 - v("reciclar")) * 100.0,
            v("eficiencia") * g
        ),
        12 => format!(
            "âncora {} [anel {}]: liga-se a uma âncora {} de outro agente que toque ou de um filho; {}",
            if v("polaridade") >= 0.0 { "+" } else { "−" },
            if v("polaridade") >= 0.0 { "vermelho" } else { "azul" },
            if v("polaridade") >= 0.0 { "−" } else { "+" },
            if v("quebra") <= 0.0 {
                "permanente (só se solta se esticar demais)".to_string()
            } else {
                format!("solta-se em média ao fim de {:.0} passos", 1.0 / v("quebra"))
            }
        ),
        13 => format!(
            "bias [ponto {}]: emite sempre {:+.2} em {}",
            if v("canal") < 0.5 { "laranja" } else { "verde" },
            v("valor") * g,
            fmt_canal(v("canal"))
        ),
        _ => format!(
            "protease [disco com dentes]: tira energia a quem toca (fica com metade), força ×{:.2}, alcance +{:.0}; a prolina protege",
            v("forca") * g,
            v("alcance")
        ),
    }
}

/// Aminoácidos promotores (pouco frequentes num genoma ao acaso).
pub const PROMOTERS: [char; 3] = ['C', 'H', 'W'];
/// Promotor da âncora: tirosina (as colas dos mexilhões são proteínas ricas
/// em DOPA, que vem da tirosina). O modificador escolhe a variante (% 6).
pub const ANCHOR_PROMOTER: char = 'Y';
/// Promotor do bias (emissor constante de α ou β): glutamina. O modificador
/// escolhe a variante (% 6).
pub const BIAS_PROMOTER: char = 'Q';

fn aa_index(l: char) -> u8 {
    AA_LETTERS.iter().position(|&x| x == l).unwrap() as u8
}

pub fn is_promoter(aa: u8) -> bool {
    PROMOTERS.iter().any(|&l| aa_index(l) == aa) || aa == aa_index(ANCHOR_PROMOTER) || aa == aa_index(BIAS_PROMOTER)
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
                let organ = if aa == aa_index(ANCHOR_PROMOTER) {
                    (Organ::Anchor as u8, idx % VARIANTS as u8, gain)
                } else if aa == aa_index(BIAS_PROMOTER) {
                    (Organ::Bias as u8, idx % VARIANTS as u8, gain)
                } else {
                    (idx % CODED_ORGAN_TYPES as u8, idx / CODED_ORGAN_TYPES as u8, gain)
                };
                body.push(Residue { aa, organ: Some(organ) });
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
        "ANCHOR",
        "BIAS",
    ];
    for (i, name) in names.iter().enumerate() {
        s += &format!("const ORGAN_{name}: u32 = {i}u;\n");
    }
    s += &format!(
        "const ORGAN_TYPES: u32 = {ORGAN_TYPES}u;\nconst CODED_ORGAN_TYPES: u32 = {CODED_ORGAN_TYPES}u;\nconst ORGAN_VARIANTS: u32 = {VARIANTS}u;\n"
    );
    // 1 = promotor dos órgãos codificados (C, H, W), 2 = da âncora (Y), 3 = do bias (Q).
    let promo: Vec<String> = (0..20u8)
        .map(|a| {
            let v = if a == aa_index(ANCHOR_PROMOTER) {
                2
            } else if a == aa_index(BIAS_PROMOTER) {
                3
            } else {
                is_promoter(a) as u32
            };
            format!("{v}u")
        })
        .collect();
    s += &format!("const AA_IS_PROMOTER = array<u32, 20>({});\n", promo.join(", "));
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
    fn tyrosine_makes_an_anchor() {
        // AUG | UAU (Y) + AAU (modificador 1 -> variante 1) | UAA
        let g = bases("AUGUAUAAUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body[1].organ, Some((Organ::Anchor as u8, 1, GAIN_DEFAULT)));
    }

    #[test]
    fn glutamine_makes_a_bias() {
        // AUG | CAA (Q) + AAG (modificador 2 -> variante 2) | UAA
        let g = bases("AUGCAAAAGUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body[1].organ, Some((Organ::Bias as u8, 2, GAIN_DEFAULT)));
    }

    #[test]
    fn promoter_before_stop_is_a_plain_residue() {
        // AUG | CAU (H, promotor) | UAA (stop): sem órgão; o stop termina a cadeia.
        let g = bases("AUGCAUUAA");
        let body = translate_organs(&g, true);
        assert_eq!(body, vec![Residue { aa: aa_index('M'), organ: None }, Residue { aa: aa_index('H'), organ: None }]);
    }
}
