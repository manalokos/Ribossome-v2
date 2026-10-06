//! Órgãos: "super-aminoácidos" com uma função simples.
//!
//! Codificação (`assets/codigo_orgaos.json`, editável na página): um
//! aminoácido PROMOTOR seguido de um aminoácido MODIFICADOR forma um órgão se
//! a tabela tiver o par (promotor, modificador) -> (tipo, variante); o órgão
//! ocupa UMA posição do corpo. Codões sinónimos dão o mesmo órgão (conta a
//! proteína, como na biologia). Um SEGUNDO codão (se não for stop) dá a
//! INTENSIDADE: 64 níveis logarítmicos, ganho = 2^((índice − 32)/8), de
//! ×0,06 a ×15 (9 bases no total); sem ele o ganho é 1 (6 bases).
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
//!   armazenamento: DESATIVADO (a capacidade de energia vem do volume dos
//!   aminoácidos do corpo; o tipo fica só para ler cenas antigas);
//! - fotossistema: bit 0 = energia da luz (0) ou reativar gastos (1);
//! - protease: família (o que corta), força e canal que a ativa;
//! - bias de idade: sinal que decai com a idade.
//!
//! Esta é a única fonte de verdade: o shader recebe as constantes geradas.

use super::amino::{STOP, codon};

pub const ORGAN_TYPES: usize = 19;

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
    Chemosynthesis = 14,
    Proofreading = 15,
    Dormancy = 16,
    AgeBias = 17,
    Holdfast = 18,
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
    "quimiossíntese",
    "revisão (menos mutações)",
    "dormência (metabolismo lento)",
    "bias de idade",
    "ventosa (fixa-se ao terreno)",
];

/// Letras curtas para o inspetor.
pub const ORGAN_SYMBOLS: [char; ORGAN_TYPES] = ['B', 'μ', 'f', 'l', 'e', '◷', 'r', 's', 'ψ', 'Ψ', 'φ', 'ξ', '⚓', 'b', 'χ', 'π', 'z', 'j', 'v'];

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
    pd("ganho", "multiplica o que sente (negativo inverte). O sensor já devolve a ocupação do recetor (0..1) ou, no direcional, o contraste relativo (−1..1)"),
    pd("modo", "0 = pelo NÍVEL, 1 = pela VARIAÇÃO (quimiotaxia)"),
    pd("memoria", "0..1: fração da carga do sensor que fica em cada passo (descarga = 1 − isto; ~1/(1 − isto) passos). No nível é a própria leitura; na variação é a referência lenta"),
];

/// Sensores "de luz" (físicos): as mesmas propriedades e o que sentem.
const LIGHT_SENSOR_PROPS: &[PropDef] = &[
    pd("canal", "0 = emite em α, 1 = em β"),
    pd("ganho", "multiplica o que sente (negativo inverte). O sensor já devolve a ocupação do recetor (0..1) ou, no direcional, o contraste relativo (−1..1)"),
    pd("modo", "0 = pelo NÍVEL, 1 = pela VARIAÇÃO (quimiotaxia)"),
    pd("memoria", "0..1: fração da carga do sensor que fica em cada passo (descarga = 1 − isto; ~1/(1 − isto) passos). No nível é a própria leitura; na variação é a referência lenta"),
    pd("alvo", "0 = luz, 1 = temperatura, 2 = redutor das fumarolas, 3 = terreno (grãos)"),
];

/// Sensores "de comida": as mesmas propriedades e o que sentem.
const FOOD_SENSOR_PROPS: &[PropDef] = &[
    pd("canal", "0 = emite em α, 1 = em β"),
    pd("ganho", "multiplica o que sente (negativo inverte). O sensor já devolve a ocupação do recetor (0..1) ou, no direcional, o contraste relativo (−1..1)"),
    pd("modo", "0 = pelo NÍVEL, 1 = pela VARIAÇÃO (quimiotaxia)"),
    pd("memoria", "0..1: fração da carga do sensor que fica em cada passo (descarga = 1 − isto; ~1/(1 − isto) passos). No nível é a própria leitura; na variação é a referência lenta"),
    pd("alvo", "0 = comida (ativados; os canais pesados pelas afinidades do aminoácido SEGUINTE da cadeia, a antena), 1 = gastos (já sem variantes), 2 = corpos de outros agentes"),
];

/// Propriedades de cada tipo de órgão, por ordem (a mesma na GPU).
pub const ORGAN_PROPS: [&[PropDef]; ORGAN_TYPES] = [
    &[
        pd("forca", "multiplica a catálise do promotor"),
        pd("vies_AU", "-1..1: prefere A/U (+) ou G/C (−)"),
        pd("fecha", "−1 = sempre aberta; 2 = fecha com sinal γ positivo; 3 = fecha com sinal δ positivo (fechada não come nem deixa fugir energia)"),
    ],
    &[pd("amplificacao", "multiplica a dobra da junta pelos sinais"), pd("canal", "0 = só α, 1 = só β, 2 = ambos")],
    FOOD_SENSOR_PROPS,
    LIGHT_SENSOR_PROPS,
    SENSOR_PROPS,
    &[
        pd("canal", "0 = emite em α, 1 = em β"),
        pd("periodo", "passos por ciclo"),
        pd("mod_alfa", "o relógio acelera (+) ou abranda (−) com o nível de α"),
        pd("mod_beta", "o mesmo com β"),
    ],
    &[
        pd("funcao", "0 = SWITCH (passa o sinal do canal de entrada para o de saída e trava a entrada aqui), 1 = cópia (emite na saída, a entrada segue), 2 = inversor (emite o simétrico), 3 = GATE que fecha (entrada acima do limiar: o canal de saída não passa aqui), 4 = GATE que abre (o canal de saída só passa aqui com a entrada acima do limiar), 5 = limiar (emite só a parte da entrada acima do limiar)"),
        pd("ganho", "multiplica o que emite (× a força do 3.º codão)"),
        pd("limiar", "limiar das portas e do modo 5 (módulo do sinal de entrada)"),
    ],
    &[pd("capacidade", "sem efeito: o órgão está desativado (a capacidade vem do volume dos aminoácidos)")],
    FOOD_SENSOR_PROPS,
    LIGHT_SENSOR_PROPS,
    &[pd("reciclar", "0..1: fração da luz usada para reativar gastos (o resto dá energia)"), pd("eficiencia", "multiplica o rendimento")],
    &[
        pd("familia", "o que corta na vítima: 1 = lisina e arginina, 2 = aspartato e asparagina, 3 = fenilalanina, tirosina, triptofano e leucina (coluna 'protease: alvo' dos aminoácidos)"),
        pd("forca", "multiplica o risco de lise que causa (× intensidade)"),
        pd("canal", "−1 = sempre ativa; 2 = só com sinal γ positivo; 3 = só com sinal δ positivo (proporcional ao sinal, até 1)"),
    ],
    &[
        pd("polaridade", "+1 ou −1: liga-se a âncoras de polaridade oposta de outros agentes"),
        pd("quebra", "probabilidade por passo de se soltar (0 = permanente)"),
    ],
    &[pd("canal", "0 = emite em α, 1 = em β"), pd("valor", "sinal constante emitido (× intensidade)")],
    &[
        pd("reciclar", "0..1: fração do redutor usada para reativar gastos (o resto dá energia)"),
        pd("eficiencia", "multiplica o que consome"),
    ],
    &[pd("protecao", "divide a taxa de mutação das cópias deste agente por 1 + a soma das proteções (× intensidade)")],
    &[
        pd("fator", "o metabolismo do agente (manutenção, comer, quimiossíntese e copiar, tudo junto) é multiplicado por isto quando o órgão está a atuar em pleno"),
        pd("canal", "−1 = atua sempre; 2 = só com sinal γ positivo; 3 = só com sinal δ positivo (proporcional ao sinal, até 1)"),
    ],
    &[
        pd("canal", "0 = α, 1 = β"),
        pd("valor", "sinal emitido ao nascer (× intensidade)"),
        pd("meia_vida", "passos de vida até o sinal cair para metade"),
    ],
    &[
        pd("forca", "arrasto extra deste resíduo quando toca em entulho ou rocha (× intensidade): prende-o ao sítio"),
        pd("larga", "−1 = agarra sempre; 2 = larga com sinal γ positivo; 3 = larga com sinal δ positivo"),
        pd("emite", "canal do sinal que emite enquanto está agarrada (2 = γ, 3 = δ; −1 = nenhum)"),
    ],
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
    let alvo = || match v("alvo").round() as i32 {
        1 => "GASTOS (rasto de quem come)",
        2 => "CORPOS de outros agentes",
        _ => "comida (ativados)",
    };
    let alvo_fisico = || match v("alvo").round() as i32 {
        1 => "TEMPERATURA",
        2 => "REDUTOR das fumarolas",
        3 => "TERRENO (grãos: entulho e rocha)",
        _ => "luz",
    };
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
            "boca [disco com abertura escura]: come monómeros ativados, força ×{:.2}, {}, {}",
            v("forca") * g,
            match v("vies_AU") {
                x if x > 0.05 => format!("prefere A/U ({x:+.2})"),
                x if x < -0.05 => format!("prefere G/C ({x:+.2})"),
                _ => "sem preferência extra".into(),
            },
            match v("fecha") {
                c if c < 0.0 => "sempre aberta",
                c if c < 2.5 => "fecha com sinal γ positivo",
                _ => "fecha com sinal δ positivo",
            }
        ),
        1 => format!(
            "músculo [elipse às riscas]: dobra ×{:.2} com {}",
            v("amplificacao") * g,
            ["α", "β", "α e β"][(v("canal").round().clamp(0.0, 2.0)) as usize]
        ),
        2 => format!("{} · sente {}", sensor("sensor de comida TOTAL", "coroa de antenas verdes"), alvo()),
        3 => format!("{} · sente {}", sensor("sensor físico TOTAL", "coroa de antenas amarelas"), alvo_fisico()),
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
        6 => {
            // Os canais vêm do 3.º codão (gain_idx): entrada = bits 0–1,
            // saída = bits 2–3, força = bits 4–5.
            let (cin, cout) = (RELAY_CHANNELS[(gain_idx & 3) as usize], RELAY_CHANNELS[((gain_idx >> 2) & 3) as usize]);
            let mag = RELAY_MAG[((gain_idx >> 4) & 3) as usize] * v("ganho");
            match v("funcao").round() as i32 {
                0 => format!("relé SWITCH [losango]: passa {cin} para {cout} (×{mag:.2}) e trava {cin} aqui"),
                1 => format!("relé cópia [losango]: lê {cin} e emite em {cout} (×{mag:.2})"),
                2 => format!("relé inversor [losango]: lê {cin} e emite −{cin} em {cout} (×{mag:.2})"),
                3 => format!("relé GATE [losango]: com |{cin}| > {:.2}, {cout} não passa aqui", v("limiar")),
                4 => format!("relé GATE [losango]: {cout} só passa aqui com |{cin}| > {:.2}", v("limiar")),
                _ => format!("relé limiar [losango]: emite em {cout} a parte de {cin} acima de {:.2} (×{mag:.2})", v("limiar")),
            }
        }
        15 => format!("revisão [escudo]: taxa de mutação das cópias ÷ (1 + {:.1})", v("protecao") * g),
        16 => format!(
            "dormência [lua]: metabolismo × {:.2} (come, copia e gasta mais devagar), {}",
            v("fator").clamp(0.01, 1.0).powf(g),
            match v("canal") {
                c if c < 0.0 => "sempre".to_string(),
                c if c < 2.5 => "só com sinal γ positivo".to_string(),
                _ => "só com sinal δ positivo".to_string(),
            }
        ),
        11 => format!(
            "protease [disco com dentes]: desfaz quem toca (corta {}), força ×{:.2}, {}; a prolina protege",
            match v("familia") {
                f if f < 1.5 => "lisina e arginina",
                f if f < 2.5 => "aspartato e asparagina",
                _ => "aromáticos e leucina",
            },
            v("forca") * g,
            match v("canal") {
                c if c < 0.0 => "sempre ativa".to_string(),
                c if c < 2.5 => "só com sinal γ positivo".to_string(),
                _ => "só com sinal δ positivo".to_string(),
            }
        ),
        18 => format!(
            "ventosa [disco com cruz]: prende este resíduo ao entulho ou à rocha em que toca (arrasto +{:.0}), {}, {}",
            v("forca") * g,
            match v("larga") {
                c if c < 0.0 => "agarra sempre",
                c if c < 2.5 => "larga com sinal γ positivo",
                _ => "larga com sinal δ positivo",
            },
            match v("emite") {
                c if c < 0.0 => "não emite sinal",
                c if c < 2.5 => "emite γ enquanto agarra",
                _ => "emite δ enquanto agarra",
            }
        ),
        7 => "armazenamento [disco com anéis]: DESATIVADO (já não se forma; a capacidade de energia vem do volume dos aminoácidos do corpo)".to_string(),
        8 => format!("{} · sente {}", sensor("sensor de comida DIRECIONAL (esquerda − direita)", "2 antenas verdes"), alvo()),
        9 => format!("{} · sente {}", sensor("sensor físico DIRECIONAL (esquerda − direita)", "2 antenas amarelas"), alvo_fisico()),
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
        14 => format!(
            "quimiossíntese [disco amarelo-enxofre]: consome o redutor das fumarolas; {:.0}% para reativar gastos, {:.0}% para energia, eficiência ×{:.2}",
            v("reciclar") * 100.0,
            (1.0 - v("reciclar")) * 100.0,
            v("eficiencia") * g
        ),
        13 => format!(
            "bias [ponto {}]: emite sempre {:+.2} em {}",
            if v("canal") < 0.5 { "laranja" } else { "verde" },
            v("valor") * g,
            fmt_canal(v("canal"))
        ),
        _ => format!(
            "bias de idade [meio disco]: emite {:+.2} em {} ao nascer; cai para metade a cada {:.0} passos de vida",
            v("valor") * g,
            fmt_canal(v("canal")),
            v("meia_vida")
        ),
    }
}

/// Índice de intensidade por omissão (ganho 1).
pub const GAIN_DEFAULT: u8 = 32;

/// Ganho de um índice de intensidade (0..63).
/// Relé: nomes dos 4 canais e forças escolhidas pelo 3.º codão.
pub const RELAY_CHANNELS: [&str; 4] = ["α", "β", "γ", "δ"];
pub const RELAY_MAG: [f32; 4] = [0.5, 1.0, 2.0, 4.0];

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
/// (tipo + 1) | (parâmetro << 5) | (intensidade << 8): tipo em 5 bits (até
/// 31 tipos), variante em 3.
pub fn organ_code(r: &Residue) -> u16 {
    r.organ.map_or(0, |(t, p, g)| (t as u16 + 1) | ((p as u16) << 5) | ((g as u16) << 8))
}

/// Tradução com órgãos (espelho exato do shader): a partir do primeiro AUG
/// (`require_start`) ou da base 0, até ao primeiro stop ou 64 resíduos.
/// `code` = `table::code_to_gpu` (promotor·20 + modificador).
pub fn translate_organs(genome: &[u8], require_start: bool, code: &[u32]) -> Vec<Residue> {
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
        // Promotor + modificador com entrada na tabela: órgão.
        if i + 6 <= genome.len() {
            let m = codon(genome[i + 3], genome[i + 4], genome[i + 5]);
            let c = if m == STOP { 0 } else { code[aa as usize * 20 + m as usize] };
            if c != 0 {
                // Segundo modificador (intensidade), se existir e não for stop.
                let (gain, used) =
                    if i + 9 <= genome.len() && codon(genome[i + 6], genome[i + 7], genome[i + 8]) != STOP {
                        (genome[i + 6] * 16 + genome[i + 7] * 4 + genome[i + 8], 9)
                    } else {
                        (GAIN_DEFAULT, 6)
                    };
                body.push(Residue { aa, organ: Some(((c & 0x1F) as u8 - 1, (c >> 5) as u8, gain)) });
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
        "CHEMO",
        "PROOFREAD",
        "DORMANCY",
        "AGE_BIAS",
        "HOLDFAST",
    ];
    for (i, name) in names.iter().enumerate() {
        s += &format!("const ORGAN_{name}: u32 = {i}u;\n");
    }
    s += &format!("const ORGAN_TYPES: u32 = {ORGAN_TYPES}u;\nconst ORGAN_VARIANTS: u32 = {VARIANTS}u;\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::amino::AA_LETTERS;
    use crate::life::table::{code_to_gpu, embedded_code};

    fn bases(s: &str) -> Vec<u8> {
        s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
    }

    fn aa_index(l: char) -> u8 {
        AA_LETTERS.iter().position(|&x| x == l).unwrap() as u8
    }

    fn code() -> Vec<u32> {
        code_to_gpu(&embedded_code())
    }

    /// Bases de um codão do aminoácido `l` (o primeiro na ordem A U G C).
    fn codon_of(l: char) -> String {
        let a = aa_index(l);
        let i = (0..64u8).find(|&i| codon(i / 16, (i / 4) % 4, i % 4) == a).unwrap();
        [i / 16, (i / 4) % 4, i % 4].iter().map(|&b| "AUGC".as_bytes()[b as usize] as char).collect()
    }

    /// Um par (promotor, modificador) da tabela que dá o órgão `t`, variante `v`.
    fn pair_for(t: Organ, v: u32) -> (char, char) {
        let c = embedded_code();
        for (p, row) in &c {
            for (m, e) in row {
                if *e == [t as u32, v] {
                    return (p.chars().next().unwrap(), m.chars().next().unwrap());
                }
            }
        }
        panic!("a tabela não tem {t:?} variante {v}");
    }

    #[test]
    fn promoter_plus_modifier_makes_an_organ() {
        // AUG (M) | promotor + modificador (boca, variante 0) + UUU (intensidade) | GGU (G) | UAA
        let (p, m) = pair_for(Organ::Mouth, 0);
        let g = bases(&format!("AUG{}{}UUUGGUUAA", codon_of(p), codon_of(m)));
        let body = translate_organs(&g, true, &code());
        assert_eq!(body.len(), 3);
        assert_eq!(body[0].organ, None);
        // UUU: 1*16 + 1*4 + 1 = 21 -> intensidade 21.
        assert_eq!(body[1].organ, Some((Organ::Mouth as u8, 0, 21)));
        assert_eq!(body[2].organ, None);
        // Sem segundo modificador (stop a seguir): intensidade por omissão, 6 bases.
        let b2 = translate_organs(&bases(&format!("AUG{}{}UAA", codon_of(p), codon_of(m))), true, &code());
        assert_eq!(b2[1].organ, Some((Organ::Mouth as u8, 0, GAIN_DEFAULT)));
        assert!((organ_gain(GAIN_DEFAULT) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn synonymous_codons_give_the_same_organ() {
        // GCA e GCG são ambos alanina: o mesmo órgão (ou nenhum, nos dois).
        let a = translate_organs(&bases("AUGUGUGCAUAA"), true, &code());
        let b = translate_organs(&bases("AUGUGUGCGUAA"), true, &code());
        assert_eq!(a[1].organ, b[1].organ);
    }

    #[test]
    fn anchor_and_bias_from_the_table() {
        for t in [Organ::Anchor, Organ::Bias] {
            let (p, m) = pair_for(t, 0);
            let b = translate_organs(&bases(&format!("AUG{}{}UAA", codon_of(p), codon_of(m))), true, &code());
            assert_eq!(b[1].organ, Some((t as u8, 0, GAIN_DEFAULT)));
        }
    }

    #[test]
    fn unmapped_pair_is_a_plain_residue() {
        // G (GGU) não é promotor: GGU GCA = dois resíduos normais.
        let body = translate_organs(&bases("AUGGGUGCAUAA"), true, &code());
        assert_eq!(body.iter().filter(|r| r.organ.is_some()).count(), 0);
        assert_eq!(body.len(), 3);
    }

    #[test]
    fn promoter_before_stop_is_a_plain_residue() {
        // AUG | UGU (C, promotor) | UAA (stop): sem órgão; o stop termina a cadeia.
        let body = translate_organs(&bases("AUGUGUUAA"), true, &code());
        assert_eq!(body, vec![Residue { aa: aa_index('M'), organ: None }, Residue { aa: aa_index('C'), organ: None }]);
    }
}
