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
//!   DIRECIONAIS leem só UM lado da cadeia (esquerdo se o índice de
//!   intensidade for par, direito se ímpar; o quiral troca-os);
//! - relógio: canal, período e modulação por α/β vêm das variantes (assets/orgaos.json);
//! - relé: bits 0–1 modo (α->β, β->α, inverte α, inverte β), bit 2 ganho ×2;
//! - boca: catálise ×(2 + p); músculo: resposta ×(2 + p/2);
//!   armazenamento: soma a capacidade da variante × intensidade à do corpo
//!   (que vem do volume dos aminoácidos, 0,5 por resíduo médio); é pesado;
//! - fotossistema: bit 0 = energia da luz (0) ou reativar gastos (1);
//! - protease: família (o que corta), força e canal que a ativa;
//! - bias de idade: sinal que decai com a idade.
//!
//! Esta é a única fonte de verdade: o shader recebe as constantes geradas.

use super::amino::{STOP, codon};

pub const ORGAN_TYPES: usize = 21;

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
    Chiral = 19,
    /// Não vem do código dos órgãos: é o tradutor que o põe entre dois genes.
    Linker = 20,
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
    "quiral (inverte o lado das dobras)",
    "fio (liga dois genes)",
];

/// Os mesmos nomes em inglês, SÓ para mostrar (interface, gráficos, editor,
/// relatório). `ORGAN_NAMES` continua a ser a chave guardada nas cenas e nas
/// estatísticas ("% com <nome>"): não trocar um pelo outro.
pub const ORGAN_NAMES_EN: [&str; ORGAN_TYPES] = [
    "mouth",
    "muscle (amplifier)",
    "food sensor",
    "light sensor",
    "energy sensor",
    "clock",
    "relay",
    "storage",
    "one-sided food sensor",
    "one-sided light sensor",
    "photosystem",
    "protease",
    "anchor",
    "bias",
    "chemosynthesis",
    "proofreading (fewer mutations)",
    "dormancy (slow metabolism)",
    "age bias",
    "holdfast (grips the terrain)",
    "chiral (flips the side of the bends)",
    "linker (joins two genes)",
];

/// Letras curtas para o inspetor.
pub const ORGAN_SYMBOLS: [char; ORGAN_TYPES] = ['B', 'μ', 'f', 'l', 'e', '◷', 'r', 's', 'ψ', 'Ψ', 'φ', 'ξ', '⚓', 'b', 'χ', 'π', 'z', 'j', 'v', 'q', '~'];

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
    pd("canal", "0 = emits on α, 1 = on β"),
    pd("ganho", "multiplies what it senses (negative inverts). The sensor already returns the receptor occupancy (0..1); the directional one reads only one side of the chain"),
    pd("modo", "0 = by the LEVEL, 1 = by the CHANGE (chemotaxis)"),
    pd("memoria", "0..1: fraction of the sensor's charge that remains on each step (discharge = 1 − this; ~1/(1 − this) steps). For the level it is the reading itself; for the change it is the slow reference"),
];

/// Sensores "de luz" (físicos): as mesmas propriedades e o que sentem.
const LIGHT_SENSOR_PROPS: &[PropDef] = &[
    pd("canal", "0 = emits on α, 1 = on β"),
    pd("ganho", "multiplies what it senses (negative inverts). The sensor already returns the receptor occupancy (0..1); the directional one reads only one side of the chain"),
    pd("modo", "0 = by the LEVEL, 1 = by the CHANGE (chemotaxis)"),
    pd("memoria", "0..1: fraction of the sensor's charge that remains on each step (discharge = 1 − this; ~1/(1 − this) steps). For the level it is the reading itself; for the change it is the slow reference"),
    pd("alvo", "0 = light, 1 = temperature, 2 = vent reductant, 3 = terrain (grains)"),
];

/// Sensores "de comida": as mesmas propriedades e o que sentem.
const FOOD_SENSOR_PROPS: &[PropDef] = &[
    pd("canal", "0 = emits on α, 1 = on β"),
    pd("ganho", "multiplies what it senses (negative inverts). The sensor already returns the receptor occupancy (0..1); the directional one reads only one side of the chain"),
    pd("modo", "0 = by the LEVEL, 1 = by the CHANGE (chemotaxis)"),
    pd("memoria", "0..1: fraction of the sensor's charge that remains on each step (discharge = 1 − this; ~1/(1 − this) steps). For the level it is the reading itself; for the change it is the slow reference"),
    pd("alvo", "0 = food (activated monomers; the channels weighted by the affinities of the NEXT amino acid in the chain, the antenna), 1 = spent monomers (no variants any more), 2 = bodies of other agents (the antenna is the NEXT amino acid: D or E sees only bodies rich in lysine and arginine, what the family 1 protease cuts; K or R sees aspartate and asparagine, family 2; F, L, W, Y, I or V sees the aromatics and leucine, family 3; P, proline, smells the OPEN PROTEASES of others; any other neighbor sees them all)"),
];

/// Propriedades de cada tipo de órgão, por ordem (a mesma na GPU).
pub const ORGAN_PROPS: [&[PropDef]; ORGAN_TYPES] = [
    &[
        pd("forca", "multiplies the promoter's catalysis"),
        pd("vies_AU", "-1..1: prefers A/U (+) or G/C (−)"),
        pd("fecha", "−1 = always open; 2 = closes with a positive γ signal; 3 = closes with a positive δ signal (closed, it neither eats nor lets energy leak)"),
    ],
    &[pd("amplificacao", "multiplies the bending of the joint by the signals"), pd("canal", "0 = α only, 1 = β only, 2 = both")],
    FOOD_SENSOR_PROPS,
    LIGHT_SENSOR_PROPS,
    SENSOR_PROPS,
    &[
        pd("canal", "0 = emits on α, 1 = on β"),
        pd("periodo", "steps per cycle"),
        pd("mod_alfa", "the clock speeds up (+) or slows down (−) with the level of α"),
        pd("mod_beta", "the same with β"),
    ],
    &[
        pd("funcao", "0 = SWITCH (passes the signal from the input channel to the output channel and blocks the input here), 1 = copy (emits on the output, the input carries on), 2 = inverter (emits the negative), 3 = closing GATE (input above the threshold: the output channel does not pass here), 4 = CAPACITOR (slowly accumulates the input; on reaching the threshold it fires and emits on the output while it empties), 5 = DIFFERENTIATOR (emits on the output the difference between the current input and its recent average, amplified)"),
        pd("ganho", "multiplies what it emits (× the strength from the 3rd codon); for the capacitor it is the pulse height"),
        pd("limiar", "gate threshold (magnitude of the input signal) or charge at which the capacitor fires"),
        pd("carga", "capacitor: how much it charges per step and per unit of signal (it loses one tenth of this per step); differentiator: how fast the average follows the input (0..1 per step)"),
        pd("descarga", "capacitor: how much it empties per step while firing (threshold ÷ discharge = pulse duration in steps)"),
    ],
    &[pd("capacidade", "energy this organ adds to the body's capacity (× the intensity, limited between ×0.5 and ×2); the body alone stores 0.5 per residue of average volume")],
    FOOD_SENSOR_PROPS,
    LIGHT_SENSOR_PROPS,
    &[pd("reciclar", "0..1: fraction of the light used to reactivate spent monomers (the rest gives energy)"), pd("eficiencia", "multiplies the yield")],
    &[
        pd("alcance", "how far BEYOND contact it reaches (world units): 0 = touching only; ~40 = medium; ~100 = long. More reach = more drag, more mass and more expense while it is on"),
        pd("forca", "multiplies the risk of lysis it causes (× intensity)"),
        pd("canal", "−1 = always active; 2 = only with a positive γ signal; 3 = only with a positive δ signal (proportional to the signal, up to 1)"),
    ],
    &[
        pd("polaridade", "no longer decides what it binds to (any anchor will do); it only picks the color of the ring: +1 red, −1 blue"),
        pd("quebra", "probability per step of letting go (0 = permanent)"),
    ],
    &[pd("canal", "0 = emits on α, 1 = on β"), pd("valor", "constant signal emitted (× intensity)")],
    &[
        pd("reciclar", "0..1: fraction of the reductant used to reactivate spent monomers (the rest gives energy)"),
        pd("eficiencia", "multiplies what it consumes"),
    ],
    &[pd("protecao", "divides the mutation rate of this agent's copies by 1 + the sum of the protections (× intensity)")],
    &[
        pd("fator", "the agent's metabolism (maintenance, eating, chemosynthesis and copying, all together) is multiplied by this when the organ is acting at full strength"),
        pd("canal", "−1 = always acts; 2 = only with a positive γ signal; 3 = only with a positive δ signal (proportional to the signal, up to 1)"),
    ],
    &[
        pd("canal", "0 = α, 1 = β"),
        pd("valor", "signal emitted at birth (× intensity)"),
        pd("meia_vida", "steps of life until the signal falls to half"),
    ],
    &[
        pd("forca", "extra drag of this residue when it touches rubble or rock (× intensity): holds it in place"),
        pd("larga", "−1 = always grips; 2 = lets go with a positive γ signal; 3 = lets go with a positive δ signal"),
        pd("emite", "channel of the signal it emits while gripping (2 = γ, 3 = δ; −1 = none)"),
    ],
    &[],
    &[],
];

fn fmt_canal(v: f32) -> &'static str {
    if v < 0.5 { "α" } else { "β" }
}

/// Descrição em linguagem corrente de um órgão (tipo, variante, índice de
/// intensidade) com os valores da tabela.
pub fn describe(t: u8, p: u8, gain_idx: u8, table: &[super::table::OrganRow]) -> String {
    let g = organ_gain(gain_idx);
    let t = t as usize;
    let Some(row) = table.get(t) else { return format!("organ {t}") };
    let v = |name: &str| row.variantes.get(p as usize).and_then(|m| m.get(name)).copied().unwrap_or(0.0);
    let alvo = || match v("alvo").round() as i32 {
        1 => "SPENT monomers (the trail of those who eat)",
        2 => "BODIES of other agents (all, or only those that one protease family cuts, depending on the next amino acid: D/E -> K,R; K/R -> D,N; F/L/W/Y/I/V -> aromatics)",
        _ => "food (activated monomers)",
    };
    let alvo_fisico = || match v("alvo").round() as i32 {
        1 => "TEMPERATURE",
        2 => "vent REDUCTANT",
        3 => "TERRAIN (grains: rubble and rock)",
        _ => "light",
    };
    let sensor = |o_que: &str, aspeto: &str| {
        format!(
            "{o_que} [{aspeto}]: emits on {}, gain ×{:.2}, {}",
            fmt_canal(v("canal")),
            v("ganho") * g,
            if v("modo") < 0.5 {
                "by the LEVEL".to_string()
            } else {
                format!("by the CHANGE (memory {:.2})", v("memoria"))
            }
        )
    };
    match t {
        0 => format!(
            "mouth [disc with a dark opening]: eats activated monomers, strength ×{:.2}, {}, {}",
            v("forca") * g,
            match v("vies_AU") {
                x if x > 0.05 => format!("prefers A/U ({x:+.2})"),
                x if x < -0.05 => format!("prefers G/C ({x:+.2})"),
                _ => "no extra preference".into(),
            },
            match v("fecha") {
                c if c < 0.0 => "always open",
                c if c < 2.5 => "closes with a positive γ signal",
                _ => "closes with a positive δ signal",
            }
        ),
        1 => format!(
            "muscle [striped ellipse]: bend ×{:.2} with {}",
            v("amplificacao") * g,
            ["α", "β", "α and β"][(v("canal").round().clamp(0.0, 2.0)) as usize]
        ),
        2 => format!("{} · senses {}", sensor("TOTAL food sensor", "crown of green antennae"), alvo()),
        3 => format!("{} · senses {}", sensor("TOTAL physical sensor", "crown of yellow antennae"), alvo_fisico()),
        4 => sensor("internal energy sensor (with proline next: PAIN, the energy that proteases take from it, emitted on γ/δ)", "disc with a golden ring"),
        7 => format!("storage [disc with rings]: adds {:.1} to the body's energy capacity (the intensity only counts between ×0.5 and ×2); several in a row along the chain merge into a larger store, +25% for each extra one", v("capacidade") * g.clamp(0.5, 2.0)),
        8 | 9 => {
            // Um lado só: par = esquerda, ímpar = direita (troca depois de um quiral).
            let lado = if gain_idx & 1 == 0 { "LEFT" } else { "RIGHT" };
            if t == 8 {
                format!("{} · senses {}", sensor(&format!("food sensor on the {lado} side"), "green antennae on one side"), alvo())
            } else {
                format!("{} · senses {}", sensor(&format!("physical sensor on the {lado} side"), "yellow antennae on one side"), alvo_fisico())
            }
        }
        5 => format!(
            "clock [dial]: on {}, period {:.0} steps, strength ×{g:.2}{}",
            fmt_canal(v("canal")),
            v("periodo"),
            if v("mod_alfa").abs() + v("mod_beta").abs() > 0.0 {
                format!("; speeds up with α ×{:+.2} and with β ×{:+.2}", v("mod_alfa"), v("mod_beta"))
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
                0 => format!("relay SWITCH [diamond]: passes {cin} to {cout} (×{mag:.2}) and blocks {cin} here"),
                1 => format!("relay copy [diamond]: reads {cin} and emits on {cout} (×{mag:.2})"),
                2 => format!("relay inverter [diamond]: reads {cin} and emits −{cin} on {cout} (×{mag:.2})"),
                3 => format!("relay GATE [diamond]: with |{cin}| > {:.2}, {cout} does not pass here", v("limiar")),
                4 => format!(
                    "CAPACITOR [diamond]: accumulates {cin} (full in ~{:.0} steps of signal 1); on reaching {:.2} it fires {mag:+.2} on {cout} for ~{:.0} steps",
                    v("limiar") / v("carga").max(1e-4),
                    v("limiar"),
                    v("limiar") / v("descarga").max(1e-4)
                ),
                _ => format!("DIFFERENTIATOR [diamond]: emits on {cout} the change of {cin} relative to the recent average (×{mag:.2}; the average follows at {:.0}% per step)", v("carga") * 100.0),
            }
        }
        15 => format!("proofreading [shield]: mutation rate of the copies ÷ (1 + {:.1})", v("protecao") * g),
        16 => format!(
            "dormancy [moon]: metabolism × {:.2} (eats, copies and spends more slowly), {}",
            v("fator").clamp(0.01, 1.0).powf(g),
            match v("canal") {
                c if c < 0.0 => "always".to_string(),
                c if c < 2.5 => "only with a positive γ signal".to_string(),
                _ => "only with a positive δ signal".to_string(),
            }
        ),
        11 => format!(
            "protease [toothed disc; spikes if it has reach]: takes energy from {}, strength ×{:.2}, {}. Cuts according to the next amino acid (D/E -> lysine and arginine; K/R -> aspartate and asparagine; F/L/W/Y/I/V -> aromatics and leucine; other -> all three at one third). It spends energy while it is on; whoever has a protease of one family resists that family; proline protects",
            match v("alcance") {
                r if r < 10.0 => "whoever touches it".to_string(),
                r => format!("whoever is closer than {r:.0} units"),
            },
            v("forca") * g,
            match v("canal") {
                c if c < 0.0 => "always on".to_string(),
                c if c < 2.5 => "only with a positive γ signal".to_string(),
                _ => "only with a positive δ signal".to_string(),
            }
        ),
        12 => format!(
            "anchor [{} ring]: grips another anchor, a holdfast or a relay (free ones) of another agent that touches it, or a child; {}",
            if v("polaridade") >= 0.0 { "teal" } else { "blue" },
            if v("quebra") <= 0.0 {
                "permanent (only lets go if stretched too far)".to_string()
            } else {
                format!("lets go after {:.0} steps on average", 1.0 / v("quebra"))
            }
        ),
        14 => format!(
            "chemosynthesis [sulfur-yellow disc]: consumes the vent reductant; {:.0}% to reactivate spent monomers, {:.0}% for energy, efficiency ×{:.2}",
            v("reciclar") * 100.0,
            (1.0 - v("reciclar")) * 100.0,
            v("eficiencia") * g
        ),
        13 => format!(
            "bias [{} dot]: always emits {:+.2} on {}",
            if v("canal") < 0.5 { "orange" } else { "green" },
            v("valor") * g,
            fmt_canal(v("canal"))
        ),
        17 => format!(
            "age bias [half disc]: emits {:+.2} on {} at birth; halves every {:.0} steps of life",
            v("valor") * g,
            fmt_canal(v("canal")),
            v("meia_vida")
        ),
        _ => ORGAN_NAMES_EN.get(t).map_or_else(|| row.nome.clone(), |n| n.to_string()),
    }
}

/// Aminoácido do resíduo de fio entre dois genes (glicina).
pub const LINKER_AA: u8 = 5;

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
            // SEGUNDO GENE: se houver outro AUG depois do stop, a leitura
            // recomeça aí e os dois corpos ficam ligados por UM resíduo de
            // fio (órgão LINKER: comprido conforme o intervalo, mole, sem
            // ângulo de repouso e sem conduzir sinal). Como no shader.
            let from = i + 3;
            let next = (from..genome.len().saturating_sub(2)).find(|&j| genome[j..j + 3] == [0, 1, 2]);
            match next {
                Some(j) if body.len() + 1 < super::amino::MAX_BODY => {
                    let variant = (((j - from) / 9) as u8).min(5);
                    body.push(Residue { aa: LINKER_AA, organ: Some((20, variant, GAIN_DEFAULT)) });
                    i = j;
                    continue;
                }
                _ => break,
            }
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
        "CHIRAL",
        "LINKER",
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
