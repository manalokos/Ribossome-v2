//! Aminoácidos: SÓ dados medidos, numa única tabela (fonte de verdade para
//! o CPU e para os shaders, que recebem o WGSL gerado aqui).
//!
//! Fontes:
//! - massa do resíduo (Da), volume da cadeia lateral (Å³) e Chou-Fasman
//!   (Pa, Pb, Pt): os valores já usados no v3 (docs/handoff/MAPA_DE_PORTAGEM.md);
//! - hidrofobicidade: Kyte & Doolittle 1982 (J Mol Biol 157:105);
//! - pKa das cadeias laterais: valores de referência de Lehninger
//!   (Asp 3,65, Glu 4,25, His 6,00, Cys 8,18, Tyr 10,07, Lys 10,53, Arg 12,48);
//!   a carga a pH 7 é calculada (Henderson-Hasselbalch), não escrita à mão;
//! - propensão catalítica: data/catalytic_propensity.csv
//!   (scripts/catalytic_propensity.py, M-CSA sobre Swiss-Prot 2026_03);
//! - energias de contacto: data/mj1996.csv (Miyazawa & Jernigan 1996);
//! - flexibilidade: data/flexibility_vihinen1994.csv (Vihinen et al. 1994)
//!   (scripts/aaindex_extract.py, a partir da AAindex);
//! - sensibilidades aos sinais internos α e β (rad por unidade de sinal):
//!   NÃO são química medida. São os valores de desenho afinados no v3
//!   (shaders/shared.wgsl, commit 7199600, jan. 2026), antes de os sinais
//!   terem sido desligados lá. Todas as juntas respondem aos dois sinais.

/// Ordem alfabética do código de uma letra (como no v3).
pub const AA_LETTERS: [char; 20] =
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

#[derive(Clone, Copy, Debug)]
pub struct AminoAcid {
    pub letter: char,
    /// Massa do resíduo (Da).
    pub mass: f32,
    /// Volume da cadeia lateral (Å³).
    pub volume: f32,
    /// Chou-Fasman: hélice, folha, volta.
    pub p_helix: f32,
    pub p_sheet: f32,
    pub p_turn: f32,
    /// Hidrofobicidade de Kyte-Doolittle.
    pub hydropathy: f32,
    /// pKa da cadeia lateral ionizável (0 = não ionizável) e se é ácida.
    pub pka: f32,
    pub acidic: bool,
    /// Propensão catalítica medida (cadeia lateral).
    pub catalytic: f32,
}

#[allow(clippy::too_many_arguments)]
const fn aa(
    letter: char,
    mass: f32,
    volume: f32,
    cf: [f32; 3],
    hydropathy: f32,
    pka: f32,
    acidic: bool,
    catalytic: f32,
) -> AminoAcid {
    AminoAcid {
        letter,
        mass,
        volume,
        p_helix: cf[0],
        p_sheet: cf[1],
        p_turn: cf[2],
        hydropathy,
        pka,
        acidic,
        catalytic,
    }
}

pub const AMINO: [AminoAcid; 20] = [
    aa('A', 71.08, 88.6, [1.42, 0.83, 0.66], 1.8, 0.0, false, 0.020),
    aa('C', 103.14, 108.5, [0.70, 1.19, 1.19], 2.5, 8.18, true, 5.100),
    aa('D', 115.09, 111.1, [1.01, 0.54, 1.46], -3.5, 3.65, true, 3.201),
    aa('E', 129.12, 138.4, [1.51, 0.37, 0.74], -3.5, 4.25, true, 1.997),
    aa('F', 147.18, 189.9, [1.13, 1.38, 0.60], 2.8, 0.0, false, 0.416),
    aa('G', 57.05, 60.1, [0.57, 0.75, 1.56], -0.4, 0.0, false, 0.024),
    aa('H', 137.14, 153.2, [1.00, 0.87, 0.95], -3.2, 6.00, false, 8.259),
    aa('I', 113.16, 166.7, [1.08, 1.60, 0.47], 4.5, 0.0, false, 0.039),
    aa('K', 128.17, 168.6, [1.16, 0.74, 1.01], -3.9, 10.53, false, 1.457),
    aa('L', 113.16, 166.7, [1.21, 1.30, 0.59], 3.8, 0.0, false, 0.042),
    aa('M', 131.19, 162.9, [1.45, 1.05, 0.60], 1.9, 0.0, false, 0.228),
    aa('N', 114.10, 114.1, [0.67, 0.89, 1.56], -3.5, 0.0, false, 0.984),
    aa('P', 97.12, 112.7, [0.57, 0.55, 1.52], -1.6, 0.0, false, 0.036),
    aa('Q', 128.13, 143.8, [1.11, 1.10, 0.98], -3.5, 0.0, false, 0.468),
    aa('R', 156.19, 173.4, [0.98, 0.93, 0.95], -4.5, 12.48, false, 1.666),
    aa('S', 87.08, 89.0, [0.77, 0.75, 1.43], -0.8, 0.0, false, 0.822),
    aa('T', 101.10, 116.1, [0.83, 1.19, 0.96], -0.7, 0.0, false, 0.560),
    aa('V', 99.13, 140.0, [1.06, 1.70, 0.50], 4.2, 0.0, false, 0.031),
    aa('W', 186.21, 227.8, [1.08, 1.37, 0.96], -0.9, 0.0, false, 1.460),
    aa('Y', 163.18, 193.6, [0.69, 1.47, 1.14], -1.3, 10.07, true, 2.114),
];

impl AminoAcid {
    /// Carga média da cadeia lateral a um dado pH (Henderson-Hasselbalch).
    pub fn charge_at(&self, ph: f32) -> f32 {
        if self.pka == 0.0 {
            return 0.0;
        }
        if self.acidic { -1.0 / (1.0 + 10f32.powf(self.pka - ph)) } else { 1.0 / (1.0 + 10f32.powf(ph - self.pka)) }
    }
}

/// Bases do RNA: 0 = A, 1 = U, 2 = G, 3 = C (a mesma ordem dos canais da grelha).
pub const BASES: [char; 4] = ['A', 'U', 'G', 'C'];
/// Complemento de Watson-Crick: A<->U, G<->C.
pub const fn complement(b: u8) -> u8 {
    b ^ 1
}

pub const STOP: u8 = 255;

/// Código genético padrão. Índice = 16·b1 + 4·b2 + b3 na ordem A U G C;
/// valor = índice em `AMINO` ou STOP.
pub const fn codon(b1: u8, b2: u8, b3: u8) -> u8 {
    let l = match (b1, b2, b3) {
        // UUU UUC Phe; UUA UUG Leu
        (1, 1, 1) | (1, 1, 3) => 'F',
        (1, 1, 0) | (1, 1, 2) => 'L',
        // CU* Leu
        (3, 1, _) => 'L',
        // AUU AUC AUA Ile; AUG Met
        (0, 1, 2) => 'M',
        (0, 1, _) => 'I',
        // GU* Val
        (2, 1, _) => 'V',
        // UC* Ser; CC* Pro; AC* Thr; GC* Ala
        (1, 3, _) => 'S',
        (3, 3, _) => 'P',
        (0, 3, _) => 'T',
        (2, 3, _) => 'A',
        // UAU UAC Tyr; UAA UAG stop
        (1, 0, 1) | (1, 0, 3) => 'Y',
        (1, 0, _) => '*',
        // CAU CAC His; CAA CAG Gln
        (3, 0, 1) | (3, 0, 3) => 'H',
        (3, 0, _) => 'Q',
        // AAU AAC Asn; AAA AAG Lys
        (0, 0, 1) | (0, 0, 3) => 'N',
        (0, 0, _) => 'K',
        // GAU GAC Asp; GAA GAG Glu
        (2, 0, 1) | (2, 0, 3) => 'D',
        (2, 0, _) => 'E',
        // UGU UGC Cys; UGA stop; UGG Trp
        (1, 2, 1) | (1, 2, 3) => 'C',
        (1, 2, 0) => '*',
        (1, 2, _) => 'W',
        // CG* Arg
        (3, 2, _) => 'R',
        // AGU AGC Ser; AGA AGG Arg
        (0, 2, 1) | (0, 2, 3) => 'S',
        (0, 2, _) => 'R',
        // GG* Gly
        (2, 2, _) => 'G',
        _ => '*',
    };
    if l == '*' {
        return STOP;
    }
    let mut i = 0;
    while i < 20 {
        if AA_LETTERS[i] == l {
            return i as u8;
        }
        i += 1;
    }
    STOP
}

pub const MAX_BODY: usize = 64;

/// Traduz a partir da primeira base (como a GPU, por omissão) até ao
/// primeiro stop ou MAX_BODY resíduos.
pub fn translate(genome: &[u8]) -> Vec<u8> {
    translate_from(genome, false)
}

/// Traduz a partir do primeiro AUG (`require_start`) ou da primeira base.
pub fn translate_from(genome: &[u8], require_start: bool) -> Vec<u8> {
    let start = if require_start {
        let Some(s) = genome.windows(3).position(|w| w == [0, 1, 2]) else { return Vec::new() };
        s
    } else {
        0
    };
    let mut body = Vec::new();
    for c in genome[start..].as_chunks::<3>().0 {
        let a = codon(c[0], c[1], c[2]);
        if a == STOP || body.len() >= MAX_BODY {
            break;
        }
        body.push(a);
    }
    body
}

fn csv_rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines().filter(|l| !l.starts_with('#') && !l.starts_with("aa")).map(|l| l.split(',').collect())
}

/// Matriz de contactos de Miyazawa-Jernigan 1996 (e_ij em RT), na ordem de `AMINO`.
pub fn mj_matrix() -> [[f32; 20]; 20] {
    let mut m = [[0.0; 20]; 20];
    for (i, row) in csv_rows(include_str!("../../data/mj1996.csv")).enumerate() {
        assert_eq!(row[0].chars().next(), Some(AA_LETTERS[i]));
        for j in 0..20 {
            m[i][j] = row[j + 1].parse().expect("mj1996.csv");
        }
    }
    m
}

/// Flexibilidade normalizada (B-values) de Vihinen 1994, na ordem de `AMINO`.
pub fn flexibility() -> [f32; 20] {
    let mut f = [0.0; 20];
    for (i, row) in csv_rows(include_str!("../../data/flexibility_vihinen1994.csv")).enumerate() {
        assert_eq!(row[0].chars().next(), Some(AA_LETTERS[i]));
        f[i] = row[1].parse().expect("flexibility csv");
    }
    f
}

/// Sensibilidade de cada junta ao sinal α e ao sinal β (valores do v3, ordem de `AMINO`).
pub const SIGNAL_SENSITIVITY: [(f32, f32); 20] = [
    (-0.2, 0.2),      // A
    (0.0, 0.349066),  // C
    (-0.2, 0.3),      // D
    (0.1, 0.12),      // E
    (0.2, -0.33),     // F
    (0.7, 0.1),       // G
    (0.2, -0.61),     // H
    (-0.3, 0.69),     // I
    (0.6, -0.16),     // K
    (-0.3332, 0.1),   // L
    (0.14, -0.64),    // M
    (0.2, 0.3),       // N
    (0.5, -0.1),      // P
    (0.24, -0.4),     // Q
    (0.5, -0.15),     // R
    (-0.349066, 0.0), // S
    (0.1, -0.5),      // T
    (-0.3, 0.73),     // V
    (0.31, -0.1),     // W
    (-0.2, 0.52),     // Y
];

/// Tabelas WGSL geradas (código genético e propriedades por aminoácido).
pub fn wgsl() -> String {
    let mut s = String::from("// ---- aminoácidos (gerado de src/life/amino.rs) ----\n");
    s += "const AA_STOP: u32 = 255u;\n";
    s += "const CODON_TABLE = array<u32, 64>(";
    for i in 0..64u8 {
        let c = codon(i >> 4, (i >> 2) & 3, i & 3);
        s += &format!("{}u{}", c, if i < 63 { ", " } else { "" });
    }
    s += ");\n";
    let col = |name: &str, f: &dyn Fn(&AminoAcid) -> f32| {
        let vals: Vec<String> = AMINO.iter().map(|a| format!("{:.4}", f(a))).collect();
        format!("const {name} = array<f32, 20>({});\n", vals.join(", "))
    };
    s += &col("AA_MASS", &|a| a.mass);
    s += &col("AA_VOLUME", &|a| a.volume);
    s += &col("AA_P_HELIX", &|a| a.p_helix);
    s += &col("AA_P_SHEET", &|a| a.p_sheet);
    s += &col("AA_P_TURN", &|a| a.p_turn);
    s += &col("AA_HYDROPATHY", &|a| a.hydropathy);
    s += &col("AA_CHARGE_PH7", &|a| a.charge_at(7.0));
    s += &col("AA_CATALYTIC", &|a| a.catalytic);
    let sa: Vec<String> = SIGNAL_SENSITIVITY.iter().map(|v| format!("{:.4}", v.0)).collect();
    let sb: Vec<String> = SIGNAL_SENSITIVITY.iter().map(|v| format!("{:.4}", v.1)).collect();
    s += &format!("const AA_ALPHA_SENS = array<f32, 20>({});\n", sa.join(", "));
    s += &format!("const AA_BETA_SENS = array<f32, 20>({});\n", sb.join(", "));
    let flex = flexibility();
    let fv: Vec<String> = flex.iter().map(|v| format!("{v:.4}")).collect();
    s += &format!("const AA_FLEX = array<f32, 20>({});\n", fv.join(", "));
    let mj = mj_matrix();
    let mv: Vec<String> = mj.iter().flatten().map(|v| format!("{v:.2}")).collect();
    s += &format!("const AA_MJ = array<f32, 400>({});\n", mv.join(", "));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genetic_code_counts() {
        let mut stops = 0;
        let mut per = [0; 20];
        for i in 0..64u8 {
            let c = codon(i >> 4, (i >> 2) & 3, i & 3);
            if c == STOP {
                stops += 1;
            } else {
                per[c as usize] += 1;
            }
        }
        assert_eq!(stops, 3);
        // Degenerescência do código padrão: L, R, S têm 6 codões; M e W um.
        let n = |l: char| per[AA_LETTERS.iter().position(|&x| x == l).unwrap()];
        assert_eq!((n('L'), n('R'), n('S'), n('M'), n('W'), n('I')), (6, 6, 6, 1, 1, 3));
        assert_eq!(per.iter().sum::<i32>(), 61);
    }

    #[test]
    fn translation_starts_at_aug() {
        // ..G G AUG UUU UGG UAA.. -> M F W
        let g = [2, 2, 0, 1, 2, 1, 1, 1, 1, 2, 2, 1, 0, 0];
        let body: String = translate_from(&g, true).iter().map(|&i| AA_LETTERS[i as usize]).collect();
        assert_eq!(body, "MFW");
        // Sem AUG obrigatório: lê desde a base 0 (GGA UGU UUU GGU AA -> G C F G).
        let body0: String = translate(&g).iter().map(|&i| AA_LETTERS[i as usize]).collect();
        assert_eq!(body0, "GCFG");
    }

    #[test]
    fn charges_at_ph7() {
        let q = |l: char| AMINO[AA_LETTERS.iter().position(|&x| x == l).unwrap()].charge_at(7.0);
        assert!((q('D') + 1.0).abs() < 0.01 && (q('K') - 1.0).abs() < 0.01);
        assert!(q('H') > 0.05 && q('H') < 0.15, "His a pH 7 ~ +0,09");
        assert_eq!(q('A'), 0.0);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn mj_and_flexibility_tables() {
        let m = mj_matrix();
        let i = |l: char| AA_LETTERS.iter().position(|&x| x == l).unwrap();
        // Valores publicados (MJ 1996, Table 3): L-L, C-C, K-K.
        assert_eq!((m[i('L')][i('L')], m[i('C')][i('C')], m[i('K')][i('K')]), (-7.37, -5.44, -0.12));
        for a in 0..20 {
            for b in 0..20 {
                assert_eq!(m[a][b], m[b][a], "MJ tem de ser simétrica");
            }
        }
        let f = flexibility();
        assert_eq!((f[i('G')], f[i('W')]), (1.031, 0.904));
    }

    #[test]
    fn catalytic_table_matches_csv() {
        let csv = include_str!("../../data/catalytic_propensity.csv");
        for line in csv.lines().filter(|l| !l.starts_with('#') && !l.starts_with("aa")) {
            let f: Vec<&str> = line.split(',').collect();
            let l = f[0].chars().next().unwrap();
            let p: f32 = f[4].parse().unwrap();
            let a = AMINO[AA_LETTERS.iter().position(|&x| x == l).unwrap()];
            assert!((a.catalytic - p).abs() < 0.001, "{l}: {} != {p}", a.catalytic);
        }
    }
}
