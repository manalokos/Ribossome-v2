//! Traduz um genoma (GENOME="AUG ...") nas duas cadeias: aminoácidos, órgãos
//! (tipo, variante, intensidade) e alvos de protease de cada corpo. Só CPU.
use ribossome::life::organs::{ORGAN_NAMES_EN, ORGAN_SYMBOLS, translate_organs};
use ribossome::life::table;

fn main() {
    let text = std::env::var("GENOME").expect("GENOME=...");
    let g: Vec<u8> = text.chars().filter_map(|c| "AUGC".find(c.to_ascii_uppercase()).map(|i| i as u8)).collect();
    let rc = ribossome::species::reverse_complement(&g);
    let code = table::code_to_gpu(&table::load_code().0);
    let (amino, _) = table::load();
    for (name, s) in [("cadeia A", &g), ("cadeia B (complemento reverso)", &rc)] {
        let body = translate_organs(s, true, &code);
        println!("{name}: {} bases, {} resíduos", s.len(), body.len());
        let mut targets = [0u32; 3];
        for (k, r) in body.iter().enumerate() {
            let a = &amino[r.aa as usize];
            let m = a.protease_alvo.max(0.0) as u32;
            for f in 0..3 {
                if m & (1 << f) != 0 {
                    targets[f] += 1;
                }
            }
            let org = r.organ.map(|(t, v, i)| format!("  <- {} {} variante {} intensidade {}", ORGAN_SYMBOLS[t as usize], ORGAN_NAMES_EN[t as usize], v, i)).unwrap_or_default();
            println!("  {k:2} {:4} alvo {m}{org}", a.letra);
        }
        println!("  alvos por família (1,2,3): {targets:?} de {}", body.len());
    }
}
