//! Quantas espécies há numa cena? Agrupa os genomas dos agentes vivos por
//! semelhança (ver `ribossome::species`) para vários limiares e descreve as
//! espécies com mais agentes ao limiar THRESHOLD (por omissão 0,15).
//! SCENE (por omissão o autosave).
use ribossome::gpu::Gpu;
use ribossome::life::table::code_to_gpu;
use ribossome::params::WorldConfig;
use ribossome::species::{cluster, living_genomes, organ_symbols, reverse_complement};
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let thr: f32 = std::env::var("THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let genomes = living_genomes(&gpu, &w);
    let total = genomes.len().max(1) as f32;
    println!("epoch {}, {} agentes", w.params.epoch, genomes.len());
    for t in [0.05, 0.10, 0.15, 0.25, 0.40] {
        let s = cluster(&genomes, t);
        let big = s.iter().filter(|s| s.count as f32 >= 0.01 * total).count();
        let top: Vec<String> = s.iter().take(5).map(|s| format!("{:.0}%", 100.0 * s.count as f32 / total)).collect();
        println!("  limiar {:.0}%: {} grupos, {big} com >= 1% dos agentes; os maiores: {}", t * 100.0, s.len(), top.join(" "));
    }
    let code = code_to_gpu(&w.organ_code);
    let rs = w.params.require_start != 0;
    println!("espécies ao limiar {:.0}% (as com >= 1% dos agentes):", thr * 100.0);
    for (i, s) in cluster(&genomes, thr).iter().enumerate().filter(|(_, s)| s.count as f32 >= 0.01 * total) {
        let (n1, o1) = organ_symbols(&s.leader, rs, &code);
        let (n2, o2) = organ_symbols(&reverse_complement(&s.leader), rs, &code);
        // Sequência de aminoácidos das duas fitas (minúscula = resíduo com órgão).
        let seq = |g: &[u8]| -> String {
            ribossome::life::organs::translate_organs(g, rs, &code)
                .iter()
                .map(|r| {
                    let c = ribossome::life::amino::AMINO[r.aa as usize].letter;
                    if r.organ.is_some() { c.to_ascii_lowercase() } else { c }
                })
                .collect()
        };
        println!("      A: {}   B: {}", seq(&s.leader), seq(&reverse_complement(&s.leader)));
        println!(
            "  #{i}: {:5} agentes ({:4.1}%), {} genomas distintos, {} bases; fita A ({:.0}%): {n1} resíduos [{o1}] | fita B: {n2} resíduos [{o2}]",
            s.count,
            100.0 * s.count as f32 / total,
            s.distinct,
            s.leader.len(),
            100.0 * s.same_strand as f32 / s.count as f32
        );
    }
}
