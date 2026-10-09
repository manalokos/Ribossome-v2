//! LINHAGENS AO LONGO DO TEMPO: lê o bloco "linhagens" de uma cena e mostra,
//! por censo, quantos ramos havia, quanto pesava o maior e a diversidade
//! efetiva (1 / Σp², o número de espécies "a sério").
//! SCENE (por omissão o autosave), ROWS (linhas da tabela, 30).
use ribossome::lineage::Lineages;
use ribossome::world::Scene;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let rows: usize = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let l = Lineages::from_bytes(scene.extra_block("linhagens").expect("a cena não tem linhagens")).unwrap();
    println!("{} censos de {} em {} epochs ({} a {}), {} ramos no total", l.censuses, l.every, l.every, l.first_epoch, l.last_epoch, l.branches.len());
    // epoch -> contagens dos ramos presentes nesse censo.
    let mut by_epoch: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut born: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    for b in &l.branches {
        for &(e, n) in &b.counts {
            by_epoch.entry(e).or_default().push(n);
        }
        let e = born.entry(b.born).or_default();
        if b.parent.is_some() { e.0 += 1 } else { e.1 += 1 }
    }
    let all: Vec<(&u32, &Vec<u32>)> = by_epoch.iter().collect();
    let step = (all.len() / rows).max(1);
    println!("{:>10} {:>6} {:>8} {:>8} {:>9} {:>9}", "epoch", "ramos", "maior %", "efetiva", "novos", "raízes");
    for chunk in all.chunks(step) {
        let n = chunk.len() as f32;
        let (mut ramos, mut top, mut eff, mut novos, mut raizes) = (0.0, 0.0, 0.0, 0u32, 0u32);
        for (e, c) in chunk {
            let total: f32 = c.iter().sum::<u32>() as f32;
            ramos += c.len() as f32;
            top += *c.iter().max().unwrap() as f32 / total * 100.0;
            eff += 1.0 / c.iter().map(|&x| (x as f32 / total).powi(2)).sum::<f32>();
            if let Some(b) = born.get(e) {
                novos += b.0;
                raizes += b.1;
            }
        }
        println!("{:>10} {:>6.1} {:>8.1} {:>8.2} {:>9} {:>9}", chunk[0].0, ramos / n, top / n, eff / n, novos, raizes);
    }
    let alive: Vec<_> = l.branches.iter().filter(|b| l.alive(b)).collect();
    println!("vivos agora: {} ramos; idades (epochs): {:?}", alive.len(), alive.iter().map(|b| b.last - b.born).collect::<Vec<_>>());
}
