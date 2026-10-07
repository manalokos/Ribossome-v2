//! Relatório de uma cena gravada, sem a aplicação: SCENE (por omissão o
//! autosave) -> OUT (por omissão saves/relatorios/relatorio_<epoch>.html).
//! STEPS = passos a correr antes (por omissão 8: sem correr nenhum não há
//! ataques a decorrer para mostrar). Usa o registo de linhagens da cena;
//! CENSOS e CADA fazem um registo de experiência antes de gerar.
use ribossome::gpu::Gpu;
use ribossome::lineage::Lineages;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let mut lineages = scene.extra_block("linhagens").and_then(Lineages::from_bytes);
    w.load_scene(&gpu, &scene).unwrap();
    // CENSOS=n e CADA=k: corre n × k passos com um censo das linhagens em
    // cada k (para experimentar o registo numa cena que não o tem).
    let censos: u32 = std::env::var("CENSOS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let cada: u32 = std::env::var("CADA").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    for c in 0..censos {
        let l = lineages.get_or_insert_with(|| Lineages { every: cada, ..Default::default() });
        let genomes = ribossome::species::living_genomes(&gpu, &w);
        l.census(w.params.epoch, &ribossome::species::cluster(&genomes, 0.15), genomes.len());
        if c + 1 < censos {
            let mut done = 0;
            while done < cada {
                let k = ribossome::world::MAX_STEPS_PER_FRAME.min(cada - done);
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                w.encode_steps(&gpu.queue, &mut enc, k);
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle();
                done += k;
            }
        }
    }
    if steps > 0 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, steps);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    }
    let epoch = w.params.epoch;
    let html = ribossome::report::generate(&gpu, &w, lineages.as_ref(), &format!("Ribossome: {path}, epoch {epoch}"));
    let out = std::env::var("OUT").unwrap_or_else(|_| format!("saves/relatorios/relatorio_{epoch}.html"));
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&out, &html).unwrap();
    println!("{out}: {:.1} MB", html.len() as f64 / 1e6);
    // ARVORE=ficheiro: também a página só com a árvore das linhagens.
    if let (Ok(path), Some(l)) = (std::env::var("ARVORE"), lineages.as_ref()) {
        std::fs::write(&path, ribossome::tree_view::page(l, &w, &format!("Árvore das linhagens, epoch {epoch}"), Some(&ribossome::tree_view::portraits(&gpu, &w, l)))).unwrap();
        println!("{path}");
    }
}
