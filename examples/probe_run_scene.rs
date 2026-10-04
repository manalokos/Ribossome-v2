//! Corre uma cena sem janela e mostra a população ao longo do tempo: para
//! ver se uma mudança (tabelas, regras) deixa viver a população de uma cena
//! antes de a aplicar ao mundo a correr. SCENE (por omissão o autosave),
//! STEPS (por omissão 6000), EVERY (por omissão 1000).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let (steps, every) = (envf("STEPS", 6000.0) as u32, envf("EVERY", 1000.0) as u32);
    let report = |w: &World, done: u32| {
        let a = w.read_agents_blocking(&gpu);
        let alive: Vec<_> = a.iter().filter(|a| a.alive != 0).collect();
        let c = w.life_counters_blocking(&gpu);
        let e = alive.iter().map(|a| a.energy).sum::<f32>() / alive.len().max(1) as f32;
        println!("+{done:5}: vivos {:6}  energia média {e:6.1}  nascimentos {}  mortes {} (fome {})", alive.len(), c.births, c.deaths, c.starved);
    };
    report(&w, 0);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % every < k || done == steps {
            report(&w, done);
        }
    }
}
