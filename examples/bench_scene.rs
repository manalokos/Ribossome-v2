//! Quanto custa gravar e carregar uma cena no mundo completo com N agentes
//! (N, por omissão 100 000). Grava em saves/bench_scene.ribo e apaga-o.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{Scene, World};

fn main() {
    let n: u32 = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    world.seed_matter(&gpu, 1);
    world.params.death_probability = 0.0;
    world.params.pairing_rate = 0.0;
    let mut rng = ribossome::life::SplitMix(5);
    let s = cfg.sim_size();
    let mut left = n;
    while left > 0 {
        let k = left.min(30_000);
        let reqs: Vec<SpawnRequest> = ribossome::life::seed_requests(k, [12, 120], true, s, &mut rng);
        world.request_seeds(&reqs);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 4);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        left -= k;
    }
    let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
    let path = std::path::PathBuf::from("saves/bench_scene.ribo");
    let t = std::time::Instant::now();
    let job = world.save_scene(&gpu, path.clone(), serde_json::json!({}), false);
    let read_gpu = t.elapsed().as_secs_f32();
    let msg = job.join().unwrap().unwrap();
    println!("{alive} agentes: leitura da GPU {read_gpu:.2} s (a app para isto), total {:.2} s", t.elapsed().as_secs_f32());
    println!("  {msg}");
    let t = std::time::Instant::now();
    let scene = Scene::read(&path).unwrap();
    let t_read = t.elapsed().as_secs_f32();
    let mut w2 = World::new(&gpu, cfg, 2);
    let t = std::time::Instant::now();
    w2.load_scene(&gpu, &scene).unwrap();
    println!("carregar: ler e descomprimir {t_read:.2} s, enviar para a GPU {:.2} s", t.elapsed().as_secs_f32());
    println!("agentes depois de carregar: {}", w2.life_counters_blocking(&gpu).alive(cfg.max_agents));
    std::fs::remove_file(&path).ok();
}
