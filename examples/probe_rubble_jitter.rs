//! Vibração no entulho: nadadores construídos postos DENTRO de entulho do
//! terreno do projeto; mede, passo a passo, |Δrot| e |Δpos| médios e a
//! fração de passos em que a rotação troca de sinal (instabilidade numérica
//! = sinal alternado passo a passo). Compara com os mesmos em água livre.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { max_agents: 4096, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    world.seed_matter(&gpu, 1);
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.sedimentation = 0.0;
    world.params.spawn_energy = 60.0;
    let g = world.read_gamma_blocking(&gpu);
    let n = cfg.grid_size as usize;
    let w = cfg.world_units_per_cell as f32;
    let genome = bases(&format!("AUG UGU UCU {} UAA", "GGU ".repeat(15)));
    let mut rng = ribossome::life::SplitMix(9);
    let (mut rub, mut wat) = (Vec::new(), Vec::new());
    while rub.len() < 150 || wat.len() < 150 {
        let x = (rng.f32() * n as f32) as usize % n;
        let y = (rng.f32() * n as f32) as usize % n;
        let gg = g[y * n + x];
        let pos = ((x as f32 + 0.5) * w, (y as f32 + 0.5) * w);
        if (gg == 1 || gg == 2) && rub.len() < 150 { rub.push(pos); }
        if gg == 0 && wat.len() < 150 && y > n / 3 { wat.push(pos); }
    }
    let reqs: Vec<SpawnRequest> = rub.iter().chain(wat.iter()).map(|&(x, y)| SpawnRequest::with_genome(x, y, &genome)).collect();
    world.request_seeds(&reqs);
    let run = |w: &mut World, k: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    run(&mut world, 200);
    let start = world.read_agents_blocking(&gpu);
    let mut prev = start.clone();
    let mut prev_drot = vec![0f32; prev.len()];
    // [entulho, água]: soma |Δrot|, soma |Δpos|, trocas de sinal, amostras
    let mut acc = [[0f64; 4]; 2];
    for _ in 0..200 {
        run(&mut world, 1);
        let now = world.read_agents_blocking(&gpu);
        for (i, (a, b)) in prev.iter().zip(&now).enumerate() {
            if a.alive == 0 || b.alive == 0 || a.id != b.id { continue; }
            let cx = ((b.pos_x / w) as usize).min(n - 1);
            let cy = ((b.pos_y / w) as usize).min(n - 1);
            let k = if g[cy * n + cx] > 0 { 0 } else { 1 };
            let drot = b.rot - a.rot;
            acc[k][0] += drot.abs() as f64;
            acc[k][1] += ((b.pos_x - a.pos_x) as f64).hypot((b.pos_y - a.pos_y) as f64);
            if drot * prev_drot[i] < 0.0 { acc[k][2] += 1.0; }
            acc[k][3] += 1.0;
            prev_drot[i] = drot;
        }
        prev = now;
    }
    // Avanço líquido em 200 passos, conforme onde o agente começou.
    let mut net = [[0f64; 2]; 2];
    for (a, b) in start.iter().zip(&prev) {
        if a.alive == 0 || b.alive == 0 || a.id != b.id { continue; }
        let cx = ((a.pos_x / w) as usize).min(n - 1);
        let cy = ((a.pos_y / w) as usize).min(n - 1);
        let k = if g[cy * n + cx] > 0 { 0 } else { 1 };
        net[k][0] += ((b.pos_x - a.pos_x) as f64).hypot((b.pos_y - a.pos_y) as f64);
        net[k][1] += 1.0;
    }
    for (k, name) in ["entulho", "água"].iter().enumerate() {
        println!("{name}: avanço líquido médio em 200 passos {:.0}", net[k][0] / net[k][1].max(1.0));
        let c = acc[k][3].max(1.0);
        println!("{name}: |Δrot| {:.4} rad/passo, |Δpos| {:.2}/passo, troca de sinal da rotação em {:.0}% dos passos ({} amostras)",
            acc[k][0] / c, acc[k][1] / c, acc[k][2] / c * 100.0, acc[k][3]);
    }
}
