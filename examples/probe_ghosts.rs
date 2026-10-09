//! RESTOS DE QUEM MORRE: carrega uma cena, aponta a câmara a um agente,
//! provoca mortes durante uns passos e grava imagens em vários momentos da
//! animação (as peças a separarem-se). SCENE, OUT (prefixo dos PNG), SIZE.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let out = std::env::var("OUT").unwrap_or_else(|_| "restos".into());
    let size: u32 = std::env::var("SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let agents = w.read_agents_blocking(&gpu);
    // O agente com mais vizinhos a menos de 400 unidades (entre uma amostra).
    let alive: Vec<_> = agents.iter().filter(|a| a.alive != 0 && a.body_len >= 8).collect();
    let centre = alive
        .iter()
        .step_by((alive.len() / 400).max(1))
        .max_by_key(|a| alive.iter().filter(|b| (a.pos_x - b.pos_x).abs() < 400.0 && (a.pos_y - b.pos_y).abs() < 400.0).count())
        .map(|a| [a.pos_x, a.pos_y])
        .unwrap();
    // FLOW=1: em vez disso, o agente (com corpo) onde a água corre mais.
    let centre = if std::env::var("FLOW").is_ok() {
        let vel: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.velocity_buf)).to_vec();
        let fs = w.cfg.fluid_size as usize;
        let cell = |p: f32| ((p / w.cfg.sim_size() * fs as f32) as usize).min(fs - 1);
        let speed = |a: &&&ribossome::params::Agent| {
            let v = vel[cell(a.pos_y) * fs + cell(a.pos_x)];
            ((v[0] * v[0] + v[1] * v[1]).sqrt() * 1000.0) as u32
        };
        alive.iter().max_by_key(speed).map(|a| [a.pos_x, a.pos_y]).unwrap()
    } else {
        centre
    };
    let cam = Camera { center: centre, zoom: size as f32 / 900.0 };
    let half = 450.0;
    let cap = Capture::new(&gpu, &w, size);
    let run = |w: &mut World, n: u32| {
        w.set_draw_rect(&gpu.queue, Some(([centre[0] - half, centre[1] - half], [centre[0] + half, centre[1] + half])));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    let shot = |w: &World, name: &str| {
        cap.view.epoch.set(w.params.epoch);
        cap.view.ghost_steps.set(600.0);
        let rgba = cap.render(&gpu, w, &cam, 0, 0.0);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_{name}.png"))).unwrap();
    };
    run(&mut w, 8);
    shot(&w, "0_antes");
    let normal = w.params.death_probability;
    w.params.death_probability = 2.0;
    run(&mut w, 8);
    w.params.death_probability = normal;
    for (name, steps) in [("1_inicio", 40u32), ("2_meio", 200), ("3_fim", 250)] {
        let mut left = steps;
        while left > 0 {
            let k = left.min(64);
            run(&mut w, k);
            left -= k;
        }
        shot(&w, name);
    }
    // A água ali: velocidade (células do fluido por segundo) na célula do centro
    // e o que isso dá em unidades do mundo ao fim da animação.
    let vel: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.velocity_buf)).to_vec();
    let fs = w.cfg.fluid_size as usize;
    let cell = |p: f32| ((p / w.cfg.sim_size() * fs as f32) as usize).min(fs - 1);
    let v = vel[cell(centre[1]) * fs + cell(centre[0])];
    let per_step = w.cfg.sim_size() / fs as f32 * w.params.dt;
    println!("água no centro: ({:.2}, {:.2}) células/s = {:.1} unidades do mundo em 600 passos", v[0], v[1], (v[0] * v[0] + v[1] * v[1]).sqrt() * per_step * 600.0);
    let c = w.life_counters_blocking(&gpu);
    println!("mortes {} nascimentos {}", c.deaths, c.births);
}
