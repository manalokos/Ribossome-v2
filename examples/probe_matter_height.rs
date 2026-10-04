//! Porque sobe (ou não desce) a matéria? Carrega uma cena e segue a altura
//! média dos monómeros livres (0 = fundo, 1 = topo) e a fração no quarto de
//! cima, ao longo de STEPS passos. Variantes por variáveis de ambiente:
//!   FLUID=0     desliga o fluido;
//!   NOLIFE=1    os agentes deixam de comer e de copiar (morrem e largam a
//!               matéria onde estão; deixa de haver transporte por agentes);
//!   SETTLE=x    gravidade dos monómeros;
//!   AGG=x       agregação dos ativados.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    if let Some(f) = env("FLUID") {
        w.settings.fluid_enabled = f != 0.0;
    }
    if let Some(c) = env("MG") {
        w.settings.mg_cycles = c as u32;
    }
    if let Some(c) = env("SUBSTEP") {
        w.settings.fluid_substep = c as u32;
    }
    println!("multigrid {}, ciclos V {}, resolve de {} em {} passos", w.settings.multigrid, w.settings.mg_cycles, w.settings.fluid_substep, w.settings.fluid_substep);
    if env("NOLIFE").is_some() {
        w.params.uptake_rate = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.photo_yield = 0.0;
        w.params.chemo_yield = 0.0;
    }
    if let Some(s) = env("SETTLE") {
        w.params.settle = s;
    }
    if let Some(a) = env("AGG") {
        w.params.aggregation = a;
    }
    // UNIFORM=1: espalha a matéria por igual (2 gastos por canal em cada
    // célula de água, nada no entulho nem na rocha), para medir se o
    // transporte tem viés a partir de um estado sem gradientes.
    if env("UNIFORM").is_some() {
        let gamma: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.gamma_buf)).to_vec();
        let cells: Vec<u32> = gamma.iter().flat_map(|&g| [if g == 0 { 2u32 << 16 } else { 0 }; 4]).collect();
        gpu.queue.write_buffer(&w.chem_buf, 0, bytemuck::cast_slice(&cells));
    }
    let (steps, every) = (env("STEPS").unwrap_or(20000.0) as u32, env("EVERY").unwrap_or(5000.0) as u32);
    let n = cfg.grid_size as usize;
    let report = |w: &World, done: u32| {
        let cells = w.read_cells_blocking(&gpu);
        let (mut tot, mut hsum, mut top) = (0u64, 0f64, 0u64);
        for y in 0..n {
            let row: u64 = (0..n * 4).map(|i| { let v = cells[y * n * 4 + i]; ((v & 0xFFFF) + (v >> 16)) as u64 }).sum();
            tot += row;
            hsum += row as f64 * (y as f64 + 0.5) / n as f64;
            if y >= n * 3 / 4 {
                top += row;
            }
        }
        let alive = w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count();
        println!(
            "+{done:6}: altura média da matéria livre {:.3}; {:4.1}% no quarto de cima; {alive} agentes",
            hsum / tot.max(1) as f64,
            100.0 * top as f64 / tot.max(1) as f64
        );
    };
    println!("fluido {}, gravidade dos monómeros {}, agregação {}", w.settings.fluid_enabled, w.params.settle, w.params.aggregation);
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
