//! Onde vivem os agentes e porquê: carrega uma cena (por omissão o autosave),
//! divide o mundo em blocos de 64×64 células e compara, por bloco, comida
//! ativada, temperatura, luz, redutor, entulho e agentes (e quantos têm
//! fotossistema / boca). Mostra os blocos ricos e vazios e os pobres e cheios.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    // Experiências: mudar parâmetros e correr STEPS passos antes de medir.
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = envf("SETTLE") { w.params.settle = v; }
    if let Some(v) = envf("AGG") { w.params.aggregation = v; }
    if let Some(v) = envf("PRESS") { w.params.monomer_pressure = v; }
    if envf("SUNHEAT0").is_some_and(|v| v > 0.0) { w.params.uv_strength = 0.0; }
    let steps = envf("STEPS").unwrap_or(1.0) as u32;
    let mut done = 0;
    while done < steps.max(1) {
        let k = ribossome::world::MAX_STEPS_PER_FRAME.min(steps.max(1) - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let g = cfg.grid_size as usize;
    let f = cfg.fluid_size as usize;
    let ls = g / 2;
    let cells = w.read_cells_blocking(&gpu);
    let gamma = w.read_gamma_blocking(&gpu);
    let temp = w.read_f32_blocking(&gpu, &w.temp_buf);
    let redox = w.read_f32_blocking(&gpu, &w.redox_buf);
    let light = w.read_f32_blocking(&gpu, &w.light_buf);
    let agents = w.read_agents_blocking(&gpu);
    let vel = w.read_f32_blocking(&gpu, &w.velocity_buf);
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let b = 64usize;
    let nb = g / b;
    #[derive(Default, Clone)]
    struct Blk { vx: f64, vy: f64, act: f64, spent: f64, temp: f64, light: f64, redox: f64, rubble: f64, n: f64, agents: u32, photo: u32, mouth: u32, chemo: u32 }
    let mut blk = vec![Blk::default(); nb * nb];
    for y in 0..g {
        for x in 0..g {
            let i = y * g + x;
            let k = &mut blk[(y / b) * nb + x / b];
            for c in 0..4 {
                k.act += (cells[i * 4 + c] & 0xFFFF) as f64;
                k.spent += (cells[i * 4 + c] >> 16) as f64;
            }
            k.temp += temp[(y * f / g) * f + x * f / g] as f64;
            let fi = (y * f / g) * f + x * f / g;
            k.vx += vel[fi * 2] as f64;
            k.vy += vel[fi * 2 + 1] as f64;
            k.redox += redox[(y * f / g) * f + x * f / g] as f64;
            k.light += light[(y / 2) * ls + x / 2] as f64;
            k.rubble += (gamma[i] > 0) as u32 as f64;
            k.n += 1.0;
        }
    }
    let wpc = cfg.world_units_per_cell as f32;
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let (cx, cy) = (((a.pos_x / wpc) as usize).min(g - 1), ((a.pos_y / wpc) as usize).min(g - 1));
        let k = &mut blk[(cy / b) * nb + cx / b];
        k.agents += 1;
        let mut has = [false; 16];
        for r in 0..a.body_len as usize {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            if o != 0 {
                has[((o & 0xF) - 1) as usize] = true;
            }
        }
        k.photo += has[10] as u32;
        k.mouth += has[0] as u32;
        k.chemo += has[14] as u32;
    }
    let row = |k: &Blk, i: usize| {
        format!(
            "bloco ({:2},{:2}) ativados/célula {:5.2} gastos {:5.1} T {:4.1} luz {:4.2} redutor {:5.1} entulho {:3.0}%  agentes {:4} (foto {} boca {} quimio {})",
            i % nb, i / nb, k.act / k.n, k.spent / k.n, k.temp / k.n, k.light / k.n, k.redox / k.n, 100.0 * k.rubble / k.n, k.agents, k.photo, k.mouth, k.chemo
        )
    };
    let alive: u32 = blk.iter().map(|k| k.agents).sum();
    println!("{} agentes vivos, epoch {}", alive, w.params.epoch);
    let mut idx: Vec<usize> = (0..blk.len()).collect();
    idx.sort_by(|&a, &c| (blk[c].act / blk[c].n).partial_cmp(&(blk[a].act / blk[a].n)).unwrap());
    println!("\nOS 12 BLOCOS MAIS RICOS em ativados:");
    for &i in idx.iter().take(12) { println!("  {}", row(&blk[i], i)); }
    let mut byag = idx.clone();
    byag.sort_by_key(|&i| std::cmp::Reverse(blk[i].agents));
    println!("\nOS 12 BLOCOS COM MAIS AGENTES:");
    for &i in byag.iter().take(12) { println!("  {}", row(&blk[i], i)); }
    // Médias por coluna de blocos, só nos dois terços de cima.
    println!("
POR COLUNA (blocos y >= {}), da esquerda para a direita:", nb / 3);
    for bx in 0..nb {
        let ks: Vec<&Blk> = (nb / 3..nb).map(|by| &blk[by * nb + bx]).collect();
        let s = |f: &dyn Fn(&Blk) -> f64| ks.iter().map(|k| f(k)).sum::<f64>() / ks.len() as f64;
        println!(
            "  x={:2}: água ({:+6.2}, {:+6.2}) monómeros {:5.1} (ativ {:5.2}) luz {:4.2} entulho {:3.0}% agentes {:4}",
            bx, s(&|k| k.vx / k.n), s(&|k| k.vy / k.n), s(&|k| (k.act + k.spent) / k.n), s(&|k| k.act / k.n), s(&|k| k.light / k.n), s(&|k| 100.0 * k.rubble / k.n),
            ks.iter().map(|k| k.agents).sum::<u32>()
        );
    }
    // Médias por faixa de altura.
    println!("\nPOR ALTURA (linhas de blocos, de cima para baixo):");
    for by in (0..nb).rev() {
        let ks: Vec<&Blk> = (0..nb).map(|bx| &blk[by * nb + bx]).collect();
        let s = |f: &dyn Fn(&Blk) -> f64| ks.iter().map(|k| f(k)).sum::<f64>() / ks.len() as f64;
        println!(
            "  y={:2}: ativ {:5.2} T {:4.1} luz {:4.2} redutor {:5.1} agentes {:5} (foto {:4})",
            by, s(&|k| k.act / k.n), s(&|k| k.temp / k.n), s(&|k| k.light / k.n), s(&|k| k.redox / k.n),
            ks.iter().map(|k| k.agents).sum::<u32>(), ks.iter().map(|k| k.photo).sum::<u32>()
        );
    }
}
