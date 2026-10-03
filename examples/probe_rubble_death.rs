//! Morrem mais no entulho? Carrega uma cena (por omissão o autosave),
//! classifica os agentes pelo sítio onde está o centro (água / entulho, e
//! pobre / rico em ativados) e segue-os STEPS passos: quantos sobrevivem, a
//! energia no início e no fim. BIOTURB=0 desliga a bioturbação (para ver se
//! é o custo de escavar); MOTION muda o custo de movimento.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    if let Some(v) = envf("MOTION") {
        w.params.motion_cost = v;
    }
    if let Some(v) = envf("BIOTURB") {
        w.params.bioturbation = v;
    }
    let g = cfg.grid_size as usize;
    let wpc = cfg.world_units_per_cell as f32;
    let cells = w.read_cells_blocking(&gpu);
    let gamma = w.read_gamma_blocking(&gpu);
    let agents = w.read_agents_blocking(&gpu);
    // Grupo: 0 água pobre, 1 água rica, 2 entulho pobre, 3 entulho rico.
    let names = ["água, pobre", "água, rica (>=3 ativ.)", "entulho, pobre", "entulho, rico (>=3 ativ.)"];
    let mut group: HashMap<u32, (usize, f32)> = HashMap::new();
    for a in agents.iter().filter(|a| a.alive != 0) {
        let (cx, cy) = (((a.pos_x / wpc) as usize).min(g - 1), ((a.pos_y / wpc) as usize).min(g - 1));
        // Ativados num 3×3 à volta (média por célula).
        let mut act = 0u32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (x, y) = ((cx as i32 + dx).clamp(0, g as i32 - 1) as usize, (cy as i32 + dy).clamp(0, g as i32 - 1) as usize);
                let i = y * g + x;
                act += (0..4).map(|c| cells[i * 4 + c] & 0xFFFF).sum::<u32>();
            }
        }
        let rich = act as f32 / 9.0 >= 3.0;
        let rubble = gamma[cy * g + cx] > 0;
        group.insert(a.id, (rubble as usize * 2 + rich as usize, a.energy));
    }
    println!("bioturbação {}, custo por grão {}", w.params.bioturbation, w.params.bioturbation_cost);
    let steps = envf("STEPS").unwrap_or(1000.0) as u32;
    let mut done = 0;
    // Último estado visto de cada agente: (energia, idade, no entulho?,
    // complementos capturados / genoma).
    let mut last: HashMap<u32, (f32, u32, bool, f32)> = HashMap::new();
    // Mortes: [água, entulho] × [fome (<1), pouca (1–5), com energia (>5)].
    let mut deaths = [[0u32; 3]; 2];
    let mut ages = [Vec::new(), Vec::new()];
    let mut pairing = [Vec::new(), Vec::new()];
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        let gm = w.read_gamma_blocking(&gpu);
        let now: HashMap<u32, (f32, u32, bool, f32)> = w
            .read_agents_blocking(&gpu)
            .iter()
            .filter(|a| a.alive != 0)
            .map(|a| {
                let (cx, cy) = (((a.pos_x / wpc) as usize).min(g - 1), ((a.pos_y / wpc) as usize).min(g - 1));
                (a.id, (a.energy, a.age, gm[cy * g + cx] > 0, a.pair_count as f32 / a.gene_len.max(1) as f32))
            })
            .collect();
        for (id, (e, age, rub, pc)) in &last {
            if !now.contains_key(id) {
                let r = *rub as usize;
                deaths[r][if *e < 1.0 { 0 } else if *e < 5.0 { 1 } else { 2 }] += 1;
                ages[r].push(*age as f32);
                pairing[r].push(*pc);
            }
        }
        last = now;
    }
    for (r, name) in ["água", "entulho"].iter().enumerate() {
        let n = ages[r].len().max(1) as f32;
        println!(
            "mortes no(a) {name}: {} (última energia vista <1: {}, 1–5: {}, >5: {}); idade média {:.0}, genoma copiado {:.0}%",
            ages[r].len(),
            deaths[r][0],
            deaths[r][1],
            deaths[r][2],
            ages[r].iter().sum::<f32>() / n,
            100.0 * pairing[r].iter().sum::<f32>() / n
        );
    }
    let after: HashMap<u32, f32> = w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).map(|a| (a.id, a.energy)).collect();
    println!("{steps} passos depois:");
    for (gi, name) in names.iter().enumerate() {
        let ids: Vec<(&u32, &(usize, f32))> = group.iter().filter(|(_, (gg, _))| *gg == gi).collect();
        let n = ids.len();
        let alive: Vec<f32> = ids.iter().filter_map(|(id, _)| after.get(id).copied()).collect();
        let e0 = ids.iter().map(|(_, (_, e))| *e).sum::<f32>() / n.max(1) as f32;
        let e1 = alive.iter().sum::<f32>() / alive.len().max(1) as f32;
        println!(
            "  {name:26} {n:5} agentes: sobrevivem {:5.1}%  energia {e0:5.1} -> {e1:5.1} (dos vivos)",
            100.0 * alive.len() as f32 / n.max(1) as f32
        );
    }
}
