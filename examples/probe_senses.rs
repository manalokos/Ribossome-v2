//! Compensa ter sensor de comida direcional? Carrega uma cena (por omissão o
//! autosave), divide os agentes por órgãos (sensor de comida direcional, e
//! se têm algo que mexa: músculo, relógio ou bias) e segue-os STEPS passos:
//! sobrevivência, filhos por agente (pelo id do pai), distância percorrida e
//! comida (ativados no 5×5) à volta no início e no fim.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

const MOUTH: u32 = 0;
const MUSCLE: u32 = 1;
const CLOCK: u32 = 5;
const FOOD_DIR: u32 = 8;
const BIAS: u32 = 13;

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let g = cfg.grid_size as usize;
    let wpc = cfg.world_units_per_cell as f32;
    let food_at = |cells: &[u32], x: f32, y: f32| -> f32 {
        let (cx, cy) = ((x / wpc) as i32, (y / wpc) as i32);
        let mut s = 0u32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (xx, yy) = ((cx + dx).clamp(0, g as i32 - 1) as usize, (cy + dy).clamp(0, g as i32 - 1) as usize);
                let i = yy * g + xx;
                s += (0..4).map(|c| cells[i * 4 + c] & 0xFFFF).sum::<u32>();
            }
        }
        s as f32 / 25.0
    };
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let cells0 = w.read_cells_blocking(&gpu);
    // Grupo: bit 0 = sensor de comida direcional, bit 1 = mexe-se.
    let names = ["sem sensor, sem motor", "COM sensor, sem motor", "sem sensor, com motor", "COM sensor, com motor"];
    struct S {
        group: usize,
        x: f32,
        y: f32,
        food0: f32,
        path: f32,
    }
    let mut st: HashMap<u32, S> = HashMap::new();
    for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let mut has = [false; 16];
        for r in 0..a.body_len as usize {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            if o != 0 {
                has[((o & 0xF) - 1) as usize] = true;
            }
        }
        if !has[MOUTH as usize] {
            continue;
        }
        let motor = has[MUSCLE as usize] || has[CLOCK as usize] || has[BIAS as usize];
        let group = has[FOOD_DIR as usize] as usize | ((motor as usize) << 1);
        st.insert(a.id, S { group, x: a.pos_x, y: a.pos_y, food0: food_at(&cells0, a.pos_x, a.pos_y), path: 0.0 });
    }
    let steps = envf("STEPS").unwrap_or(2000.0) as u32;
    let mut kids: HashMap<u32, u32> = HashMap::new();
    let mut seen: std::collections::HashSet<u32> = st.keys().copied().collect();
    let mut done = 0;
    let mut alive_now: HashMap<u32, (f32, f32)> = HashMap::new();
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        alive_now.clear();
        for a in w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0) {
            alive_now.insert(a.id, (a.pos_x, a.pos_y));
            if seen.insert(a.id) && st.contains_key(&a.parent) {
                *kids.entry(a.parent).or_default() += 1;
            }
            if let Some(s) = st.get_mut(&a.id) {
                s.path += ((a.pos_x - s.x).powi(2) + (a.pos_y - s.y).powi(2)).sqrt();
                s.x = a.pos_x;
                s.y = a.pos_y;
            }
        }
    }
    let cells1 = w.read_cells_blocking(&gpu);
    println!("{} agentes com boca, {steps} passos", st.len());
    for (gi, name) in names.iter().enumerate() {
        let ids: Vec<(&u32, &S)> = st.iter().filter(|(_, s)| s.group == gi).collect();
        let n = ids.len().max(1) as f32;
        let alive: Vec<&(&u32, &S)> = ids.iter().filter(|(id, _)| alive_now.contains_key(id)).collect();
        let na = alive.len().max(1) as f32;
        let kids_pc = ids.iter().map(|(id, _)| *kids.get(id).unwrap_or(&0) as f32).sum::<f32>() / n;
        let path = alive.iter().map(|(_, s)| s.path).sum::<f32>() / na / wpc;
        let f0 = alive.iter().map(|(_, s)| s.food0).sum::<f32>() / na;
        let f1 = alive.iter().map(|(id, _)| { let p = alive_now[*id]; food_at(&cells1, p.0, p.1) }).sum::<f32>() / na;
        println!(
            "  {name:24} {:5} agentes: sobrevivem {:5.1}%  filhos/agente {kids_pc:.2}  percurso {path:6.1} células  comida à volta {f0:.2} -> {f1:.2}",
            ids.len(),
            100.0 * alive.len() as f32 / n
        );
    }
}
