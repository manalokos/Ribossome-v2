//! Compensa ter fotossistema? Carrega uma cena, divide os agentes por terem
//! ou não fotossistema e por estarem ou não na zona com luz (o oitavo de
//! cima do mundo) e segue-os STEPS passos: sobrevivência, filhos por agente,
//! energia média (em fração da capacidade não se sabe: mostra a energia) e
//! quanto subiram ou desceram.
use std::collections::{HashMap, HashSet};

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let size = cfg.sim_size();
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    struct S {
        group: usize,
        y0: f32,
        e0: f32,
    }
    let mut st: HashMap<u32, S> = HashMap::new();
    for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let photo = (0..a.body_len as usize).any(|r| {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            o != 0 && (o & 0x1F) - 1 == 10
        });
        let lit = a.pos_y / size > 0.875;
        st.insert(a.id, S { group: photo as usize | ((lit as usize) << 1), y0: a.pos_y / size, e0: a.energy });
    }
    let steps = envf("STEPS").unwrap_or(3000.0) as u32;
    println!("epoch {}, sol agora {:.2} (0 = noite), {steps} passos", w.params.epoch, w.params.daylight(w.params.epoch));
    let mut kids: HashMap<u32, u32> = HashMap::new();
    let mut seen: HashSet<u32> = st.keys().copied().collect();
    let mut now: HashMap<u32, (f32, f32)> = HashMap::new();
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        now.clear();
        for a in w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0) {
            now.insert(a.id, (a.pos_y / size, a.energy));
            if seen.insert(a.id) && st.contains_key(&a.parent) {
                *kids.entry(a.parent).or_default() += 1;
            }
        }
    }
    let names = ["sem fotossistema, no escuro", "COM fotossistema, no escuro", "sem fotossistema, na luz", "COM fotossistema, na luz"];
    for (gi, name) in names.iter().enumerate() {
        let ids: Vec<(&u32, &S)> = st.iter().filter(|(_, s)| s.group == gi).collect();
        let n = ids.len().max(1) as f32;
        let alive: Vec<&(&u32, &S)> = ids.iter().filter(|(id, _)| now.contains_key(id)).collect();
        let na = alive.len().max(1) as f32;
        println!(
            "  {name:28} {:5} agentes: sobrevivem {:5.1}%  filhos/agente {:.2}  energia {:5.1} -> {:5.1}  altura {:.2} -> {:.2}",
            ids.len(),
            100.0 * alive.len() as f32 / n,
            ids.iter().map(|(id, _)| *kids.get(id).unwrap_or(&0) as f32).sum::<f32>() / n,
            alive.iter().map(|(_, s)| s.e0).sum::<f32>() / na,
            alive.iter().map(|(id, _)| now[*id].1).sum::<f32>() / na,
            alive.iter().map(|(_, s)| s.y0).sum::<f32>() / na,
            alive.iter().map(|(id, _)| now[*id].0).sum::<f32>() / na,
        );
    }
}
