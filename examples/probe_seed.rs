//! Mundo novo semeado (como o botão "semear"): SEEDS sementes de genoma
//! curto montadas da sopa, no mundo por omissão, e a população ao longo de
//! STEPS passos. PARAMS=nome=valor,... muda parâmetros. Serve para ver se
//! uma regra faz a população explodir logo à partida.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 7);
    w.seed_matter(&gpu, 7);
    w.settings.fluid_enabled = envf("FLUID", 0.0) != 0.0;
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let mut rng = ribossome::life::SplitMix(11);
    let s = cfg.sim_size();
    let n = envf("SEEDS", 3000.0) as usize;
    let (lo, hi) = (envf("LEN_MIN", 24.0) as u32, envf("LEN_MAX", 60.0) as u32);
    let reqs: Vec<SpawnRequest> =
        (0..n).map(|_| SpawnRequest::new(s * rng.f32(), s * rng.f32(), lo + (rng.f32() * (hi - lo + 1) as f32) as u32, 1)).collect();
    w.request_seeds(&reqs);
    let (steps, every) = (envf("STEPS", 12000.0) as u32, envf("EVERY", 2000.0) as u32);
    let organs_of = |w: &World| -> (usize, f32, f32) {
        let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
        let agents = w.read_agents_blocking(&gpu);
        let (mut alive, mut mouth, mut len) = (0usize, 0usize, 0f32);
        for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            alive += 1;
            len += a.body_len as f32;
            mouth += (0..a.body_len as usize).any(|r| {
                let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
                o != 0 && (o & 0x1F) - 1 == 0
            }) as usize;
        }
        (alive, 100.0 * mouth as f32 / alive.max(1) as f32, len / alive.max(1) as f32)
    };
    let mut done = 0;
    let mut peak = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % every < k || done == steps {
            let (alive, mouth, len) = organs_of(&w);
            peak = peak.max(alive);
            println!("+{done:6}: vivos {alive:7}  com boca {mouth:5.1}%  resíduos por corpo {len:5.1}");
        }
    }
    println!("pico {peak}");
}
