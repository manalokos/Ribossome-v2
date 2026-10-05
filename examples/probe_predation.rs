//! Quanto rende a predação? Carrega uma cena, corre 200 passos e depois
//! tira SAMPLES amostras (uma a cada 10 passos) das mordidas do último
//! passo (contact_disp: .x = energia perdida, .z = energia ganha a morder).
//! Mostra, por faixa de altura: agentes, quantos têm protease, que fração
//! deles está a morder num dado passo, o ganho médio por predador e por
//! passo (comparado com a manutenção base do corpo, 0,002 por resíduo) e a
//! fração dos agentes que está a ser mordida. PARAMS=nome=valor,... muda
//! parâmetros (ex.: protease_power=10).
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
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let run = |w: &mut World, steps: u32| {
        let mut done = 0;
        while done < steps {
            let k = MAX_STEPS_PER_FRAME.min(steps - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
    };
    run(&mut w, 200);
    let samples = envf("SAMPLES").unwrap_or(20.0) as u32;
    const BANDS: usize = 4;
    #[derive(Default, Clone, Copy)]
    struct B {
        agents: f64,
        preds: f64,
        biting: f64,
        gain: f64,
        upkeep: f64,
        bitten: f64,
        lost: f64,
        pred_energy: f64,
        prey_energy: f64,
    }
    let mut b = [B::default(); BANDS];
    let size = cfg.sim_size();
    for _ in 0..samples {
        run(&mut w, 10);
        let agents = w.read_agents_blocking(&gpu);
        let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
        let bite: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.contact_disp_buf)).to_vec();
        for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            let band = ((a.pos_y / size * BANDS as f32) as usize).min(BANDS - 1);
            let pred = (0..a.body_len as usize).any(|r| {
                let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
                o != 0 && (o & 0x1F) - 1 == 11
            });
            let e = &mut b[band];
            e.agents += 1.0;
            if pred {
                e.preds += 1.0;
                e.biting += (bite[slot][2] > 0.0) as u32 as f64;
                e.gain += bite[slot][2] as f64;
                e.upkeep += 0.002 * a.body_len as f64;
                e.pred_energy += a.energy as f64;
            } else {
                e.prey_energy += a.energy as f64;
            }
            e.bitten += (bite[slot][0] > 0.0) as u32 as f64;
            e.lost += bite[slot][0] as f64;
        }
    }
    println!("epoch {}, força das proteases × {}; faixas de cima para baixo ({samples} amostras):", w.params.epoch, w.params.protease_power);
    for band in (0..BANDS).rev() {
        let e = b[band];
        let s = samples as f64;
        println!(
            "  {:6.0} agentes, {:5.0} com protease ({:4.1}%): {:4.1}% deles a atacar; risco causado {:.4}/passo por predador (manutenção base do corpo {:.4}); {:4.1}% dos agentes sob ataque, risco de lise {:.4}/passo cada; energia: predadores {:.1}, outros {:.1}",
            e.agents / s,
            e.preds / s,
            100.0 * e.preds / e.agents.max(1.0),
            100.0 * e.biting / e.preds.max(1.0),
            e.gain / e.preds.max(1.0),
            e.upkeep / e.preds.max(1.0),
            100.0 * e.bitten / e.agents.max(1.0),
            e.lost / e.bitten.max(1.0),
            e.pred_energy / e.preds.max(1.0),
            e.prey_energy / (e.agents - e.preds).max(1.0),
        );
    }
}
