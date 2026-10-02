//! Cenas gravadas: gravar, carregar num mundo novo e ter o mesmo estado
//! (matéria exata, os mesmos agentes vivos), que continua a conservar matéria.

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn run(gpu: &Gpu, world: &mut World, steps: u32) {
    let mut done = 0;
    while done < steps {
        let n = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += n;
    }
}

#[test]
fn scene_round_trip() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut a = World::new(&gpu, cfg, 3);
    let seeded = a.seed_matter(&gpu, 3);
    let mut rng = ribossome::life::SplitMix(9);
    let reqs = ribossome::life::seed_requests(300, [12, 120], true, cfg.sim_size(), &mut rng);
    a.request_seeds(&reqs);
    a.params.swim_gain = 1.7; // um parâmetro fora do valor por omissão
    a.params.death_probability = 0.0;
    a.params.maintenance_cost = 0.0;
    run(&gpu, &mut a, 300);

    let dir = std::env::temp_dir().join("ribossome_scene_test");
    let path = dir.join("cena.ribo");
    let msg = a.save_scene(&gpu, path.clone(), serde_json::json!({"x": 1}), Vec::new(), false).join().unwrap().unwrap();
    eprintln!("gravado {msg}");

    let mut b = World::new(&gpu, cfg, 99);
    let scene = Scene::read(&path).unwrap();
    let (extra, notes) = b.load_scene(&gpu, &scene).unwrap();
    assert_eq!(extra["x"], 1);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(b.params.epoch, a.params.epoch);
    assert_eq!(b.params.swim_gain, 1.7);

    assert_eq!(b.read_cells_blocking(&gpu), a.read_cells_blocking(&gpu), "grelha diferente");
    assert_eq!(b.read_gamma_blocking(&gpu), a.read_gamma_blocking(&gpu), "terreno diferente");
    let (aa, ab) = (a.read_agents_blocking(&gpu), b.read_agents_blocking(&gpu));
    let alive = aa.iter().filter(|x| x.alive != 0).count();
    assert!(alive > 50, "poucos agentes vivos ({alive}): o teste não testaria nada");
    // Os slots livres podem ter restos de agentes mortos: só contam os vivos.
    for (x, y) in aa.iter().zip(&ab).filter(|(x, y)| x.alive != 0 || y.alive != 0) {
        assert_eq!(bytemuck::bytes_of(x), bytemuck::bytes_of(y), "agente diferente");
    }
    // Por resíduo (posições, sinais) e por slot (genoma), nos slots vivos.
    for (buf_a, buf_b, per_res) in [
        (&a.body_pos_buf, &b.body_pos_buf, true),
        (&a.signals_buf, &b.signals_buf, true),
        (&a.genomes_buf, &b.genomes_buf, false),
    ] {
        let (va, vb) = (a.read_f32_blocking(&gpu, buf_a), b.read_f32_blocking(&gpu, buf_b));
        let per_slot = va.len() / cfg.max_agents as usize;
        for (slot, ag) in aa.iter().enumerate().filter(|(_, x)| x.alive != 0) {
            let n = if per_res { ag.body_len as usize * per_slot / 64 } else { per_slot };
            let r = slot * per_slot..slot * per_slot + n;
            assert_eq!(bytemuck::cast_slice::<f32, u32>(&va[r.clone()]), bytemuck::cast_slice::<f32, u32>(&vb[r]));
        }
    }
    let (ca, cb) = (a.life_counters_blocking(&gpu), b.life_counters_blocking(&gpu));
    assert_eq!((ca.free_top, ca.next_id, ca.births), (cb.free_top, cb.next_id, cb.births));
    let la = a.ledger_blocking(&gpu);
    assert_eq!(b.ledger_blocking(&gpu), la, "livro-razão diferente");
    assert_eq!(la.total(), seeded.total(), "a matéria não se conservou antes de gravar");

    // Continua a correr e a conservar matéria (com nascimentos nos slots livres).
    run(&gpu, &mut b, 500);
    let lb = b.ledger_blocking(&gpu);
    assert_eq!(lb.total(), seeded.total(), "a matéria não se conservou depois de carregar");
    let _ = std::fs::remove_dir_all(dir);
}
