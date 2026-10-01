//! Testes do mundo sem UI. Precisam de uma GPU (adaptador por omissão, sem janela).
//!
//! Conservação: mundo pequeno, N passos com TODOS os sistemas do mundo
//! ligados (fluido, temperatura, luz, reações, transporte). A matéria tem de
//! ser constante AO QUANTUM, no total e por canal (as reações mudam o
//! estado ativado/gasto, mas nunca o tipo nem a quantidade). A contagem de
//! referência é feita no CPU a partir da grelha inteira lida da GPU; o
//! livro-razão da GPU é comparado com ela.

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Ledger, MAX_STEPS_PER_FRAME, World};

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

fn channels(l: &Ledger) -> [u64; 4] {
    [l.channel(0), l.channel(1), l.channel(2), l.channel(3)]
}

#[test]
fn matter_is_conserved_exactly() {
    const STEPS: u32 = 4096;
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 7);
    // Difusão forte para haver muitos saltos (e corridas entre threads).
    world.params.diffusion = 30.0;
    world.params.settle = 1.0;
    let seeded = world.seed_matter(&gpu, 7);
    assert!(seeded.total() > 0);

    let initial_cells = world.read_cells_blocking(&gpu);
    assert_eq!(Ledger::from_cells(&initial_cells), seeded, "a sementeira não chegou intacta à GPU");
    assert_eq!(world.ledger_blocking(&gpu), seeded, "livro-razão da GPU ≠ contagem do CPU no arranque");

    run(&gpu, &mut world, STEPS);

    let cells = world.read_cells_blocking(&gpu);
    let after = Ledger::from_cells(&cells);
    let moved = cells.iter().zip(&initial_cells).filter(|(a, b)| a != b).count();
    let act_before: u64 = seeded.act.iter().map(|&v| v as u64).sum();
    let act_after: u64 = after.act.iter().map(|&v| v as u64).sum();
    eprintln!(
        "{STEPS} passos em {}²: total {} → {}, ativados {act_before} → {act_after}, {moved} de {} células×canal mudaram",
        cfg.grid_size,
        seeded.total(),
        after.total(),
        cells.len()
    );
    assert!(moved > cells.len() / 10, "quase nada se moveu: o teste não estaria a testar nada");
    assert_ne!(act_before, act_after, "nenhuma reação mudou o estado: as reações não estão a correr");
    assert_eq!(after.total(), seeded.total(), "matéria total não conservada");
    assert_eq!(channels(&after), channels(&seeded), "matéria não conservada por canal");
    assert_eq!(world.ledger_blocking(&gpu), after, "livro-razão da GPU ≠ contagem do CPU");
}

/// O fluido arranca da fumarola, fica finito, e a luz UV cai com a profundidade.
#[test]
fn fluid_and_light_are_sane() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 3);
    world.seed_matter(&gpu, 3);
    run(&gpu, &mut world, 600);

    let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
    let temp = world.read_f32_blocking(&gpu, &world.temp_buf);
    assert!(vel.iter().chain(&temp).all(|v| v.is_finite()), "NaN/Inf no fluido");
    let max_speed = vel.chunks(2).map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).fold(0.0f32, f32::max);
    let max_t = temp.iter().cloned().fold(0.0f32, f32::max);
    // Velocidade média vertical na metade de cima da pluma (deve subir: +y).
    let n = cfg.fluid_size as usize;
    let mean_vy: f32 = vel.chunks(2).map(|v| v[1]).sum::<f32>() / (n * n) as f32;
    eprintln!("fluido: |v|max {max_speed:.3} células/s, T max {max_t:.2}, vy média {mean_vy:.4}");
    assert!(max_t > 2.0, "a fumarola não aqueceu a água");
    assert!(max_speed > 0.1, "a flutuação não pôs o fluido em movimento");

    // Incompressibilidade num aquário fechado: o fluxo líquido através de
    // qualquer linha horizontal tem de ser ~0 (com Jacobi 10 era ~50%).
    for y in [n / 4, n / 2, 3 * n / 4] {
        let net: f32 = (0..n).map(|x| vel[(y * n + x) * 2 + 1]).sum::<f32>();
        let gross: f32 = (0..n).map(|x| vel[(y * n + x) * 2 + 1].abs()).sum::<f32>();
        eprintln!("linha {y}: fluxo líquido {net:.2}, total {gross:.2}");
        assert!(net.abs() < 0.1 * gross.max(1.0), "o fluido cria/destrói água na linha {y}");
    }

    let light = world.read_f32_blocking(&gpu, &world.light_buf);
    let g = cfg.grid_size as usize;
    let top = light[(g - 1) * g + g / 2];
    let bottom = light[g / 2];
    eprintln!("luz: topo {top:.3}, fundo {bottom:.5}");
    assert!(top > 0.9 && bottom < top * 0.01, "a luz não cai com a profundidade como esperado");
}
