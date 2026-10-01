//! Teste de conservação sem UI: mundo pequeno, N passos, a matéria total
//! tem de ser constante AO QUANTUM, por canal e por estado.
//! Precisa de uma GPU (corre no adaptador por omissão, sem janela).
//!
//! A contagem de referência é feita no CPU a partir da grelha inteira lida
//! da GPU; o livro-razão da GPU (redução) é comparado com essa contagem.

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Ledger, MAX_STEPS_PER_FRAME, World};

const STEPS: u32 = 4096;

#[test]
fn matter_is_conserved_exactly() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 7);
    // Difusão forte para haver muitos saltos (e corridas entre threads).
    world.params.diffusion = 30.0;
    let seeded = world.seed_matter(&gpu, 7);
    assert!(seeded.total() > 0);

    let initial_cells = world.read_cells_blocking(&gpu);
    assert_eq!(Ledger::from_cells(&initial_cells), seeded, "a sementeira não chegou intacta à GPU");
    assert_eq!(world.ledger_blocking(&gpu), seeded, "livro-razão da GPU ≠ contagem do CPU no arranque");

    let mut done = 0;
    while done < STEPS {
        let n = MAX_STEPS_PER_FRAME.min(STEPS - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += n;
    }

    let cells = world.read_cells_blocking(&gpu);
    let after = Ledger::from_cells(&cells);
    let moved = cells.iter().zip(&initial_cells).filter(|(a, b)| a != b).count();
    eprintln!(
        "{STEPS} passos em {}²: total {} → {}, {} de {} células×canal mudaram",
        cfg.grid_size,
        seeded.total(),
        after.total(),
        moved,
        cells.len()
    );
    assert!(moved > cells.len() / 10, "quase nada se moveu: o teste não estaria a testar nada");
    assert_eq!(after, seeded, "matéria não conservada (por canal e estado)");
    assert_eq!(world.ledger_blocking(&gpu), after, "livro-razão da GPU ≠ contagem do CPU");
}
