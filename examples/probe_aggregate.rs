//! Agregação dos ativados: mundo pequeno sem agentes, N passos com
//! agregação 0 e com AGG; mede o agrupamento (variância/média das contagens
//! de ativados por célula: 1 = ao acaso, > 1 = grumos) e a conservação.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Ledger, MAX_STEPS_PER_FRAME, World};

fn clumping(cells: &[u32]) -> f64 {
    let n = cells.len() / 4;
    let act: Vec<f64> = (0..n).map(|i| (0..4).map(|c| (cells[i * 4 + c] & 0xFFFF) as f64).sum()).collect();
    let mean = act.iter().sum::<f64>() / n as f64;
    let var = act.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64;
    var / mean.max(1e-9)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let agg: f32 = std::env::var("AGG").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    for a in [0.0, agg] {
        let mut w = World::new(&gpu, WorldConfig::TEST, 9);
        let seeded = w.seed_matter(&gpu, 9);
        w.params.aggregation = a;
        w.params.reactivation_rate = 0.001;
        let c0 = clumping(&w.read_cells_blocking(&gpu));
        let mut done = 0;
        while done < steps {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += MAX_STEPS_PER_FRAME;
        }
        let cells = w.read_cells_blocking(&gpu);
        let after = Ledger::from_cells(&cells);
        println!(
            "agregação {a}: agrupamento {c0:.2} -> {:.2}   matéria {} -> {}",
            clumping(&cells),
            seeded.total(),
            after.total()
        );
    }
}
