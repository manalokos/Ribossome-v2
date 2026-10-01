//! O resultado depende de como os passos são agrupados por frame?
//! Corre os mesmos N passos (a) em lotes de 64 e (b) um a um, e compara.
//! Também compara (a) com uma repetição de (a), para separar o efeito do
//! agrupamento do não-determinismo das corridas atómicas na GPU.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn run(gpu: &Gpu, batch: u32, steps: u32) -> Vec<u32> {
    let mut world = World::new(gpu, WorldConfig::TEST, 9);
    world.seed_matter(gpu, 9);
    let mut done = 0;
    while done < steps {
        let k = batch.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
    }
    gpu.wait_idle();
    world.read_cells_blocking(gpu)
}

fn diff(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let steps = 256;
    let a = run(&gpu, 64, steps);
    let a2 = run(&gpu, 64, steps);
    let b = run(&gpu, 1, steps);
    println!("{} células×canal no total", a.len());
    println!("lotes de 64 vs lotes de 64 (repetição): {} diferentes", diff(&a, &a2));
    println!("lotes de 64 vs um a um:                 {} diferentes", diff(&a, &b));
}
