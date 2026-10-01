//! Os organismos nadam? Desliga tudo o que os move por fora (fluido,
//! browniano, difusioforese, contacto) e mede o deslocamento só pela
//! mudança de forma (RFT), depois da dobragem.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn run(gpu: &Gpu, world: &mut World, steps: u32) {
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
}

fn case(gpu: &Gpu, name: &str, rft: bool, motor: f32, kt: f32, coupling: f32) {
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(gpu, cfg, 2);
    world.seed_matter(gpu, 2);
    world.settings.fluid_enabled = false;
    world.settings.contact_enabled = false;
    world.params.brownian = 0.0;
    world.params.phoretic_gain = 0.0;
    world.params.pairing_rate = 0.0; // sem nascimentos: os mesmos agentes do princípio ao fim
    world.params.death_probability = 0.0;
    world.params.rft_enabled = rft as u32;
    world.params.motor_amplitude = motor;
    world.params.thermal_kt = kt;
    world.params.joint_coupling = coupling;
    world.params.maintenance_cost =
        std::env::var("MAINT").ok().and_then(|v| v.parse().ok()).unwrap_or(world.params.maintenance_cost);
    let mut rng = ribossome::life::SplitMix(4);
    world.request_seeds(&ribossome::life::seed_requests(2000, [40, 200], true, cfg.sim_size(), &mut rng));
    run(gpu, &mut world, 150); // dobragem
    let before: HashMap<u32, (f32, f32)> = world
        .read_agents_blocking(gpu)
        .iter()
        .filter(|a| a.alive != 0 && a.body_len >= 8)
        .map(|a| (a.id, (a.pos_x, a.pos_y)))
        .collect();
    println!("  estados (livre, ligado, produto): {:?}", states(gpu, &world));
    // Deslocamento a 300 e a 600 passos: balístico (nada numa direção) cresce
    // ×2; difusivo (aleatório) cresce ×√2 ≈ 1,41.
    let disp = |world: &World| -> (f32, f32) {
        let mut d = Vec::new();
        for a in world.read_agents_blocking(gpu).iter().filter(|a| a.alive != 0) {
            if let Some(&(x, y)) = before.get(&a.id) {
                d.push(((a.pos_x - x).powi(2) + (a.pos_y - y).powi(2)).sqrt());
            }
        }
        d.sort_by(f32::total_cmp);
        (d.iter().sum::<f32>() / d.len().max(1) as f32, d.get(d.len() * 9 / 10).copied().unwrap_or(0.0))
    };
    run(gpu, &mut world, 300);
    let (m300, p300) = disp(&world);
    run(gpu, &mut world, 300);
    let (m600, p600) = disp(&world);
    println!(
        "{name:<34} médio {m300:6.1} -> {m600:6.1} (x{:.2})   p90 {p300:6.1} -> {p600:6.1} (x{:.2})   [balístico x2, difusivo x1,41; 1 célula = 30]",
        m600 / m300.max(1e-6),
        p600 / p300.max(1e-6)
    );
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    case(&gpu, "órgãos, kT 0,3", true, 0.0, 0.3, 0.9);
    case(&gpu, "órgãos, sem RFT", false, 0.0, 0.3, 0.9);
    case(&gpu, "órgãos, kT 0 (só músculos)", true, 0.0, 0.0, 0.9);
}

#[allow(dead_code)]
fn states(gpu: &Gpu, world: &World) -> [usize; 3] {
    let s: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.joint_state_buf)).to_vec();
    let mut c = [0usize; 3];
    for v in s {
        c[(v as usize).min(2)] += 1;
    }
    c
}
