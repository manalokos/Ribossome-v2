//! O órgão quiral troca o lado das dobras? Dois corpos de metioninas (ângulo
//! de repouso −0,52 rad cada): um liso e um com um órgão quiral a meio
//! (histidina + metionina). Lê a forma (body_pos) e soma os ângulos de
//! viragem antes e depois do meio. Espera-se: liso = tudo para o mesmo lado;
//! com quiral = a segunda metade para o lado contrário (um S).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let mut w = World::new(&gpu, cfg, 3);
    w.seed_matter(&gpu, 3);
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.spawn_energy = 50.0;
    let m = "AUG";
    let designs = [
        ("liso", format!("{}UAA", m.repeat(15))),
        // CAU = histidina (promotor), AUG = metionina (modificador), GAA = intensidade.
        ("com quiral a meio", format!("{}CAUAUGGAA{}UAA", m.repeat(6), m.repeat(6))),
    ];
    let s = cfg.sim_size();
    let reqs: Vec<SpawnRequest> =
        designs.iter().enumerate().map(|(i, (_, g))| SpawnRequest::with_genome(s * (0.3 + 0.4 * i as f32), s * 0.5, &bases(g))).collect();
    w.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 30);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.body_pos_buf)).to_vec();
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let name = if a.pos_x < s * 0.5 { designs[0].0 } else { designs[1].0 };
        let p = &pos[slot * 64..slot * 64 + a.body_len as usize];
        let dir: Vec<f32> = p.windows(2).map(|q| (q[1][1] - q[0][1]).atan2(q[1][0] - q[0][0])).collect();
        let turn: Vec<f32> = dir
            .windows(2)
            .map(|d| {
                let mut t = d[1] - d[0];
                while t > std::f32::consts::PI {
                    t -= std::f32::consts::TAU;
                }
                while t < -std::f32::consts::PI {
                    t += std::f32::consts::TAU;
                }
                t
            })
            .collect();
        let half = turn.len() / 2;
        println!(
            "{name}: {} resíduos; viragem na 1.ª metade {:+.2} rad, na 2.ª metade {:+.2} rad; ângulos: {}",
            a.body_len,
            turn[..half].iter().sum::<f32>(),
            turn[half..].iter().sum::<f32>(),
            turn.iter().map(|t| format!("{t:+.2}")).collect::<Vec<_>>().join(" ")
        );
    }
}
