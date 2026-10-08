//! ARMAZENAMENTO: nasce-se com energia a mais (50) e vê-se com quanto cada
//! corpo fica ao fim de um passo (a energia é cortada pela capacidade). Um
//! corpo de 10 glicinas, um de 10 fenilalaninas, e os mesmos de glicina com um
//! órgão de armazenamento de cada variante (C+M, C+W, Q+M, Q+W, W+M, W+W).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let gly = "GGU ".repeat(10);
    let organ = |pre: &str, modi: &str| bases(&format!("AUG {pre} {modi} GAA {gly} UAA"));
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("10 glicinas", bases(&format!("AUG {gly} UAA"))),
        ("10 fenilalaninas", bases(&format!("AUG {} UAA", "UUU ".repeat(10)))),
        ("glicinas + armazenamento variante 0 (C M)", organ("UGU", "AUG")),
        ("glicinas + armazenamento variante 1 (C W)", organ("UGU", "UGG")),
        ("glicinas + armazenamento variante 2 (Q M)", organ("CAA", "AUG")),
        ("glicinas + armazenamento variante 3 (Q W)", organ("CAA", "UGG")),
        ("glicinas + armazenamento variante 4 (W M)", organ("UGG", "AUG")),
        ("glicinas + armazenamento variante 5 (W W)", organ("UGG", "UGG")),
    ];
    let mut w = World::new(&gpu, cfg, 3);
    w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.uv_strength = 0.0;
    w.params.spawn_energy = 50.0;
    let reqs: Vec<SpawnRequest> = cases.iter().enumerate().map(|(i, (_, g))| SpawnRequest::with_genome(3000.0 + 2000.0 * i as f32, 5000.0, g)).collect();
    w.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 3);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    println!("{:55} {:>9} {:>9}", "corpo", "resíduos", "energia");
    for (i, (name, _)) in cases.iter().enumerate() {
        let x = 3000.0 + 2000.0 * i as f32;
        match agents.iter().find(|a| a.alive != 0 && (a.pos_x - x).abs() < 300.0 && (a.pos_y - 5000.0).abs() < 300.0) {
            Some(a) => println!("{name:55} {:9} {:9.2}", a.body_len, a.energy),
            None => println!("{name:55} (não nasceu)"),
        }
    }
}
