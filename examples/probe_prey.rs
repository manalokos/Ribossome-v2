//! Quanto vale uma presa? Para a cena (por omissão o autosave): por tamanho
//! do corpo, quantos agentes, energia média, complementos capturados (os
//! monómeros ativados que saem quando morre) e as frações de resíduos que
//! cada família de protease corta.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let bodies: Vec<u8> = gpu.read_buffer_blocking(&w.bodies_buf);
    let target: Vec<u32> = w.amino.iter().map(|r| r.protease_alvo as u32).collect();
    let pro = w.amino.iter().position(|r| r.letra == "P").unwrap();
    // classes de tamanho: <= 10, 11..18, > 18 resíduos
    let mut acc = [[0f64; 8]; 3];
    for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0 && a.body_len > 0) {
        let c = if a.body_len <= 10 { 0 } else if a.body_len <= 18 { 1 } else { 2 };
        let n = a.body_len as usize;
        let mut t = [0f64; 4];
        for k in 0..n {
            let aa = bodies[slot * 64 + k] as usize;
            for f in 0..3 {
                t[f] += ((target[aa] >> f) & 1) as f64;
            }
            t[3] += (aa == pro) as u32 as f64;
        }
        let e = &mut acc[c];
        e[0] += 1.0;
        e[1] += a.energy as f64;
        e[2] += a.pair_count as f64;
        e[3] += a.gene_len as f64;
        for f in 0..4 {
            e[4 + f] += t[f] / n as f64;
        }
    }
    println!("epoch {}, energia por monómero {}", w.params.epoch, w.params.food_power);
    for (c, name) in ["corpos até 10 resíduos", "11 a 18", "mais de 18"].iter().enumerate() {
        let e = acc[c];
        let n = e[0].max(1.0);
        println!(
            "  {name:24} {:6.0} agentes: energia {:5.1}, complementos capturados {:5.1} de {:5.1} bases; alvos: serina {:4.1}%, cisteína {:4.1}%, aspártica {:4.1}%; prolina {:4.1}%",
            e[0], e[1] / n, e[2] / n, e[3] / n, 100.0 * e[4] / n, 100.0 * e[5] / n, 100.0 * e[6] / n, 100.0 * e[7] / n
        );
    }
}
