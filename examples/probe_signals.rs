//! Os 4 canais de sinais estão a ser usados? Carrega uma cena (por omissão o
//! autosave), corre STEPS passos e mede:
//! - a atividade de cada canal (média do módulo do sinal por resíduo e a
//!   fração dos agentes com algum resíduo acima de 0,05 nesse canal);
//! - os relés da população: função × canal de entrada -> canal de saída
//!   (escolhidos pelo 3.º codão).
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let agents = w.read_agents_blocking(&gpu);
    let sig: Vec<f32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let names = ["α", "β", "γ", "δ"];
    let (mut sum, mut res, mut users, mut alive) = ([0f64; 4], 0u64, [0u32; 4], 0u32);
    let mut relays: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut with_relay = 0u32;
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        alive += 1;
        let mut any = [false; 4];
        let mut has_relay = false;
        for k in 0..a.body_len as usize {
            for (ch, used) in any.iter_mut().enumerate() {
                let v = sig[(slot * 64 + k) * 4 + ch].abs();
                sum[ch] += v as f64;
                *used |= v > 0.05;
            }
            res += 1;
            let o = (organs[slot * 32 + k / 2] >> ((k % 2) * 16)) & 0xFFFF;
            if o != 0 && (o & 0x1F) - 1 == 6 {
                has_relay = true;
                let gi = o >> 8;
                *relays.entry(((o >> 5) & 0x7, gi & 3, (gi >> 2) & 3)).or_default() += 1;
            }
        }
        for ch in 0..4 {
            users[ch] += any[ch] as u32;
        }
        with_relay += has_relay as u32;
    }
    println!("epoch {}, {alive} agentes, modo de sinais {}", w.params.epoch, w.params.signal_mode);
    for ch in 0..4 {
        println!(
            "  canal {}: |sinal| médio por resíduo {:.4}; {:.1}% dos agentes têm o canal ativo",
            names[ch],
            sum[ch] / res.max(1) as f64,
            100.0 * users[ch] as f32 / alive.max(1) as f32
        );
    }
    let funcs = ["switch", "cópia", "inversor", "gate fecha", "gate abre", "limiar"];
    let total: u32 = relays.values().sum();
    println!("relés: {total} em {with_relay} agentes ({:.1}% dos agentes)", 100.0 * with_relay as f32 / alive.max(1) as f32);
    let mut v: Vec<_> = relays.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for ((f, cin, cout), n) in v.iter().take(12) {
        println!("  {:10} {} -> {}   {n:5} ({:.0}%)", funcs[(*f as usize).min(5)], names[*cin as usize], names[*cout as usize], 100.0 * *n as f32 / total.max(1) as f32);
    }
}
