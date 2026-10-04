//! Os sensores direcionais viram o agente para a comida? Carrega uma cena
//! (por omissão o autosave) e, de DT em DT passos, mede para cada agente:
//! - rumo antes (deslocamento no intervalo anterior) e depois;
//! - para que lado está a comida (centroide dos ativados num disco de 6
//!   células à volta, no início do intervalo).
//! Índice de viragem = média de sinal(viragem) × sinal(lado da comida):
//! +1 = vira sempre para a comida, 0 = às cegas, −1 = foge dela.
//! Índice de avanço = média do cosseno entre o deslocamento e a direção da
//! comida. Compara quem tem sensor de comida direcional com os cegos.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let g = cfg.grid_size as i32;
    let wpc = cfg.world_units_per_cell as f32;
    let dt = envf("DT").unwrap_or(16.0) as u32;
    let steps = envf("STEPS").unwrap_or(1600.0) as u32;

    // Grupos pelos órgãos (no início): 0 cego (sem sensores de disco),
    // 1 sensor de comida direcional, 2 sensor físico direcional, 3 outro sensor.
    let names = ["cegos (sem sensor de disco)", "sensor de COMIDA direcional", "sensor FÍSICO direcional", "só sensores totais"];
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let mut group: HashMap<u32, usize> = HashMap::new();
    for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let mut has = [false; 16];
        for r in 0..a.body_len as usize {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            if o != 0 {
                has[((o & 0xF) - 1) as usize] = true;
            }
        }
        let gi = if has[8] { 1 } else if has[9] { 2 } else if has[2] || has[3] { 3 } else { 0 };
        group.insert(a.id, gi);
    }

    // Direção da comida: centroide dos ativados no disco de raio 6 células.
    let food_dir = |cells: &[u32], x: f32, y: f32| -> Option<[f32; 2]> {
        let (cx, cy) = ((x / wpc) as i32, (y / wpc) as i32);
        let (mut sx, mut sy, mut tot) = (0f32, 0f32, 0f32);
        for dy in -6i32..=6 {
            for dx in -6i32..=6 {
                if dx * dx + dy * dy > 36 {
                    continue;
                }
                let (xx, yy) = (cx + dx, cy + dy);
                if xx < 0 || yy < 0 || xx >= g || yy >= g {
                    continue;
                }
                let i = (yy * g + xx) as usize;
                let c = (0..4).map(|ch| cells[i * 4 + ch] & 0xFFFF).sum::<u32>() as f32;
                sx += c * dx as f32;
                sy += c * dy as f32;
                tot += c;
            }
        }
        // Só conta se houver comida e uma assimetria clara.
        let n = (sx * sx + sy * sy).sqrt();
        (tot >= 4.0 && n / tot > 0.5).then(|| [sx / n, sy / n])
    };

    // Por grupo: Σ viragem×lado, n; Σ cos(deslocamento, comida), n; percurso.
    let mut turn = [(0f64, 0u64); 4];
    let mut adv = [(0f64, 0u64); 4];
    let mut speed = [(0f64, 0u64); 4];
    // id -> (posição anterior, rumo anterior, direção da comida no início do intervalo)
    let mut prev: HashMap<u32, ([f32; 2], Option<[f32; 2]>, Option<[f32; 2]>)> = HashMap::new();
    let mut done = 0;
    while done < steps {
        let mut left = dt;
        while left > 0 {
            let k = MAX_STEPS_PER_FRAME.min(left);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            left -= k;
        }
        done += dt;
        let cells = w.read_cells_blocking(&gpu);
        let mut now = HashMap::new();
        for a in w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0) {
            let Some(&gi) = group.get(&a.id) else { continue };
            let p = [a.pos_x, a.pos_y];
            let mut heading = None;
            if let Some((pp, ph, pf)) = prev.get(&a.id) {
                let d = [p[0] - pp[0], p[1] - pp[1]];
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                speed[gi].0 += (len / wpc) as f64;
                speed[gi].1 += 1;
                if len > 0.1 * wpc {
                    let h = [d[0] / len, d[1] / len];
                    heading = Some(h);
                    if let Some(f) = pf {
                        adv[gi].0 += (h[0] * f[0] + h[1] * f[1]) as f64;
                        adv[gi].1 += 1;
                        if let Some(h0) = ph {
                            // Viragem (rumo anterior -> rumo novo) e lado da comida, pelo produto externo.
                            let t = h0[0] * h[1] - h0[1] * h[0];
                            let s = h0[0] * f[1] - h0[1] * f[0];
                            if t.abs() > 0.02 && s.abs() > 0.2 {
                                turn[gi].0 += (t.signum() * s.signum()) as f64;
                                turn[gi].1 += 1;
                            }
                        }
                    }
                }
            }
            now.insert(a.id, (p, heading, food_dir(&cells, p[0], p[1])));
        }
        prev = now;
    }
    println!("{steps} passos, intervalos de {dt}:");
    for gi in 0..4 {
        let n = group.values().filter(|&&x| x == gi).count();
        println!(
            "  {:30} {n:6} agentes: viragem para a comida {:+.3} (n={}), avanço para a comida {:+.3} (n={}), {:.2} células por intervalo",
            names[gi],
            turn[gi].0 / turn[gi].1.max(1) as f64,
            turn[gi].1,
            adv[gi].0 / adv[gi].1.max(1) as f64,
            adv[gi].1,
            speed[gi].0 / speed[gi].1.max(1) as f64
        );
    }
}
