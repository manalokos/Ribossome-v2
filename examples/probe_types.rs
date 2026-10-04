//! A agregação segrega os tipos de monómero? Mundo sem agentes, sem fluido e
//! sem luz. Mede, ao longo do tempo, o ÍNDICE DE SEGREGAÇÃO dos ativados por
//! tipo: em blocos de 8×8 células, a variância da fração de cada canal entre
//! blocos ÷ a que se esperava se os tipos estivessem misturados ao acaso
//! (binomial). 1 = bem misturados; > 1 = manchas por tipo.
//! AGG (agregação), DIFF (difusão), UNIFORM=1 (sementeira com os 4 canais
//! baralhados em cada célula, em vez das manchas por canal do ruído fractal).
use ribossome::gpu::Gpu;
use ribossome::life::SplitMix;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn index(cells: &[u32], g: usize, shift: u32) -> f64 {
    let b = 8;
    let nb = g / b;
    let mut cnt = vec![[0u64; 4]; nb * nb];
    for y in 0..g {
        for x in 0..g {
            let k = (y / b) * nb + x / b;
            for ch in 0..4 {
                cnt[k][ch] += ((cells[(y * g + x) * 4 + ch] >> shift) & 0xFFFF) as u64;
            }
        }
    }
    let tot: [u64; 4] = std::array::from_fn(|ch| cnt.iter().map(|c| c[ch]).sum());
    let all: u64 = tot.iter().sum();
    let (mut obs, mut exp) = (0f64, 0f64);
    for ch in 0..4 {
        let p = tot[ch] as f64 / all.max(1) as f64;
        for c in &cnt {
            let n: u64 = c.iter().sum();
            if n < 8 {
                continue;
            }
            let f = c[ch] as f64 / n as f64;
            obs += (f - p) * (f - p);
            exp += p * (1.0 - p) / n as f64;
        }
    }
    obs / exp.max(1e-12)
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let g = cfg.grid_size as usize;
    let mut w = World::new(&gpu, cfg, 3);
    w.custom_terrain = Some((vec![0; g * g], vec![0.0; g * g]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    if envf("UNIFORM", 0.0) > 0.0 {
        // Baralha os canais dentro de cada célula (mesma matéria por célula e
        // por estado, tipos ao acaso): tira as manchas por canal da sementeira.
        let mut c = w.read_cells_blocking(&gpu);
        let mut rng = SplitMix(4);
        for cell in c.chunks_mut(4) {
            let (act, spent): (u32, u32) = (cell.iter().map(|v| v & 0xFFFF).sum(), cell.iter().map(|v| v >> 16).sum());
            let mut a = [0u32; 4];
            let mut s = [0u32; 4];
            for _ in 0..act {
                a[(rng.f32() * 4.0) as usize % 4] += 1;
            }
            for _ in 0..spent {
                s[(rng.f32() * 4.0) as usize % 4] += 1;
            }
            for ch in 0..4 {
                cell[ch] = a[ch] | (s[ch] << 16);
            }
        }
        gpu.queue.write_buffer(&w.chem_buf, 0, bytemuck::cast_slice(&c));
    }
    w.settings.fluid_enabled = false;
    w.settings.terrain_enabled = false;
    w.params.uv_strength = 0.0;
    w.params.settle = 0.0;
    w.params.cohesion = 0.0;
    w.params.monomer_pressure = envf("PRESS", 0.0);
    w.params.aggregation = envf("AGG", 1.0);
    w.params.diffusion = envf("DIFF", 4.0);
    w.params.reactivation_rate = 0.0;
    w.params.thermal_activation = 0.0;
    println!("agregação {}, difusão {}, sementeira {}", w.params.aggregation, w.params.diffusion, if envf("UNIFORM", 0.0) > 0.0 { "baralhada" } else { "com manchas por canal" });
    let marks = [0u32, 500, 2000, 6000, 16000];
    let mut done = 0;
    for m in marks {
        while done < m {
            let k = MAX_STEPS_PER_FRAME.min(m - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
        let c = w.read_cells_blocking(&gpu);
        println!("  passo {m:6}: segregação por tipo — ativados {:6.2}, gastos {:6.2}", index(&c, g, 0), index(&c, g, 16));
    }
}
