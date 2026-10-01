//! Geração inicial do terreno (no CPU, determinista).
//!
//! O v3 não gerava terreno no arranque: vinha do autosave/snapshots ou de
//! ruído pedido na UI. Isto é um gerador provisório de fundo marinho: um
//! leito rochoso ondulado em baixo e blocos espalhados, com entulho nas
//! margens. A física dos grãos (relaxação) dá-lhe depois a forma natural.

use crate::params::{Fumarole, WorldConfig};

/// Quanta numa célula de rocha maciça (= GAMMA_PILE_MAX do v3).
const ROCK: u32 = 6;

/// Ruído de valor 2D, suave (interpolação quíntica), em [0, 1].
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let hash = |ix: i32, iy: i32| -> f32 {
        let mut h = (ix as u32).wrapping_mul(0x8DA6_B343)
            ^ (iy as u32).wrapping_mul(0xD816_3841)
            ^ seed.wrapping_mul(0xCB1A_B31F);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297A_2D39);
        h ^= h >> 15;
        h as f32 / u32::MAX as f32
    };
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fy));
    let (ix, iy) = (x0 as i32, y0 as i32);
    let a = hash(ix, iy) + (hash(ix + 1, iy) - hash(ix, iy)) * u;
    let b = hash(ix, iy + 1) + (hash(ix + 1, iy + 1) - hash(ix, iy + 1)) * u;
    a + (b - a) * v
}

/// Ruído fractal (5 oitavas), em [0, 1].
fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut total) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..5 {
        sum += value_noise(x * freq, y * freq, seed.wrapping_add(o * 7919)) * amp;
        total += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / total
}

/// Quanta de gamma por célula do ambiente. Cada fumarola fica numa
/// chaminé aberta no leito (senão o calor ficava enterrado na rocha).
pub fn generate(cfg: &WorldConfig, seed: u32, fumaroles: &[Fumarole]) -> Vec<u32> {
    let n = cfg.grid_size as usize;
    let mut g = vec![0u32; n * n];
    for y in 0..n {
        let yf = y as f32 / n as f32;
        for x in 0..n {
            let xf = x as f32 / n as f32;
            // Leito: 3–9% da altura, ondulado.
            let floor = 0.03 + 0.06 * fbm(xf * 4.0, 0.5, seed);
            // Blocos: só na metade de baixo, rarefazendo para cima.
            let rocks = fbm(xf * 10.0, yf * 10.0, seed ^ 0xA5A5_5A5A);
            let rock_limit = 0.70 + 0.45 * yf;
            let q = if yf < floor {
                ROCK
            } else if yf < floor + 0.006 {
                // Margem de entulho por cima do leito.
                if fbm(xf * 60.0, yf * 60.0, seed ^ 0x51ED) > 0.5 { 2 } else { 1 }
            } else if yf < 0.55 && rocks > rock_limit {
                ROCK
            } else if yf < 0.55 && rocks > rock_limit - 0.02 {
                1
            } else {
                0
            };
            g[y * n + x] = q;
        }
    }
    let cell_w = cfg.world_units_per_cell as f32;
    for f in fumaroles.iter().filter(|f| f.enabled != 0) {
        let (cx, cy) = (f.x_frac * n as f32, f.y_frac * n as f32);
        let r = (f.spread / cell_w).max(2.0);
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                // Disco à volta da fumarola e coluna por cima dela até sair do leito.
                let in_disc = dx * dx + dy * dy < r * r;
                let in_chimney = dx.abs() < r * 0.6 && dy > 0.0 && (y as f32) < cy + 0.12 * n as f32;
                if in_disc || in_chimney {
                    g[y * n + x] = 0;
                }
            }
        }
    }
    g
}
