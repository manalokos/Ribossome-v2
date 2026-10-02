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

// ---- Terreno de/para imagem PNG -------------------------------------------
//
// Canal AZUL = terreno: 0 = água (0 grãos), 255 = rocha maciça (ROCK grãos),
// linear (azul fraco = entulho, 1–2; a partir de ~metade = rocha, >= 3).
// VERMELHO = fumarolas: cada mancha contígua claramente vermelha (vermelho
// alto, pouco verde, vermelho bem acima do azul) é uma fumarola, no centro
// dela. Uma imagem em cinzentos também serve (azul = cinzento; um cinzento
// nunca conta como vermelho). A imagem pode ter qualquer tamanho: é
// reamostrada (vizinho mais próximo) para a grelha.
// A linha de cima da imagem é o cimo do mundo (+y no mundo = cima no ecrã).

fn is_fumarole_px(r: u8, g: u8, b: u8) -> bool {
    r >= 200 && g <= 60 && r as u32 > b as u32 + 30
}

/// Lê um PNG e devolve (grãos por célula, fumarolas).
pub fn load_png(path: &std::path::Path, cfg: &WorldConfig) -> Result<(Vec<u32>, Vec<Fumarole>), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| format!("PNG inválido: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("PNG inválido: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("PNG indexado não suportado (grava em cinzento ou RGB)".into()),
    };
    let px = |x: usize, y: usize| -> (u8, u8, u8) {
        let i = (y * w + x) * ch;
        if ch < 3 { (buf[i], buf[i], buf[i]) } else { (buf[i], buf[i + 1], buf[i + 2]) }
    };

    // Fumarolas: manchas vermelhas contíguas na imagem original.
    let mut seen = vec![false; w * h];
    let mut fumaroles = Vec::new();
    for y0 in 0..h {
        for x0 in 0..w {
            let (r, g, b) = px(x0, y0);
            if seen[y0 * w + x0] || !is_fumarole_px(r, g, b) {
                continue;
            }
            let (mut sx, mut sy, mut n) = (0f64, 0f64, 0f64);
            let mut stack = vec![(x0, y0)];
            seen[y0 * w + x0] = true;
            while let Some((x, y)) = stack.pop() {
                sx += x as f64;
                sy += y as f64;
                n += 1.0;
                let nb = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
                for (nx, ny) in nb {
                    if nx < w && ny < h && !seen[ny * w + nx] {
                        let (r, g, b) = px(nx, ny);
                        if is_fumarole_px(r, g, b) {
                            seen[ny * w + nx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            let d = Fumarole::v3_default();
            fumaroles.push(Fumarole::new(
                ((sx / n + 0.5) / w as f64) as f32,
                (1.0 - (sy / n + 0.5) / h as f64) as f32,
                d.strength,
                d.spread,
            ));
        }
    }

    let n = cfg.grid_size as usize;
    let mut g = vec![0u32; n * n];
    for y in 0..n {
        // Linha 0 da imagem = cimo do mundo.
        let iy = ((n - 1 - y) * h) / n;
        for x in 0..n {
            let ix = (x * w) / n;
            let (_, _, b) = px(ix, iy);
            g[y * n + x] = ((b as f32 / 255.0) * ROCK as f32).round() as u32;
        }
    }
    Ok((g, fumaroles))
}

/// Grava o terreno em PNG (terreno no azul, fumarolas a vermelho).
pub fn save_png(path: &std::path::Path, cfg: &WorldConfig, gamma: &[u32], fumaroles: &[Fumarole]) -> Result<(), String> {
    let n = cfg.grid_size as usize;
    let mut rgb = vec![0u8; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let v = ((gamma[y * n + x].min(ROCK) * 255) / ROCK) as u8;
            let o = ((n - 1 - y) * n + x) * 3;
            rgb[o + 2] = v;
        }
    }
    for f in fumaroles.iter().filter(|f| f.enabled != 0) {
        let cx = (f.x_frac * n as f32) as i64;
        let cy = ((1.0 - f.y_frac) * n as f32) as i64;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && y >= 0 && (x as usize) < n && (y as usize) < n {
                    let o = (y as usize * n + x as usize) * 3;
                    // Vermelho por cima do terreno (o azul fica: o terreno
                    // lê-se na mesma; a fumarola fica na água se o azul for 0).
                    rgb[o] = 255;
                    rgb[o + 1] = 0;
                }
            }
        }
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), n as u32, n as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut w| w.write_image_data(&rgb)).map_err(|e| e.to_string())
}
