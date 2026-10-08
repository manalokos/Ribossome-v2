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
// VERMELHO = CALOR e VERDE = QUÍMICA (redutor) das fumarolas, POR PÍXEL e
// independentes: calor = r/255 e química = g/255 vezes a força de uma
// fumarola v3 no centro (`FUMAROLE_PIXEL_STRENGTH`). Vermelho puro =
// fumarola quente; amarelo = quente e química (fumarola negra); verde =
// exsudação fria (química sem calor). Um píxel CINZENTO (r = g = b) não faz
// nada, por isso imagens em cinzentos também servem (só terreno). Se a
// imagem não tiver verde nenhum, a química segue o calor (como antes). A imagem pode ter
// qualquer tamanho: é reamostrada (vizinho mais próximo) para a grelha.
// A linha de cima da imagem é o cimo do mundo (+y no mundo = cima no ecrã).

/// Força de aquecimento de um píxel vermelho puro (= centro da fumarola v3).
pub const FUMAROLE_PIXEL_STRENGTH: f32 = 5000.0;

/// Lê um PNG. Devolve (grãos por célula, calor por célula, química por
/// célula ou None se a imagem não tiver verde), calor e química em fração de
/// `FUMAROLE_PIXEL_STRENGTH`, à resolução da grelha.
pub type TerrainImage = (Vec<u32>, Vec<f32>, Option<Vec<f32>>);
pub fn load_png(path: &std::path::Path, cfg: &WorldConfig) -> Result<TerrainImage, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| format!("invalid PNG: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("invalid PNG: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("indexed PNG not supported (save it as RGB)".into()),
    };
    let n = cfg.grid_size as usize;
    let mut g = vec![0u32; n * n];
    let mut heat = vec![0f32; n * n];
    let mut chem = vec![0f32; n * n];
    for y in 0..n {
        // Linha 0 da imagem = cimo do mundo.
        let iy = ((n - 1 - y) * h) / n;
        for x in 0..n {
            let ix = (x * w) / n;
            let i = (iy * w + ix) * ch;
            let (r, gg, b) = if ch < 3 { (buf[i], buf[i], buf[i]) } else { (buf[i], buf[i + 1], buf[i + 2]) };
            g[y * n + x] = ((b as f32 / 255.0) * ROCK as f32).round() as u32;
            if !(r == gg && gg == b) {
                heat[y * n + x] = r as f32 / 255.0;
                chem[y * n + x] = gg as f32 / 255.0;
            }
        }
    }
    let has_chem = chem.iter().any(|&c| c > 0.0);
    Ok((g, heat, has_chem.then_some(chem)))
}

/// Grava o terreno em PNG: azul = grãos, vermelho = calor, verde = química
/// (fração de `FUMAROLE_PIXEL_STRENGTH`, à resolução da grelha; sem química
/// própria o verde fica a 0 e, ao ler, a química volta a seguir o calor).
pub fn save_png(path: &std::path::Path, cfg: &WorldConfig, gamma: &[u32], heat: &[f32], chem: Option<&[f32]>) -> Result<(), String> {
    let n = cfg.grid_size as usize;
    let mut rgb = vec![0u8; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let o = ((n - 1 - y) * n + x) * 3;
            rgb[o] = (heat[y * n + x].clamp(0.0, 1.0) * 255.0).round() as u8;
            if let Some(c) = chem {
                rgb[o + 1] = (c[y * n + x].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            rgb[o + 2] = ((gamma[y * n + x].min(ROCK) * 255) / ROCK) as u8;
        }
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), n as u32, n as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut w| w.write_image_data(&rgb)).map_err(|e| e.to_string())
}

/// Calor das fumarolas PONTUAIS (as do terreno gerado e dos sliders), à
/// resolução da grelha, em fração de `FUMAROLE_PIXEL_STRENGTH`. Mesmo perfil
/// que o shader usava: força·(1 − d/r)² dentro do raio.
pub fn rasterize_fumaroles(cfg: &WorldConfig, fumaroles: &[Fumarole]) -> Vec<f32> {
    let n = cfg.grid_size as usize;
    let mut heat = vec![0f32; n * n];
    let cell_world = cfg.world_units_per_cell as f32;
    for f in fumaroles.iter().filter(|f| f.enabled != 0 && f.strength > 0.0) {
        let (cx, cy) = (f.x_frac * n as f32, f.y_frac * n as f32);
        let r = (f.spread / cell_world).max(1.0);
        let (x0, x1) = (((cx - r).floor().max(0.0)) as usize, ((cx + r).ceil() as usize).min(n));
        let (y0, y1) = (((cy - r).floor().max(0.0)) as usize, ((cy + r).ceil() as usize).min(n));
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                if d < r {
                    let w0 = 1.0 - d / r;
                    heat[y * n + x] += f.strength * w0 * w0 / FUMAROLE_PIXEL_STRENGTH;
                }
            }
        }
    }
    heat
}
