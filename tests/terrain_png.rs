//! Terreno em PNG: gravar e carregar devolve os mesmos grãos e fumarolas.
use ribossome::params::{Fumarole, WorldConfig};
use ribossome::world::terrain;

#[test]
fn png_round_trip_keeps_grains_and_fumaroles() {
    let cfg = WorldConfig::TEST;
    let fum = vec![Fumarole::new(0.3, 0.2, 5000.0, 819.0), Fumarole::new(0.7, 0.6, 5000.0, 819.0)];
    let g = terrain::generate(&cfg, 7, &fum);
    let dir = std::env::temp_dir().join("ribossome_terrain_test.png");
    terrain::save_png(&dir, &cfg, &g, &fum).unwrap();
    let (g2, mut f2) = terrain::load_png(&dir, &cfg).unwrap();
    f2.sort_by(|a, b| a.x_frac.partial_cmp(&b.x_frac).unwrap());
    // As fumarolas são desenhadas por cima do terreno: fora delas, igual.
    let n = cfg.grid_size as usize;
    let near_fum = |x: usize, y: usize| {
        fum.iter().any(|f| {
            let (fx, fy) = ((f.x_frac * n as f32) as i64, (f.y_frac * n as f32) as i64);
            (x as i64 - fx).abs() <= 3 && (y as i64 - fy).abs() <= 3
        })
    };
    let mut diff = 0;
    for y in 0..n {
        for x in 0..n {
            if !near_fum(x, y) && g[y * n + x].min(6) != g2[y * n + x] {
                diff += 1;
            }
        }
    }
    assert_eq!(diff, 0, "células diferentes depois de gravar e carregar");
    assert_eq!(f2.len(), 2);
    for (a, b) in fum.iter().zip(&f2) {
        assert!((a.x_frac - b.x_frac).abs() < 0.02 && (a.y_frac - b.y_frac).abs() < 0.02, "{a:?} vs {b:?}");
    }
}
