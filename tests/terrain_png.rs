//! Terreno em PNG: gravar e carregar devolve os mesmos grãos e o mesmo calor.
use ribossome::params::{Fumarole, WorldConfig};
use ribossome::world::terrain;

#[test]
fn png_round_trip_keeps_grains_and_heat() {
    let cfg = WorldConfig::TEST;
    let fum = vec![Fumarole::new(0.3, 0.2, 5000.0, 400.0), Fumarole::new(0.7, 0.6, 2500.0, 400.0)];
    let g = terrain::generate(&cfg, 7, &fum);
    let heat = terrain::rasterize_fumaroles(&cfg, &fum);
    assert!(heat.iter().any(|&h| h > 0.9), "a fumarola de força 5000 tem de chegar a ~1 no centro");
    let path = std::env::temp_dir().join("ribossome_terrain_test.png");
    terrain::save_png(&path, &cfg, &g, &heat).unwrap();
    let (g2, h2) = terrain::load_png(&path, &cfg).unwrap();
    let n = cfg.grid_size as usize;
    for i in 0..n * n {
        assert_eq!(g[i].min(6), g2[i], "grãos diferentes na célula {i}");
        assert!((heat[i].min(1.0) - h2[i]).abs() <= 0.5 / 255.0 + 1e-6, "calor {} vs {} na célula {i}", heat[i], h2[i]);
    }
}
