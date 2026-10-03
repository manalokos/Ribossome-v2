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
    terrain::save_png(&path, &cfg, &g, &heat, None).unwrap();
    let (g2, h2, c2) = terrain::load_png(&path, &cfg).unwrap();
    assert!(c2.is_none(), "sem verde, a química segue o calor");
    let n = cfg.grid_size as usize;
    for i in 0..n * n {
        assert_eq!(g[i].min(6), g2[i], "grãos diferentes na célula {i}");
        assert!((heat[i].min(1.0) - h2[i]).abs() <= 0.5 / 255.0 + 1e-6, "calor {} vs {} na célula {i}", heat[i], h2[i]);
    }

    // Com química própria (verde): uma exsudação fria onde não há calor.
    let chem: Vec<f32> = (0..n * n).map(|i| if i % 7 == 0 { 0.6 } else { 0.0 }).collect();
    terrain::save_png(&path, &cfg, &g, &heat, Some(&chem)).unwrap();
    let (_, h3, c3) = terrain::load_png(&path, &cfg).unwrap();
    let c3 = c3.expect("o verde tem de ser lido");
    for i in 0..n * n {
        // Um píxel exatamente cinzento (r = g = b) não conta; aqui é raro.
        if heat[i] == 0.0 && chem[i] > 0.0 && g[i] == 0 {
            assert!((c3[i] - 0.6).abs() <= 0.5 / 255.0 + 1e-6, "química {} na célula {i}", c3[i]);
            assert_eq!(h3[i], 0.0, "uma exsudação fria não aquece");
        }
    }
}
