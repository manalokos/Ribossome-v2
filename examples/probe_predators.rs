//! Quem tem protease e quanto lhe custa? Carrega uma cena (qualquer tamanho
//! de mundo) e, para os agentes com protease, mostra por variante: quantos
//! são, a intensidade, que outra maneira de captar têm (boca, luz,
//! quimiossíntese), a energia que têm contra a dos outros, e o CUSTO de ter a
//! protease aberta por passo (manutenção × força × intensidade × alcance / 40,
//! como em protease_active) comparado com a manutenção do resto do corpo.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let cfg = {
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        WorldConfig { grid_size: g("grid_size", 2048), fluid_size: g("fluid_size", 1024), world_units_per_cell: g("world_units_per_cell", 30), max_agents: g("max_agents", 400_000) }
    };
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &scene).unwrap();
    let agents = w.read_agents_blocking(&gpu);
    let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
    let prot = &w.organ_table[11];
    let var = |v: usize, k: &str| prot.variantes.get(v).and_then(|m| m.get(k)).copied().unwrap_or(0.0);
    let m = w.params.maintenance_cost;
    println!("epoch {}, manutenção {m} por resíduo e passo, food_power {}, protease_direct {}", w.params.epoch, w.params.food_power, w.params.protease_direct);
    // [variante] -> contagens e somas
    #[derive(Default, Clone)]
    struct Acc {
        n: u32,
        gain: f64,
        energy: f64,
        open_cost: f64,
        body_cost: f64,
        mouth: u32,
        photo: u32,
        chemo: u32,
        none: u32,
        res: f64,
    }
    let mut acc = vec![Acc::default(); 6];
    let (mut others, mut e_others) = (0u32, 0.0f64);
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let body = &organs[slot * 64..slot * 64 + (a.body_len as usize).min(64)];
        let ps: Vec<(usize, f32)> = body.iter().filter(|&&c| c != 0 && (c & 0x1F) - 1 == 11).map(|&c| (((c >> 5) & 7) as usize, 2f32.powf(((c >> 8) as f32 - 32.0) / 8.0))).collect();
        if ps.is_empty() {
            others += 1;
            e_others += a.energy as f64;
            continue;
        }
        let has = |t: u16| body.iter().any(|&c| c != 0 && (c & 0x1F) - 1 == t);
        // (classifica pelo protease mais forte do corpo)
        let (v, g) = ps.iter().copied().fold((0, 0.0), |b, x| if x.1 > b.1 { x } else { b });
        let open: f32 = ps.iter().map(|&(v, g)| var(v, "forca") * g * var(v, "alcance") / 40.0).sum();
        let x = &mut acc[v.min(5)];
        x.n += 1;
        x.gain += g as f64;
        x.energy += a.energy as f64;
        x.open_cost += (m * open) as f64;
        x.body_cost += (m * a.body_len as f32) as f64;
        x.res += a.body_len as f64;
        x.mouth += has(0) as u32;
        x.photo += has(10) as u32;
        x.chemo += has(14) as u32;
        x.none += (!has(0) && !has(10) && !has(14)) as u32;
    }
    println!("sem protease: {others} agentes, energia média {:.2}", e_others / others.max(1) as f64);
    for (v, x) in acc.iter().enumerate().filter(|(_, x)| x.n > 0) {
        let k = x.n as f64;
        println!(
            "variante {v} (alcance {:.0}, força {:.0}, {}): {} agentes, {:.1} resíduos, intensidade média ×{:.2}, energia {:.2}; aberta custa {:.4}/passo (o corpo {:.4}); também têm boca {:.0}%, luz {:.0}%, quimio {:.0}%, NADA {:.0}%",
            var(v, "alcance"),
            var(v, "forca"),
            if var(v, "canal") < 0.0 { "sempre ligada" } else { "ligada por sinal" },
            x.n,
            x.res / k,
            x.gain / k,
            x.energy / k,
            x.open_cost / k,
            x.body_cost / k,
            100.0 * x.mouth as f64 / k,
            100.0 * x.photo as f64 / k,
            100.0 * x.chemo as f64 / k,
            100.0 * x.none as f64 / k
        );
    }
}
