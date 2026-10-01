//! Efeito de mutações: encontra pares AVÔ–NETO vivos (mesma cadeia: o
//! filho é o complemento reverso, o neto volta à cadeia do avô) cujos
//! genomas diferem, imprime genoma e proteína com as diferenças assinaladas
//! e grava um PNG com os pares lado a lado (avô à esquerda, neto à direita).
//! Variáveis: SEEDS, STEPS, MUT (taxa de mutação), PAIRS, OUT.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::life::amino::{AA_LETTERS, BASES, translate};
use ribossome::params::{Agent, WorldConfig};
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn genome(words: &[u32], slot: usize, len: u32) -> Vec<u8> {
    (0..len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect()
}

/// Duas sequências alinhadas por posição, com ^ por baixo das diferenças.
fn diff_line(a: &str, b: &str) -> String {
    let n = a.len().max(b.len());
    (0..n).map(|i| if a.as_bytes().get(i) == b.as_bytes().get(i) { ' ' } else { '^' }).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.seed_matter(&gpu, 1);
    world.params.mutation_rate = env("MUT", world.params.mutation_rate);
    let mut rng = ribossome::life::SplitMix(7);
    let reqs = ribossome::life::seed_requests(env("SEEDS", 3000), [30, 200], true, cfg.sim_size(), &mut rng);
    world.request_seeds(&reqs);
    let steps: u32 = env("STEPS", 1500);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let agents: Vec<Agent> = world.read_agents_blocking(&gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.genomes_buf)).to_vec();
    let by_id: HashMap<u32, usize> =
        agents.iter().enumerate().filter(|(_, a)| a.alive != 0).map(|(s, a)| (a.id, s)).collect();

    let want: usize = env("PAIRS", 6);
    let mut pairs = Vec::new();
    for (s, a) in agents.iter().enumerate() {
        if a.alive == 0 || a.body_len < 10 {
            continue;
        }
        let Some(&ps) = by_id.get(&a.parent) else { continue };
        let Some(&gs) = by_id.get(&agents[ps].parent) else { continue };
        let ga = genome(&words, s, a.gene_len);
        let gg = genome(&words, gs, agents[gs].gene_len);
        if ga != gg && agents[gs].body_len >= 10 {
            pairs.push((gs, s));
            if pairs.len() >= want {
                break;
            }
        }
    }
    println!("{} pares avô–neto com genomas diferentes (mostro {})\n", pairs.len(), pairs.len().min(want));

    let tile = 256u32;
    let cap = Capture::new(&gpu, &world, tile);
    let rows = pairs.len().max(1) as u32;
    let mut sheet = vec![0u8; (tile * 2 * tile * rows * 4) as usize];
    for (r, &(gs, s)) in pairs.iter().enumerate() {
        let (g, c) = (&agents[gs], &agents[s]);
        let gen_g: String = genome(&words, gs, g.gene_len).iter().map(|&b| BASES[b as usize]).collect();
        let gen_c: String = genome(&words, s, c.gene_len).iter().map(|&b| BASES[b as usize]).collect();
        let prot = |gn: &str| -> String {
            let bases: Vec<u8> = gn.chars().map(|ch| BASES.iter().position(|&x| x == ch).unwrap() as u8).collect();
            translate(&bases).iter().map(|&i| AA_LETTERS[i as usize]).collect()
        };
        let (pg, pc) = (prot(&gen_g), prot(&gen_c));
        println!(
            "par {r}: avô {} bases / {} resíduos  →  neto {} bases / {} resíduos",
            g.gene_len, g.body_len, c.gene_len, c.body_len
        );
        println!("  RNA avô   {gen_g}");
        println!("  RNA neto  {gen_c}");
        println!("            {}", diff_line(&gen_g, &gen_c));
        println!("  prot avô  {pg}");
        println!("  prot neto {pc}");
        println!("            {}\n", diff_line(&pg, &pc));
        for (col, (a, slot)) in [(g, gs), (c, s)].iter().enumerate() {
            cap.view.focus.set(*slot as u32);
            let cam = Camera { center: [a.pos_x, a.pos_y], zoom: 1.6 };
            let rgba = cap.render(&gpu, &world, &cam, 0, 0.25);
            for y in 0..tile {
                let src = (y * tile * 4) as usize;
                let dst = (((r as u32 * tile + y) * tile * 2 + col as u32 * tile) * 4) as usize;
                sheet[dst..dst + (tile * 4) as usize].copy_from_slice(&rgba[src..src + (tile * 4) as usize]);
            }
        }
    }
    let out = std::path::PathBuf::from(env("OUT", String::from("target/mutants.png")));
    let file = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut enc = png::Encoder::new(file, tile * 2, tile * rows);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&sheet).unwrap();
    println!("{}", out.display());
}
