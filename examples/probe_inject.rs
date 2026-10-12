//! Um predador bem feito vinga neste mundo? Carrega uma cena (qualquer
//! tamanho), procura ao acaso um genoma curto em que UMA forma é predador (só
//! uma protease da variante VARIANT, por omissão 0 = contacto, sempre ligada)
//! e A OUTRA forma tem uma boca (para a linhagem ter de que viver nas duas
//! gerações), semeia N cópias dele junto de agentes vivos e segue a linhagem:
//! de tantos em tantos passos, quantos agentes do mundo têm protease, a sua
//! energia, e as mortes por protease. Se sobem, as regras deixam um predador
//! vingar e o que falta é a mutação certa; se descem, não compensa.
//! PARAMS=nome=valor,... muda parâmetros antes de correr.
use ribossome::gpu::Gpu;
use ribossome::life::organs::{ORGAN_SYMBOLS, translate_organs};
use ribossome::life::table;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::species::reverse_complement;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let variant = envf("VARIANT", 0.0) as u8;
    let count = envf("N", 1500.0) as usize;
    let steps = envf("STEPS", 30000.0) as u32;
    let every = envf("EVERY", 3000.0) as u32;
    let gpu = Gpu::new_headless().unwrap();
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let cfg = {
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        WorldConfig { grid_size: g("grid_size", 2048), fluid_size: g("fluid_size", 1024), world_units_per_cell: g("world_units_per_cell", 30), max_agents: g("max_agents", 400_000) }
    };
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &scene).unwrap();
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let code = table::code_to_gpu(&w.organ_code);
    let rs = w.params.require_start != 0;
    // Procura: AUG ... CAU (para as duas cadeias começarem por AUG), miolo ao acaso.
    let mut rng = ribossome::life::SplitMix(envf("SEED", 7.0) as u64);
    let organs_of = |g: &[u8]| -> Vec<(u8, u8, u8)> { translate_organs(g, rs, &code).iter().filter_map(|r| r.organ).collect() };
    let mut found: Option<Vec<u8>> = None;
    let want_other = envf("OTHER", 0.0) as u8; // órgão que a outra forma tem de ter (0 = boca)
    for _ in 0..20_000_000u64 {
        let mid = 12 + (rng.next_u64() % 4) as usize * 3;
        let mut g = vec![0u8, 1, 2];
        g.extend((0..mid).map(|_| (rng.next_u64() & 3) as u8));
        g.extend([3u8, 0, 1]); // CAU
        let a = organs_of(&g);
        if a.len() != 1 || a[0].0 != 11 || a[0].1 != variant || !(28..=40).contains(&a[0].2) {
            continue;
        }
        let rc = reverse_complement(&g);
        let b = organs_of(&rc);
        if b.len() == 1 && b[0].0 == want_other && translate_organs(&rc, rs, &code).len() >= 3 && translate_organs(&g, rs, &code).len() >= 3 {
            found = Some(g);
            break;
        }
    }
    let genome = found.expect("não encontrei um genoma com essas duas formas");
    let letters = |g: &[u8]| g.iter().map(|&b| "AUGC".as_bytes()[b as usize] as char).collect::<String>();
    let show = |g: &[u8]| {
        let body = translate_organs(g, rs, &code);
        let o: String = body.iter().filter_map(|r| r.organ.map(|(t, v, i)| format!("{}(variante {v}, intensidade {i})", ORGAN_SYMBOLS[t as usize]))).collect();
        format!("{} resíduos, {o}", body.len())
    };
    println!("genoma {} ({} bases): forma A {}; forma B {}", letters(&genome), genome.len(), show(&genome), show(&reverse_complement(&genome)));
    // Semeia junto de agentes vivos (onde há gente para caçar).
    let agents = w.read_agents_blocking(&gpu);
    let alive: Vec<usize> = (0..agents.len()).filter(|&s| agents[s].alive != 0).collect();
    let mut reqs = Vec::new();
    for _ in 0..count {
        let a = &agents[alive[(rng.next_u64() % alive.len() as u64) as usize]];
        reqs.push(SpawnRequest::with_genome(a.pos_x + 60.0 * (rng.f32() - 0.5), a.pos_y + 60.0 * (rng.f32() - 0.5), &genome));
    }
    w.request_seeds(&reqs);
    println!("epoch {}: {} vivos; semeados {count}; protease_power {}, direta {}, manutenção {}", w.params.epoch, alive.len(), w.params.protease_power, w.params.protease_direct, w.params.maintenance_cost);
    println!("  passos   vivos   com protease (energia)   da linhagem semeada: forma A / forma B   mortes por protease no intervalo");
    let mut done = 0;
    let mut last_bites = None;
    while done <= steps {
        let ag = w.read_agents_blocking(&gpu);
        let org: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
        let counters: Vec<u32> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.life_counters_buf));
        let bite: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.contact_disp_buf)).to_vec();
        let (mut biting, mut took) = (0u32, 0f64);
        let (mut n, mut pr, mut e, mut fa, mut fb) = (0u32, 0u32, 0f64, 0u32, 0u32);
        for (slot, a) in ag.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            n += 1;
            let body = &org[slot * 64..slot * 64 + (a.body_len as usize).min(64)];
            let has = body.iter().any(|&c| c != 0 && (c & 0x1F) - 1 == 11);
            if has {
                pr += 1;
                e += a.energy as f64;
                if bite[slot][2] > 0.0 {
                    biting += 1;
                    took += bite[slot][2] as f64;
                }
            }
            // (a linhagem semeada, grosso modo: o mesmo comprimento de genoma, ±3)
            if (a.gene_len as i64 - genome.len() as i64).abs() <= 3 {
                if has {
                    fa += 1;
                } else if body.iter().any(|&c| c != 0 && (c & 0x1F) - 1 == want_other as u16) && body.iter().filter(|&&c| c != 0).count() == 1 {
                    fb += 1;
                }
            }
        }
        let bites = counters.get(7).copied().unwrap_or(0);
        let d = last_bites.map_or("-".to_string(), |b: u32| bites.wrapping_sub(b).to_string());
        last_bites = Some(bites);
        println!("  {done:>6}  {n:>6}   {pr:>6} ({:.2})          {fa:>6} / {fb:<6}   a morder {:.1}% (tiram {:.2}/passo cada)   mortes {d}", e / pr.max(1) as f64, 100.0 * biting as f64 / pr.max(1) as f64, took / biting.max(1) as f64);
        if done >= steps {
            break;
        }
        let mut k = 0;
        while k < every {
            let s = (every - k).min(MAX_STEPS_PER_FRAME);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, s);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            k += s;
        }
        done += every;
    }
}
