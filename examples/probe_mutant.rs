//! Ter uma protease compensa a um nativo deste mundo? Carrega uma cena,
//! pega no genoma mais comum (uma espécie que já vinga), e constrói dois
//! mutantes com 6 bases a mais no mesmo sítio:
//!   - "protease": a forma A ganha UMA protease (variante VARIANT, por omissão
//!     0 = contacto, sempre ligada) e tudo o resto fica igual nas duas formas;
//!   - "neutro": as mesmas 6 bases a mais mas sem órgão novo (o controlo: o
//!     que custa só ter o genoma e o corpo mais compridos).
//! Semeia N de cada (em corridas separadas) junto de agentes vivos e segue-os
//! pelo comprimento do genoma. Se o da protease se aguenta melhor do que o
//! neutro, a protease paga-se; se cai mais depressa, é um peso.
//! MUTANT=protease|neutro, PARAMS=nome=valor,...
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::life::organs::{ORGAN_SYMBOLS, translate_organs};
use ribossome::life::table;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::species::reverse_complement;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let want_protease = std::env::var("MUTANT").map_or(true, |m| m != "neutro");
    let variant = envf("VARIANT", 0.0) as u8;
    let count = envf("N", 1500.0) as usize;
    let steps = envf("STEPS", 20000.0) as u32;
    let every = envf("EVERY", 4000.0) as u32;
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
    let organs_of = |g: &[u8]| -> Vec<(u8, u8)> { translate_organs(g, rs, &code).iter().filter_map(|r| r.organ.map(|(t, v, _)| (t, v))).collect() };
    let agents = w.read_agents_blocking(&gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    let genome_of = |slot: usize, n: usize| -> Vec<u8> { (0..n).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect() };
    let alive: Vec<usize> = (0..agents.len()).filter(|&s| agents[s].alive != 0).collect();
    // O genoma mais comum (contando as duas cadeias juntas) sem protease.
    let mut tally: HashMap<Vec<u8>, u32> = HashMap::new();
    for &s in &alive {
        let g = genome_of(s, agents[s].gene_len as usize);
        let rc = reverse_complement(&g);
        *tally.entry(if g <= rc { g } else { rc }).or_default() += 1;
    }
    let mut common: Vec<(Vec<u8>, u32)> = tally.into_iter().collect();
    common.sort_by(|a, b| b.1.cmp(&a.1));
    let letters = |g: &[u8]| g.iter().map(|&b| "AUGC".as_bytes()[b as usize] as char).collect::<String>();
    let show = |g: &[u8]| {
        let body = translate_organs(g, rs, &code);
        let o: String = body.iter().filter_map(|r| r.organ.map(|(t, _, _)| ORGAN_SYMBOLS[t as usize])).collect();
        format!("{} resíduos [{o}]", body.len())
    };
    // Para o MESMO nativo: um mutante com protease e um neutro (as duas
    // corridas comparam-se uma com a outra).
    let find = |strand: &[u8], protease: bool| -> Option<Vec<u8>> {
        let (oa, ob) = (organs_of(strand), organs_of(&reverse_complement(strand)));
        let (la, lb) = (translate_organs(strand, rs, &code).len(), translate_organs(&reverse_complement(strand), rs, &code).len());
        let start = strand.windows(3).position(|x| x == [0, 1, 2])?;
        for at in (start + 3..strand.len().saturating_sub(3)).step_by(3) {
            for ins in 0..4096u32 {
                let six: Vec<u8> = (0..6).map(|k| ((ins >> (2 * k)) & 3) as u8).collect();
                let mut m = strand[..at].to_vec();
                m.extend(&six);
                m.extend(&strand[at..]);
                let (ma, mb) = (organs_of(&m), organs_of(&reverse_complement(&m)));
                let (ba, bb) = (translate_organs(&m, rs, &code).len(), translate_organs(&reverse_complement(&m), rs, &code).len());
                if mb != ob || bb < lb || bb > lb + 2 {
                    continue;
                }
                let ok = if protease {
                    let mut rest = ma.clone();
                    match rest.iter().position(|o| *o == (11, variant)) {
                        Some(i) => {
                            rest.remove(i);
                            rest == oa && ba >= la && ba <= la + 2
                        }
                        None => false,
                    }
                } else {
                    ma == oa && ba == la + 2
                };
                if ok {
                    return Some(m);
                }
            }
        }
        None
    };
    let mut chosen: Option<(Vec<u8>, Vec<u8>, u32)> = None;
    'outer: for (native, n) in common.iter().take(40) {
        for strand in [native.clone(), reverse_complement(native)] {
            let (oa, ob) = (organs_of(&strand), organs_of(&reverse_complement(&strand)));
            if oa.is_empty() || ob.is_empty() || oa.iter().chain(&ob).any(|o| o.0 == 11) || strand.len() > 120 {
                continue;
            }
            if let (Some(mp), Some(mn)) = (find(&strand, true), find(&strand, false)) {
                chosen = Some((strand.clone(), if want_protease { mp } else { mn }, *n));
                break 'outer;
            }
        }
    }
    let (native, mutant, n_native) = chosen.expect("não encontrei um mutante assim nos 40 genomas mais comuns");
    println!("nativo ({} agentes): {} ({} bases): forma A {}; forma B {}", n_native, letters(&native), native.len(), show(&native), show(&reverse_complement(&native)));
    println!("mutante {}: {} ({} bases): forma A {}; forma B {}", if want_protease { "COM PROTEASE" } else { "NEUTRO" }, letters(&mutant), mutant.len(), show(&mutant), show(&reverse_complement(&mutant)));
    let mut rng = ribossome::life::SplitMix(envf("SEED", 7.0) as u64);
    let mut reqs = Vec::new();
    for _ in 0..count {
        let a = &agents[alive[(rng.next_u64() % alive.len() as u64) as usize]];
        reqs.push(SpawnRequest::with_genome(a.pos_x + 60.0 * (rng.f32() - 0.5), a.pos_y + 60.0 * (rng.f32() - 0.5), &mutant));
    }
    w.request_seeds(&reqs);
    println!("epoch {}: {} vivos; semeados {count}", w.params.epoch, alive.len());
    println!("  passos   vivos   mutantes (o genoma semeado e variantes próximas, {} bases): com protease / sem   energia deles | energia de todos   mortes por protease", mutant.len());
    let mutant_rc = reverse_complement(&mutant);
    let mut done = 0;
    let mut last_bites = None;
    while done <= steps {
        let ag = w.read_agents_blocking(&gpu);
        let org: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
        let counters: Vec<u32> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.life_counters_buf));
        let now: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
        let genome_now = |slot: usize, n: usize| -> Vec<u8> { (0..n).map(|i| ((now[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect() };
        let (mut n, mut with, mut without, mut e, mut e_all) = (0u32, 0u32, 0u32, 0f64, 0f64);
        for (slot, a) in ag.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            n += 1;
            e_all += a.energy as f64;
            let same = a.gene_len as usize == mutant.len() && {
                let g = genome_now(slot, mutant.len());
                let d = |x: &[u8]| g.iter().zip(x).filter(|(p, q)| p != q).count();
                d(&mutant) <= 3 || d(&mutant_rc) <= 3
            };
            if same {
                let has = org[slot * 64..slot * 64 + (a.body_len as usize).min(64)].iter().any(|&c| c != 0 && (c & 0x1F) - 1 == 11);
                if has { with += 1 } else { without += 1 }
                e += a.energy as f64;
            }
        }
        let bites = counters.get(7).copied().unwrap_or(0);
        let d = last_bites.map_or("-".to_string(), |b: u32| bites.wrapping_sub(b).to_string());
        last_bites = Some(bites);
        println!("  {done:>6}  {n:>6}   {with:>6} / {without:<6}   {:.2} | {:.2}   {d}", e / (with + without).max(1) as f64, e_all / n.max(1) as f64);
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
