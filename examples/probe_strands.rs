//! As duas cadeias: cada filho é o complemento reverso do pai, por isso uma
//! linhagem alterna entre dois fenótipos (fita + e fita −). Carrega uma cena
//! (por omissão o autosave), traduz o genoma de cada agente e o complemento
//! reverso dele (= o fenótipo dos filhos, sem contar mutações) e mostra:
//! - para cada fenótipo (foto, quimio, boca, armazenamento), o que têm os
//!   filhos;
//! - pares pai-filho vivos (pelo id do pai);
//! - quantos agentes têm o complemento do seu genoma presente na população.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::life::organs::translate_organs;
use ribossome::life::table::code_to_gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

const GENOME_WORDS: usize = 16;
// Tipos de órgão (ver assets/orgaos.json).
const MOUTH: u8 = 0;
const STORAGE: u8 = 7;
const PHOTO: u8 = 10;
const CHEMO: u8 = 14;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Ph {
    photo: bool,
    chemo: bool,
    mouth: bool,
    storage: bool,
}

impl Ph {
    fn label(&self) -> String {
        let mut s = String::new();
        for (on, c) in [(self.photo, 'F'), (self.chemo, 'Q'), (self.mouth, 'B'), (self.storage, 'A')] {
            s.push(if on { c } else { '.' });
        }
        s
    }
}

fn phenotype(genome: &[u8], require_start: bool, code: &[u32]) -> Ph {
    let mut p = Ph::default();
    for r in translate_organs(genome, require_start, code) {
        if let Some((t, _, _)) = r.organ {
            match t {
                PHOTO => p.photo = true,
                CHEMO => p.chemo = true,
                MOUTH => p.mouth = true,
                STORAGE => p.storage = true,
                _ => {}
            }
        }
    }
    p
}

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    let agents = w.read_agents_blocking(&gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    let code = code_to_gpu(&w.organ_code);
    let rs = w.params.require_start != 0;

    struct A {
        id: u32,
        parent: u32,
        energy: f32,
        y: f32,
        me: Ph,
        kids: Ph,
        genome: Vec<u8>,
    }
    let mut list = Vec::new();
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let n = a.gene_len as usize;
        let g: Vec<u8> = (0..n).map(|i| ((words[slot * GENOME_WORDS + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
        let rc: Vec<u8> = g.iter().rev().map(|b| b ^ 1).collect();
        list.push(A {
            id: a.id,
            parent: a.parent,
            energy: a.energy,
            y: a.pos_y / w.cfg.sim_size(),
            me: phenotype(&g, rs, &code),
            kids: phenotype(&rc, rs, &code),
            genome: g,
        });
    }
    println!("epoch {}, {} agentes vivos (F foto, Q quimio, B boca, A armazenamento)\n", w.params.epoch, list.len());

    // 1. Fenótipo -> fenótipo dos filhos.
    let mut pairs: HashMap<(Ph, Ph), (u32, f32, f32)> = HashMap::new();
    for a in &list {
        let e = pairs.entry((a.me, a.kids)).or_default();
        e.0 += 1;
        e.1 += a.energy;
        e.2 += a.y;
    }
    let mut v: Vec<_> = pairs.into_iter().collect();
    v.sort_by_key(|(_, (n, _, _))| std::cmp::Reverse(*n));
    println!("EU -> OS MEUS FILHOS (complemento reverso), os 20 mais comuns:");
    println!("  eu    filhos   agentes  energia  altura (0 fundo, 1 topo)");
    for ((me, kids), (n, e, y)) in v.iter().take(20) {
        println!("  {}  ->  {}   {:6}   {:6.1}   {:.2}", me.label(), kids.label(), n, e / *n as f32, y / *n as f32);
    }

    // 2. Os fotossintéticos: o que são os filhos?
    let photo: Vec<&A> = list.iter().filter(|a| a.me.photo).collect();
    let np = photo.len().max(1) as f32;
    let pct = |f: &dyn Fn(&A) -> bool| 100.0 * photo.iter().filter(|a| f(a)).count() as f32 / np;
    println!("\nCOM FOTOSSISTEMA ({}): os filhos têm", photo.len());
    println!("  fotossistema {:.0}%  armazenamento {:.0}%  boca {:.0}%  quimiossíntese {:.0}%", pct(&|a| a.kids.photo), pct(&|a| a.kids.storage), pct(&|a| a.kids.mouth), pct(&|a| a.kids.chemo));
    println!("  e eles próprios: armazenamento {:.0}%  boca {:.0}%  quimiossíntese {:.0}%", pct(&|a| a.me.storage), pct(&|a| a.me.mouth), pct(&|a| a.me.chemo));

    // 3. Pares pai-filho vivos.
    let by_id: HashMap<u32, &A> = list.iter().map(|a| (a.id, a)).collect();
    let mut pf: HashMap<(Ph, Ph), u32> = HashMap::new();
    for a in &list {
        if let Some(p) = by_id.get(&a.parent) {
            *pf.entry((p.me, a.me)).or_default() += 1;
        }
    }
    let total: u32 = pf.values().sum();
    let mut v: Vec<_> = pf.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("\nPARES PAI -> FILHO VIVOS ({total}), os 12 mais comuns:");
    for ((p, c), n) in v.iter().take(12) {
        println!("  {}  ->  {}   {n}", p.label(), c.label());
    }

    // 4. O complemento do meu genoma existe na população?
    let set: std::collections::HashSet<&[u8]> = list.iter().map(|a| a.genome.as_slice()).collect();
    let with_rc = list
        .iter()
        .filter(|a| {
            let rc: Vec<u8> = a.genome.iter().rev().map(|b| b ^ 1).collect();
            set.contains(rc.as_slice())
        })
        .count();
    println!("\n{with_rc} de {} agentes têm o seu complemento reverso exato vivo na população", list.len());
}
