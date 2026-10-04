//! "Espécies" vistas de fora: grupos de genomas parecidos. É uma ferramenta
//! do observador (as criaturas não conhecem genomas): nada disto entra nas
//! regras do mundo.
//!
//! Cada filho é o complemento reverso do pai, por isso uma linhagem tem duas
//! fitas; a distância entre dois genomas é a menor das distâncias de edição
//! entre um e o outro ou o complemento reverso do outro, a dividir pelo
//! comprimento do maior. O agrupamento é por "líder": os genomas distintos
//! entram por ordem de abundância e cada um junta-se ao primeiro líder a
//! menos de `threshold`; senão passa a ser líder de uma espécie nova.

use std::collections::HashMap;

pub struct Species {
    /// Genoma do líder (o mais abundante do grupo), bases 0..3.
    pub leader: Vec<u8>,
    /// Agentes no grupo e quantos genomas distintos.
    pub count: u32,
    pub distinct: u32,
    /// Agentes na fita do líder (os outros estão na complementar).
    pub same_strand: u32,
}

pub fn reverse_complement(g: &[u8]) -> Vec<u8> {
    g.iter().rev().map(|b| b ^ 1).collect()
}

/// Distância de edição com banda: None se passar de `max`.
fn edit_within(a: &[u8], b: &[u8], max: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    const BIG: usize = usize::MAX / 2;
    let mut prev = vec![BIG; b.len() + 1];
    let mut cur = vec![BIG; b.len() + 1];
    for (j, p) in prev.iter_mut().enumerate().take(max.min(b.len()) + 1) {
        *p = j;
    }
    for i in 1..=a.len() {
        let lo = i.saturating_sub(max).max(1);
        let hi = (i + max).min(b.len());
        cur[lo - 1] = if lo == 1 { i } else { BIG };
        let mut best = cur[lo - 1];
        for j in lo..=hi {
            let sub = prev[j - 1] + (a[i - 1] != b[j - 1]) as usize;
            let v = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
            cur[j] = v;
            best = best.min(v);
        }
        if hi < b.len() {
            cur[hi + 1] = BIG;
        }
        if best > max {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (prev[b.len()] <= max).then_some(prev[b.len()])
}

/// Agrupa os genomas em espécies (ver o topo do módulo). Devolve-as por
/// ordem decrescente de agentes.
pub fn cluster(genomes: &[Vec<u8>], threshold: f32) -> Vec<Species> {
    let mut counts: HashMap<&[u8], u32> = HashMap::new();
    for g in genomes {
        *counts.entry(g.as_slice()).or_default() += 1;
    }
    let mut distinct: Vec<(&[u8], u32)> = counts.into_iter().collect();
    distinct.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut out: Vec<Species> = Vec::new();
    for (g, n) in distinct {
        let rc = reverse_complement(g);
        let mut found = None;
        for (si, s) in out.iter().enumerate() {
            let max = (threshold * g.len().max(s.leader.len()) as f32) as usize;
            if edit_within(g, &s.leader, max).is_some() {
                found = Some((si, true));
                break;
            }
            if edit_within(&rc, &s.leader, max).is_some() {
                found = Some((si, false));
                break;
            }
        }
        match found {
            Some((si, same)) => {
                out[si].count += n;
                out[si].distinct += 1;
                out[si].same_strand += if same { n } else { 0 };
            }
            None => out.push(Species { leader: g.to_vec(), count: n, distinct: 1, same_strand: n }),
        }
    }
    out.sort_by(|a, b| b.count.cmp(&a.count));
    out
}

/// Genomas dos agentes vivos (bases 0..3), pela ordem dos slots.
pub fn living_genomes(gpu: &crate::gpu::Gpu, w: &crate::world::World) -> Vec<Vec<u8>> {
    const GENOME_WORDS: usize = 16;
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    w.read_agents_blocking(gpu)
        .iter()
        .enumerate()
        .filter(|(_, a)| a.alive != 0)
        .map(|(slot, a)| {
            (0..a.gene_len as usize).map(|i| ((words[slot * GENOME_WORDS + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect()
        })
        .collect()
}

/// Os órgãos de um genoma, como símbolos pela ordem do corpo (para as duas
/// fitas ver `reverse_complement`).
pub fn organ_symbols(genome: &[u8], require_start: bool, code: &[u32]) -> (usize, String) {
    let body = crate::life::organs::translate_organs(genome, require_start, code);
    let s = body.iter().filter_map(|r| r.organ.map(|(t, _, _)| crate::life::organs::ORGAN_SYMBOLS[t as usize])).collect();
    (body.len(), s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strands_and_mutants_are_one_species() {
        let a: Vec<u8> = (0..120).map(|i| ((i * 7 + i / 3) % 4) as u8).collect();
        let mut m = a.clone();
        m[10] ^= 2;
        m.remove(50);
        let other: Vec<u8> = (0..120).map(|i| ((i * 5 + i / 7 + 1) % 4) as u8).collect();
        let s = cluster(&[a.clone(), a.clone(), reverse_complement(&a), m, other], 0.15);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].count, s[0].distinct, s[0].same_strand), (4, 3, 3));
        assert_eq!(edit_within(&a, &a, 0), Some(0));
    }
}
