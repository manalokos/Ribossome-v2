//! LINHAGENS REGISTADAS: a árvore da vida como aconteceu. De tempos a tempos
//! (um "censo") agrupam-se os genomas vivos em espécies e liga-se cada uma à
//! do censo anterior de que descende. É uma ferramenta do observador (as
//! criaturas não conhecem genomas): nada disto entra nas regras do mundo.
//!
//! Regra de ligação, por ordem de abundância:
//!   - a espécie continua o ramo vivo mais parecido, se estiver a menos de
//!     `SAME` do genoma atual desse ramo (o ramo passa a ter este genoma);
//!   - senão abre um ramo novo, filho do ramo (vivo ou extinto) mais
//!     parecido, se estiver a menos de `BRANCH`;
//!   - senão é uma raiz (não se sabe de onde veio; no primeiro censo são
//!     todas raízes).
//! Só entram espécies com pelo menos `MIN_SHARE` dos agentes.

use serde::{Deserialize, Serialize};

use crate::species::{Species, distance};

/// Distância máxima para ser o mesmo ramo (o limiar das espécies).
const SAME: f32 = 0.15;
/// Distância máxima para se reconhecer o pai de um ramo novo (dois genomas
/// ao acaso ficam a ~0,5: acima de 0,35 já não se distingue parentesco).
const BRANCH: f32 = 0.35;
/// Fração mínima dos agentes para uma espécie entrar no censo.
const MIN_SHARE: f32 = 0.005;
/// Espécies por censo, no máximo.
const MAX_PER_CENSUS: usize = 40;
/// Um ramo que falhe mais censos seguidos do que isto está extinto.
const GRACE: u32 = 3;

#[derive(Clone, Serialize, Deserialize)]
pub struct Branch {
    pub id: u32,
    pub parent: Option<u32>,
    /// Epoch do primeiro e do último censo em que apareceu.
    pub born: u32,
    pub last: u32,
    /// Índice do último censo em que apareceu.
    pub seen: u32,
    /// Genoma do líder no primeiro censo e no último.
    pub first: Vec<u8>,
    pub leader: Vec<u8>,
    /// (epoch, agentes) em cada censo em que apareceu.
    pub counts: Vec<(u32, u32)>,
    pub peak: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Lineages {
    /// Epochs entre censos.
    pub every: u32,
    pub next_epoch: u32,
    /// Censos feitos e epoch do primeiro e do último.
    pub censuses: u32,
    pub first_epoch: u32,
    pub last_epoch: u32,
    pub branches: Vec<Branch>,
}

impl Default for Lineages {
    fn default() -> Self {
        Self { every: 50_000, next_epoch: 0, censuses: 0, first_epoch: 0, last_epoch: 0, branches: Vec::new() }
    }
}

impl Lineages {
    /// Junta um censo: as espécies vivas (por ordem de abundância) e o total
    /// de agentes.
    pub fn census(&mut self, epoch: u32, species: &[Species], total: usize) {
        let idx = self.censuses;
        if idx == 0 {
            self.first_epoch = epoch;
        }
        let min = ((total as f32 * MIN_SHARE) as u32).max(5);
        let mut taken = vec![false; self.branches.len()];
        for s in species.iter().filter(|s| s.count >= min).take(MAX_PER_CENSUS) {
            // O ramo vivo e livre mais parecido.
            let mut best: Option<(usize, f32)> = None;
            // O ramo mais parecido de todos (para ser o pai).
            let mut near: Option<(usize, f32)> = None;
            for (i, b) in self.branches.iter().enumerate() {
                let d = distance(&s.leader, &b.leader);
                if near.is_none_or(|(_, m)| d < m) {
                    near = Some((i, d));
                }
                let alive = idx > 0 && b.seen + GRACE + 1 >= idx;
                if alive && i < taken.len() && !taken[i] && best.is_none_or(|(_, m)| d < m) {
                    best = Some((i, d));
                }
            }
            match best.filter(|(_, d)| *d <= SAME) {
                Some((i, _)) => {
                    taken[i] = true;
                    let b = &mut self.branches[i];
                    b.leader = s.leader.clone();
                    b.last = epoch;
                    b.seen = idx;
                    b.counts.push((epoch, s.count));
                    b.peak = b.peak.max(s.count);
                }
                None => {
                    let parent = near.filter(|(_, d)| *d <= BRANCH).map(|(i, _)| self.branches[i].id);
                    self.branches.push(Branch {
                        id: self.branches.len() as u32,
                        parent,
                        born: epoch,
                        last: epoch,
                        seen: idx,
                        first: s.leader.clone(),
                        leader: s.leader.clone(),
                        counts: vec![(epoch, s.count)],
                        peak: s.count,
                    });
                }
            }
        }
        self.censuses += 1;
        self.last_epoch = epoch;
        self.next_epoch = epoch.saturating_add(self.every.max(1000));
    }

    /// O ramo ainda está vivo (apareceu num dos últimos censos).
    pub fn alive(&self, b: &Branch) -> bool {
        b.seen + GRACE + 1 >= self.censuses
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        serde_json::from_slice(b).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(leader: Vec<u8>, count: u32) -> Species {
        Species { leader, count, distinct: 1, same_strand: count }
    }

    #[test]
    fn branches_continue_split_and_die() {
        let a: Vec<u8> = (0..100).map(|i| ((i * 7 + i / 3) % 4) as u8).collect();
        // Um parente afastado (30% diferente): ramo novo, filho do primeiro.
        let mut b = a.clone();
        for i in 0..30 {
            b[i * 3] ^= 2;
        }
        let other: Vec<u8> = (0..100).map(|i| ((i * 5 + i / 7 + 1) % 4) as u8).collect();
        let mut l = Lineages { every: 1000, ..Default::default() };
        l.census(0, &[sp(a.clone(), 100)], 100);
        l.census(1000, &[sp(a.clone(), 80), sp(b.clone(), 20)], 100);
        l.census(2000, &[sp(b.clone(), 60), sp(other.clone(), 40)], 100);
        assert_eq!(l.branches.len(), 3);
        assert_eq!(l.branches[0].counts.len(), 2);
        assert_eq!(l.branches[1].parent, Some(0));
        assert_eq!(l.branches[2].parent, None, "sem parentesco reconhecível: raiz");
        for e in 3..8 {
            l.census(e * 1000, &[sp(b.clone(), 100)], 100);
        }
        assert!(!l.alive(&l.branches[0]) && l.alive(&l.branches[1]));
        let back = Lineages::from_bytes(&l.to_bytes()).unwrap();
        assert_eq!(back.branches.len(), 3);
    }
}
