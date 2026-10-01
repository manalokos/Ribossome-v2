//! Vida: genoma, aminoácidos, organismos.

pub mod amino;

use crate::params::SpawnRequest;

/// splitmix64: RNG determinista do lado do CPU (só escolhe posições e
/// comprimentos dos pedidos; a montagem do genoma é feita na GPU).
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniforme em [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Pedidos de sementes em pontos aleatórios do mundo, com comprimento
/// uniforme em `len` (inclusivo). A GPU monta cada genoma com os monómeros
/// ativados mais próximos; um ponto sem bases suficientes simplesmente falha.
pub fn seed_requests(n: u32, len: [u32; 2], aug: bool, sim_size: f32, rng: &mut SplitMix) -> Vec<SpawnRequest> {
    let (lo, hi) = (len[0].min(len[1]).max(3), len[0].max(len[1]).min(256));
    (0..n)
        .map(|_| SpawnRequest {
            pos_x: rng.f32() * sim_size,
            pos_y: rng.f32() * sim_size,
            gene_len: lo + (rng.next_u64() % (hi - lo + 1) as u64) as u32,
            flags: aug as u32,
        })
        .collect()
}
