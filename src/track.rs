//! PISTA DE CORRIDAS: banco de ensaio da natação (ver
//! `shaders/common/track.wgsl`, que tem as mesmas fórmulas). Aqui só se gera
//! o terreno: rocha em todo o lado menos no corredor da pista.

use crate::params::WorldConfig;

const R0: f32 = 0.30;
const WAVE: f32 = 0.14;
const LOBES: f32 = 5.0;
const HALF: f32 = 0.022;
const HALF_WAVE: f32 = 0.3;
/// Grãos por célula de rocha (>= 3 é sólido).
const ROCK: u32 = 4;

/// Raio do eixo da pista no ângulo `theta` (unidades do mundo).
pub fn axis_r(sim: f32, theta: f32) -> f32 {
    sim * R0 * (1.0 + WAVE * (LOBES * theta).sin())
}

/// Meia-largura do corredor no ângulo `theta`.
pub fn half_width(sim: f32, theta: f32) -> f32 {
    sim * HALF * (1.0 + HALF_WAVE * (2.0 * theta + 1.0).sin())
}

/// Distância à parede mais próxima (positiva dentro do corredor).
pub fn wall_dist(sim: f32, x: f32, y: f32) -> f32 {
    let (dx, dy) = (x - 0.5 * sim, y - 0.5 * sim);
    let theta = dy.atan2(dx);
    half_width(sim, theta) - ((dx * dx + dy * dy).sqrt() - axis_r(sim, theta)).abs()
}

/// Terreno (grãos por célula) e calor (nenhum) da pista.
pub fn terrain(cfg: &WorldConfig) -> (Vec<u32>, Vec<f32>) {
    let n = cfg.grid_size as usize;
    let w = cfg.world_units_per_cell as f32;
    let sim = cfg.sim_size();
    let mut gamma = vec![ROCK; n * n];
    for y in 0..n {
        for x in 0..n {
            if wall_dist(sim, (x as f32 + 0.5) * w, (y as f32 + 0.5) * w) > 0.0 {
                gamma[y * n + x] = 0;
            }
        }
    }
    (gamma, vec![0.0; n * n])
}

/// Parâmetros do ensaio: só a energia do avanço conta (sem comer, sem luz
/// do sol, sem afundar) e os filhos são cópias iguais ao pai. Não há
/// monómeros: na pista os genomas nascem e copiam-se sem matéria.
pub fn preset(p: &mut crate::params::SimParams) {
    // Parte-se SEMPRE dos valores por omissão (menos os que o passo escreve
    // sozinho: epoch, semente, pincel...): o ensaio não herda afinações do
    // mundo que estava a correr, para os resultados serem comparáveis.
    const KEEP: [&str; 11] = ["epoch", "seed", "fluid_dt", "fluid_enabled", "max_agents", "fumarole_count", "spawn_count", "paint_x", "paint_y", "paint_radius", "paint_grains"];
    for (name, v) in crate::params::SimParams::default().to_named() {
        if !KEEP.contains(&name) && !name.starts_with('_') {
            p.set_named(name, v);
        }
    }
    p.track_mode = 1;
    p.copy_same = 1;
    p.uptake_rate = 0.0;
    p.skin_uptake = 0.0;
    p.photo_yield = 0.0;
    p.uv_damage = 1.0;
    p.sedimentation = 0.0;
    // Terreno parado (as paredes não se desfazem).
    p.sediment_transport = 0.0;
    p.bioturbation = 0.0;
    // Nasce-se com pouca energia: senão quem tem um órgão de armazenamento
    // guarda a energia inicial toda e sobrevive muito mais tempo sem avançar.
    p.spawn_energy = 3.0;
    // Com a manutenção normal um agente parado morre de fome em ~300 passos,
    // cedo demais para se ver quem avança: um quarto dela dá ~1200. Quem
    // avança bem não morre de fome; a vida média (5000 passos) é o que o tira
    // para dar lugar aos filhos.
    p.maintenance_cost = 0.0005;
    p.track_gain = 0.3;
    p.track_lifespan = 5000;
}

/// Um ponto do eixo da pista (para semear agentes): `t` em 0..1 dá a volta,
/// `off` em −1..1 desvia para as paredes (0 = no eixo).
pub fn point(sim: f32, t: f32, off: f32) -> [f32; 2] {
    let theta = t * std::f32::consts::TAU;
    let r = axis_r(sim, theta) + 0.7 * off * half_width(sim, theta);
    [0.5 * sim + r * theta.cos(), 0.5 * sim + r * theta.sin()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corridor_is_a_closed_ring() {
        let cfg = WorldConfig::DEFAULT;
        let (g, _) = terrain(&cfg);
        let open = g.iter().filter(|&&v| v == 0).count();
        // O corredor ocupa uma fração pequena mas não nula do mundo.
        assert!(open > g.len() / 50 && open < g.len() / 4, "células abertas: {open}");
        // O eixo está sempre dentro do corredor e do mundo.
        let sim = cfg.sim_size();
        for i in 0..360 {
            let p = point(sim, i as f32 / 360.0, 0.0);
            assert!(p[0] > 0.0 && p[1] > 0.0 && p[0] < sim && p[1] < sim);
            assert!(wall_dist(sim, p[0], p[1]) > 0.0);
        }
    }
}
