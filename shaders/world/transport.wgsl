// TRANSPORTE DE MONÓMEROS. Esqueleto da fase 1: só a difusão quantizada do
// `transport_quanta` do v3 (simulation.wgsl). Advecção, assentamento,
// reações e terreno entram na fase 2, portados do mesmo kernel.

const DIFF_HOP_P: f32 = 0.01;
const DIFF_HOP_FLOOR: f32 = 0.15;
const MAX_HOPS_PER_CELL_CH: u32 = 8u;

@compute @workgroup_size(16, 16)
fn transport_quanta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    let idx = y * GRID_SIZE + x;

    // Sem fluido ainda: agitação 0, a difusão fica no chão (DIFF_HOP_FLOOR).
    let agitation = 0.0;
    let p_diff = clamp(DIFF_HOP_P * mix(DIFF_HOP_FLOOR, 1.0, agitation) * max(params.diffusion, 0.0), 0.0, 0.5);

    for (var ch = 0u; ch < 4u; ch++) {
        let slot = idx * 4u + ch;
        let v_ch = atomicLoad(&chem_grid[slot]);
        let act_n = v_ch & CHEM_STATE_MASK;
        let spent_n = v_ch >> 16u;
        let n = min(act_n + spent_n, MAX_HOPS_PER_CELL_CH);
        for (var k = 0u; k < n; k++) {
            let h = hash(slot ^ (k * 668265263u) ^ (params.seed * 2246822519u) ^ (params.epoch * 374761393u));
            if (f32(h) / 4294967295.0 >= p_diff) { continue; }
            // Estado do monómero que salta, escolhido em proporção da mistura.
            let pick_spent = hash_f32(h ^ 0x51ED270Bu) < f32(spent_n) / f32(max(act_n + spent_n, 1u));
            let d = hash(h ^ 0xC2B2AE35u) & 3u;
            var dx = 0;
            var dy = 0;
            if (d == 0u) { dx = 1; } else if (d == 1u) { dx = -1; }
            else if (d == 2u) { dy = 1; } else { dy = -1; }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) { continue; }
            let t_idx = u32(ny) * GRID_SIZE + u32(nx);
            if (chem_cell_total(t_idx) >= chem_capacity(t_idx)) { continue; }
            if (chem_take_state_one(slot, pick_spent)) {
                chem_add_state(t_idx, ch, 1u, pick_spent);
            }
        }
    }
}
