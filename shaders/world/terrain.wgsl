// DINÂMICA DO TERRENO (v3 simulation.wgsl: relax_gamma_pass, gamma_move_one,
// compute_gamma_slope). Gamma são quanta inteiros movidos por trocas
// atómicas: o terreno conserva-se tal como os monómeros.

// Altura de um quantum (unidades de "altura" do v3) e regras dos grãos.
const GAMMA_QUANTUM: f32 = 0.25;
// Uma célula com >= SANDPILE_DIFF quanta a mais que o vizinho mais baixo
// deixa escorregar um quantum com probabilidade GAMMA_RELAX_P (ângulo de
// repouso emergente: declives abaixo do limiar são estáveis para sempre).
const SANDPILE_DIFF: u32 = 3u;
const GAMMA_RELAX_P: f32 = 0.12;
// Grãos soltos isolados fazem passeio aleatório à procura de companhia.
const GAMMA_STRAY_WALK_P: f32 = 0.05;
// O entulho solto é levado pela corrente a esta fração da lei dos monómeros.
const GAMMA_SEDIMENT_FACTOR: f32 = 0.5;
// Uma pilha de 2 ao lado de uma célula vazia deixa cair o quantum de cima.
const GAMMA_SHED_P: f32 = 0.03;

fn gamma_take_one(idx: u32) -> bool {
    var taken = false;
    loop {
        let old = atomicLoad(&gamma_grid[idx]);
        if (old == 0u) { break; }
        let res = atomicCompareExchangeWeak(&gamma_grid[idx], old, old - 1u);
        if (res.exchanged) { taken = true; break; }
    }
    return taken;
}

// A ÚNICA forma de o terreno se mover: um quantum vai de `src` para `dst`,
// e os monómeros que deixam de caber em `dst` (a capacidade acabou de
// encolher) TROCAM para `src`, que ganhou espaço. Como um grão de areia a
// cair e a deslocar água: o terreno nunca enterra monómeros.
fn gamma_move_one(src: u32, dst: u32) -> bool {
    if (!gamma_take_one(src)) { return false; }
    atomicAdd(&gamma_grid[dst], 1u);
    let cap = chem_capacity(dst);
    var guard = 0u;
    loop {
        if (guard >= 64u) { break; }
        guard += 1u;
        if (chem_cell_total(dst) <= cap) { break; }
        var moved = false;
        for (var c = 0u; c < 4u; c++) {
            let slot = dst * 4u + c;
            if (chem_take_state_one(slot, true)) {
                chem_add_state(src, c, 1u, true);
                moved = true;
                break;
            }
            if (chem_take_state_one(slot, false)) {
                chem_add_state(src, c, 1u, false);
                moved = true;
                break;
            }
        }
        if (!moved) { break; }
    }
    return true;
}

fn gamma_height_at(ix: i32, iy: i32) -> f32 {
    let x = u32(clamp(ix, 0, i32(GRID_SIZE) - 1));
    let y = u32(clamp(iy, 0, i32(GRID_SIZE) - 1));
    return f32(gamma_count(y * GRID_SIZE + x)) * GAMMA_QUANTUM;
}

fn gamma_slope_at(idx: u32) -> vec2<f32> {
    return slope_grid[idx];
}

// DECLIVE do terreno (só terreno: os monómeros não contam). Gradiente de 8
// vizinhos com os diagonais pesados por 1/√2, por unidade do MUNDO.
@compute @workgroup_size(16, 16)
fn compute_gamma_slope(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= GRID_SIZE || gid.y >= GRID_SIZE) { return; }
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let l = gamma_height_at(ix - 1, iy);
    let r = gamma_height_at(ix + 1, iy);
    let b = gamma_height_at(ix, iy - 1);
    let t = gamma_height_at(ix, iy + 1);
    let bl = gamma_height_at(ix - 1, iy - 1);
    let br = gamma_height_at(ix + 1, iy - 1);
    let tl = gamma_height_at(ix - 1, iy + 1);
    let tr = gamma_height_at(ix + 1, iy + 1);
    let k = 1.0 / (4.0 * 1.41421356237);
    let dx = (r - l) * 0.5 + (tr + br - tl - bl) * k;
    let dy = (t - b) * 0.5 + (tl + tr - bl - br) * k;
    slope_grid[gid.y * GRID_SIZE + gid.x] = vec2<f32>(dx, dy) / f32(WORLD_UNITS_PER_CELL);
}

fn relax_gamma_pass(gid: vec3<u32>, phase: u32) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    let idx = y * GRID_SIZE + x;
    let n = gamma_count(idx);

    // GRÃOS SOLTOS (abaixo da banda da pilha de areia): uma só regra de
    // energia mínima. O grão vai para onde teria MAIS vizinhos ocupados
    // (aninha-se em cantos e reentrâncias e forma blocos compactos, nunca
    // torres, porque nunca sobe acima do seu nível). A mobilidade cai 4×
    // por cada ligação; a corrente ainda arranca grãos mal presos.
    if (n >= 1u && n < SANDPILE_DIFF) {
        var bonds = 0u;
        var best_idx = idx;
        var best_score = -1i;
        var best_c = 0xFFFFu;
        var em_idx = idx;
        var em_score = -1i;
        var em_found = false;
        var tie_h = hash(idx ^ (params.epoch * 1597334677u) ^ (params.seed * 3812015801u) ^ (phase * 2654435761u));
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                if (dx == 0i && dy == 0i) { continue; }
                let nx = i32(x) + dx;
                let ny = i32(y) + dy;
                if (nx < 0i || ny < 0i || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) { continue; }
                let ni = u32(ny) * GRID_SIZE + u32(nx);
                let c = gamma_count(ni);
                if (c >= 1u) { bonds += 1u; }
                if (c <= n) {
                    var score = 0i;
                    for (var sy = -1i; sy <= 1i; sy++) {
                        for (var sx = -1i; sx <= 1i; sx++) {
                            if (sx == 0i && sy == 0i) { continue; }
                            let qx = nx + sx;
                            let qy = ny + sy;
                            if (qx < 0i || qy < 0i || qx >= i32(GRID_SIZE) || qy >= i32(GRID_SIZE)) { continue; }
                            let qi = u32(qy) * GRID_SIZE + u32(qx);
                            if (qi == idx) { continue; }
                            if (gamma_count(qi) >= 1u) { score += 1i; }
                        }
                    }
                    tie_h = hash(tie_h ^ ni);
                    if (score > best_score
                        || (score == best_score && c < best_c)
                        || (score == best_score && c == best_c && (tie_h & 1u) == 1u)) {
                        best_score = score;
                        best_c = c;
                        best_idx = ni;
                    }
                    if (c == 0u && (score > em_score || (score == em_score && (tie_h & 2u) == 2u))) {
                        em_score = score;
                        em_idx = ni;
                        em_found = true;
                    }
                }
            }
        }
        if (bonds >= 4u) { return; } // preso
        let mob = pow(0.25, f32(bonds));

        var vs = vec2<f32>(0.0);
        if (params.fluid_enabled != 0u) {
            let fxi = min((x * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
            let fyi = min((y * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
            vs = sanitize_vec2(velocity_in[fgrid(fxi, fyi)]);
        }
        let p_sed = parcel_hop_p(vs) * GAMMA_SEDIMENT_FACTOR * mob;
        let hw = hash(idx ^ (params.epoch * 747796405u) ^ (phase * 196613u) ^ (params.seed * 2891336453u));
        let rw = f32(hw) / 4294967295.0;
        var dest = idx;
        if (n >= 2u && em_found && hash_f32(hw ^ 0x632BE5ABu) < GAMMA_SHED_P) {
            // Mini-falésia: o quantum de cima desce para a vaga mais aninhada.
            dest = em_idx;
        } else if (rw < p_sed) {
            // Salto a jusante (como o transporte dos monómeros), nunca a subir.
            var ddx = 0i;
            var ddy = 0i;
            let ax = abs(vs.x);
            let ay = abs(vs.y);
            if (hash_f32(hw ^ 0x85EBCA6Bu) < ax / max(ax + ay, 1e-6)) {
                ddx = select(-1i, 1i, vs.x > 0.0);
            } else {
                ddy = select(-1i, 1i, vs.y > 0.0);
            }
            let wx = i32(x) + ddx;
            let wy = i32(y) + ddy;
            if (wx >= 0i && wy >= 0i && wx < i32(GRID_SIZE) && wy < i32(GRID_SIZE)) {
                let wi = u32(wy) * GRID_SIZE + u32(wx);
                if (gamma_count(wi) <= n) { dest = wi; }
            }
        } else if (rw < p_sed + GAMMA_STRAY_WALK_P * mob) {
            if (best_score > i32(bonds)) {
                dest = best_idx;
            } else if (bonds == 0u) {
                // Grão livre num ótimo de nada: explora.
                var ddx = 0i;
                var ddy = 0i;
                switch ((hw >> 8u) % 8u) {
                    case 0u: { ddx = 1i; }
                    case 1u: { ddx = -1i; }
                    case 2u: { ddy = 1i; }
                    case 3u: { ddy = -1i; }
                    case 4u: { ddx = 1i; ddy = 1i; }
                    case 5u: { ddx = -1i; ddy = 1i; }
                    case 6u: { ddx = 1i; ddy = -1i; }
                    default: { ddx = -1i; ddy = -1i; }
                }
                let wx = i32(x) + ddx;
                let wy = i32(y) + ddy;
                if (wx >= 0i && wy >= 0i && wx < i32(GRID_SIZE) && wy < i32(GRID_SIZE)) {
                    let wi = u32(wy) * GRID_SIZE + u32(wx);
                    if (gamma_count(wi) <= n) { dest = wi; }
                }
            }
        }
        if (dest != idx) {
            gamma_move_one(idx, dest);
        }
        return;
    }
    if (n < SANDPILE_DIFF) { return; }

    // PILHA DE AREIA: escorrega para o vizinho (4) mais baixo.
    var best_idx = idx;
    var best_count = n;
    if (x > 0u) { let c = gamma_count(idx - 1u); if (c < best_count) { best_count = c; best_idx = idx - 1u; } }
    if (x + 1u < GRID_SIZE) { let c = gamma_count(idx + 1u); if (c < best_count) { best_count = c; best_idx = idx + 1u; } }
    if (y > 0u) { let c = gamma_count(idx - GRID_SIZE); if (c < best_count) { best_count = c; best_idx = idx - GRID_SIZE; } }
    if (y + 1u < GRID_SIZE) { let c = gamma_count(idx + GRID_SIZE); if (c < best_count) { best_count = c; best_idx = idx + GRID_SIZE; } }
    if (best_idx == idx || n - best_count < SANDPILE_DIFF) { return; }
    let h = hash(idx ^ (params.epoch * 668265263u) ^ (phase * 2654435761u) ^ (params.seed * 374761393u));
    if (f32(h) / 4294967295.0 < GAMMA_RELAX_P) {
        gamma_move_one(idx, best_idx);
    }
}

@compute @workgroup_size(16, 16)
fn relax_gamma_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax_gamma_pass(gid, 0u);
}

@compute @workgroup_size(16, 16)
fn relax_gamma_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax_gamma_pass(gid, 1u);
}
