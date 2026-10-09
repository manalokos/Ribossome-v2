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
// O entulho solto é levado pela corrente a esta fração da lei dos monómeros
// (× params.sediment_transport), só acima da velocidade crítica de arranque.
const GAMMA_SEDIMENT_FACTOR: f32 = 0.5;
// Velocidade de queda de um grão na água (Stokes), em células do fluido por
// segundo (× params.sediment_settle): a mesma unidade das correntes.
const GAMMA_SETTLE_SPEED: f32 = 0.5;
// (A coesão no arranque é params.sediment_cohesion: cada vizinho soma essa
// fração à velocidade crítica.)

// Velocidade do fluido numa célula da grelha.
fn fluid_vel_at_cell(cx: u32, cy: u32) -> vec2<f32> {
    let fxi = min((cx * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
    let fyi = min((cy * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
    return sanitize_vec2(velocity_in[fgrid(fxi, fyi)]);
}
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

// Para onde despejar um monómero que deixou de caber em `dst`, por ordem:
// 1) a célula de origem do grão (`src`, que ganhou espaço) se tiver espaço;
// 2) uma vizinha de `dst` com espaço;
// 3) uma vizinha (ou `src`) que NÃO seja rocha, mesmo cheia: o excesso sai
//    depois pelo squeeze do transporte;
// 4) `src` (a infiltração em transport.wgsl deixa-o sair da rocha depois).
fn evict_target(src: u32, dst: u32) -> u32 {
    if (chem_cell_total(src) < chem_capacity(src)) { return src; }
    let x = i32(dst % GRID_SIZE);
    let y = i32(dst / GRID_SIZE);
    var nb = array<vec2<i32>, 4>(vec2<i32>(x + 1, y), vec2<i32>(x - 1, y), vec2<i32>(x, y + 1), vec2<i32>(x, y - 1));
    var not_rock = select(0xFFFFFFFFu, src, gamma_count(src) < GAMMA_SOLID_THRESHOLD);
    for (var i = 0u; i < 4u; i++) {
        let c = nb[i];
        if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(i32(GRID_SIZE)))) { continue; }
        let n = u32(c.y) * GRID_SIZE + u32(c.x);
        if (chem_cell_total(n) < chem_capacity(n)) { return n; }
        if (not_rock == 0xFFFFFFFFu && gamma_count(n) < GAMMA_SOLID_THRESHOLD) { not_rock = n; }
    }
    return select(not_rock, src, not_rock == 0xFFFFFFFFu);
}

// A ÚNICA forma de o terreno se mover: um quantum vai de `src` para `dst`,
// e os monómeros que deixam de caber em `dst` (a capacidade acabou de
// encolher) saem para uma célula com espaço (evict_target). Como um grão de
// areia a cair e a deslocar água: o terreno nunca enterra monómeros.
fn gamma_move_one(src: u32, dst: u32) -> bool {
    if (!gamma_take_one(src)) { return false; }
    atomicAdd(&gamma_grid[dst], 1u);
    let cap = chem_capacity(dst);
    var guard = 0u;
    loop {
        if (guard >= 64u) { break; }
        guard += 1u;
        if (chem_cell_total(dst) <= cap) { break; }
        let to = evict_target(src, dst);
        var moved = false;
        for (var c = 0u; c < 4u; c++) {
            let slot = dst * 4u + c;
            if (chem_take_state_one(slot, true)) {
                chem_add_state(to, c, 1u, true);
                moved = true;
                break;
            }
            if (chem_take_state_one(slot, false)) {
                chem_add_state(to, c, 1u, false);
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
        let rr = rng_u4(idx, params.epoch, S_RELAX + phase);
        var tie_h = rr.x;
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
        let mob = pow(0.25, f32(bonds));

        // A corrente que o grão SENTE é a da água ao lado (a mais forte dos
        // 4 vizinhos sem grãos): dentro do entulho o fluido quase não anda.
        // Um grão enterrado (sem água à volta) não sente corrente nenhuma.
        var vs = vec2<f32>(0.0);
        if (params.fluid_enabled != 0u) {
            var nb4 = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
            for (var k = 0u; k < 4u; k++) {
                let qx = i32(x) + nb4[k].x;
                let qy = i32(y) + nb4[k].y;
                if (qx < 0i || qy < 0i || qx >= i32(GRID_SIZE) || qy >= i32(GRID_SIZE)) { continue; }
                if (gamma_count(u32(qy) * GRID_SIZE + u32(qx)) != 0u) { continue; }
                let v = fluid_vel_at_cell(u32(qx), u32(qy));
                if (dot(v, v) > dot(vs, vs)) { vs = v; }
            }
        }
        // A GRAVIDADE não está aqui: é a passagem vertical (grain_fall), que
        // faz descer inteiras as pilhas sem apoio. Aqui só a corrente (e a
        // coesão, que é horizontal: resiste a ser arrancado, não à queda).
        let unsupported = y > 0u && gamma_count(idx - GRID_SIZE) == 0u;
        var v_move = vec2<f32>(0.0);
        var p_sed = 0.0;
        if (unsupported && bonds == 0u) {
            // EM SUSPENSÃO (grão solto na água): anda com a corrente. A
            // queda (grain_fall) puxa-o para baixo ao mesmo tempo: sobe onde
            // a corrente a subir vence a queda, assenta onde ela abranda.
            v_move = vs;
            p_sed = parcel_hop_p(v_move);
        } else {
            // NO FUNDO, ARRANQUE (Shields): só o excesso de velocidade acima
            // da crítica arrasta; a coesão sobe a crítica.
            let speed = length(vs);
            let crit = max(params.sediment_threshold, 0.0) * (1.0 + max(params.sediment_cohesion, 0.0) * f32(bonds));
            let excess = max(speed - crit, 0.0);
            v_move = select(vec2<f32>(0.0), vs * (excess / max(speed, 1e-6)), speed > 1e-6);
            p_sed = parcel_hop_p(v_move) * GAMMA_SEDIMENT_FACTOR * max(params.sediment_transport, 0.0);
        }
        // Preso (muitos vizinhos) e sem corrente que o arranque: fica.
        if (bonds >= 4u && p_sed <= 0.0) { return; }
        let hw = rr.y;
        let rw = f32(rr.z >> 8u) * (1.0 / 16777216.0);
        let rw2 = f32(rr.w >> 8u) * (1.0 / 16777216.0);
        var dest = idx;
        if (n >= 2u && em_found && rw2 < GAMMA_SHED_P) {
            // Mini-falésia: o quantum de cima desce para a vaga mais aninhada.
            dest = em_idx;
        } else if (rw < p_sed) {
            // Salto na direção do movimento (corrente, ou corrente − queda),
            // nunca para cima de uma célula com mais grãos.
            var ddx = 0i;
            var ddy = 0i;
            let ax = abs(v_move.x);
            let ay = abs(v_move.y);
            if (f32(hw >> 8u) * (1.0 / 16777216.0) < ax / max(ax + ay, 1e-6)) {
                ddx = select(-1i, 1i, v_move.x > 0.0);
            } else {
                ddy = select(-1i, 1i, v_move.y > 0.0);
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
    if (rng_f4(idx, params.epoch, S_SAND + phase).x < GAMMA_RELAX_P) {
        gamma_move_one(idx, best_idx);
    }
}

// PINCEL: põe params.paint_grains grãos em cada célula do disco (centro e
// raio em células). Ao acrescentar grãos, os monómeros que deixam de caber
// saem para uma vizinha com espaço (a mesma regra de quando um grão cai:
// o terreno nunca enterra monómeros; os que ficarem em rocha saem depois
// pela infiltração do transporte). Tirar grãos só abre espaço.
@compute @workgroup_size(16, 16)
fn paint_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= GRID_SIZE || gid.y >= GRID_SIZE) { return; }
    let d = vec2<f32>(f32(gid.x) + 0.5 - params.paint_x, f32(gid.y) + 0.5 - params.paint_y);
    if (dot(d, d) > params.paint_radius * params.paint_radius) { return; }
    let idx = gid.y * GRID_SIZE + gid.x;
    let want = u32(clamp(params.paint_grains, 0.0, 6.0));
    atomicStore(&gamma_grid[idx], want);
    let cap = chem_capacity(idx);
    var guard = 0u;
    loop {
        if (guard >= 64u) { break; }
        guard += 1u;
        if (chem_cell_total(idx) <= cap) { break; }
        let to = evict_target(idx, idx);
        if (to == idx) { break; }
        var moved = false;
        for (var c = 0u; c < 4u; c++) {
            let slot = idx * 4u + c;
            if (chem_take_state_one(slot, true)) {
                chem_add_state(to, c, 1u, true);
                moved = true;
                break;
            }
            if (chem_take_state_one(slot, false)) {
                chem_add_state(to, c, 1u, false);
                moved = true;
                break;
            }
        }
        if (!moved) { break; }
    }
}

// GRAVIDADE DOS GRÃOS (vertical, separada da coesão): uma thread por
// coluna percorre-a de baixo para cima. Um grão solto (abaixo do limiar da
// rocha) está APOIADO se tiver por baixo, sem buracos, grãos apoiados, rocha
// ou o fundo. Os que não estão descem UMA célula juntos, a pilha inteira
// (cada célula desce para a que a de baixo acabou de deixar), por isso um
// grumo cai inteiro em vez de se desfiar. Todas as colunas usam o mesmo
// relógio (descem no mesmo passo), à velocidade de queda
// GAMMA_SETTLE_SPEED × sediment_settle: um grumo largo cai junto.
@compute @workgroup_size(64)
fn grain_fall(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= GRID_SIZE) { return; }
    let w_s = GAMMA_SETTLE_SPEED * max(params.sediment_settle, 0.0);
    let rate = parcel_hop_p(vec2<f32>(0.0, w_s));
    if (rate <= 0.0) { return; }
    // Relógio comum (ponto fixo 16.16): desce quando a parte inteira de
    // epoch·rate avança.
    let rf = u32(rate * 65536.0);
    if (((params.epoch * rf) >> 16u) == (((params.epoch + 1u) * rf) >> 16u)) { return; }
    var supported = true; // o fundo do mundo apoia
    for (var y = 0u; y < GRID_SIZE; y++) {
        let idx = y * GRID_SIZE + x;
        let n = gamma_count(idx);
        if (n == 0u) {
            supported = false;
            continue;
        }
        if (n >= GAMMA_SOLID_THRESHOLD) {
            supported = true; // rocha: fixa, apoia o que está em cima
            continue;
        }
        if (supported) { continue; }
        // Sem apoio: desce para a célula de baixo (vazia: ou já era, ou a
        // pilha de baixo acabou de descer).
        for (var k = 0u; k < n; k++) { gamma_move_one(idx, idx - GRID_SIZE); }
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
