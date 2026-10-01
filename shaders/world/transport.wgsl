// TRANSPORTE E REAÇÕES DOS MONÓMEROS — reescrito na fase 2.
//
// Cada passo tem duas fases:
//   1. `transport_scatter`: cada célula lê o estado ANTES do passo
//      (chem_grid), aplica as reações locais e envia cada um dos seus
//      monómeros para uma célula de destino, somando em chem_next.
//   2. `transport_commit`: chem_grid <- chem_next; chem_next <- 0.
// As somas inteiras não dependem da ordem das threads, por isso o passo é
// REPRODUTÍVEL bit a bit e não há interferência entre blocos da GPU (no v3
// a grelha era alterada no lugar e um monómero podia saltar duas vezes).
//
// Movimento de um monómero: parte de um ponto uniforme dentro da sua
// célula e desloca-se v·dt (sem limite de células por passo: os monómeros
// andam exatamente com a água, também na pluma rápida), mais um salto de
// difusão ou de assentamento. Cair num ponto aleatório da célula equivale a
// repartir a célula pelas células que o quadrado deslocado cobre: um campo
// uniforme num escoamento sem divergência fica uniforme. Nada cria nem
// destrói matéria: cada monómero acaba exatamente numa célula.

// ---- Transporte ----
// Difusão que depende da agitação: água calma só deixa um resíduo
// browniano (DIFF_HOP_FLOOR); água em movimento mistura à taxa toda.
const DIFF_HOP_P: f32 = 0.01;
const DIFF_HOP_FLOOR: f32 = 0.15;
const DIFF_AGITATION_SPEED: f32 = 0.5;   // células do fluido / s
// Monómeros dentro de gamma rastejam à taxa base, atenuada pela ocupação.
const BURIED_DIFF_FACTOR: f32 = 1.0;
const GAMMA_POROSITY_K: f32 = 0.3;
// Célula acima da capacidade expulsa o excesso depressa (em todo o lado).
const CHEM_SQUEEZE_P: f32 = 0.25;
// Assentamento por passo (× slider "settle").
const MONOMER_SETTLE_P: f32 = 0.002;
// Limite de monómeros sorteados por canal e célula (segurança; os
// restantes ficam no lugar). A capacidade normal é 48 no total.
const MAX_MOVERS_PER_CH: u32 = 256u;
// Entulho do terreno levado pela corrente: limite de probabilidade por passo.
const GRAIN_ADV_CAP: f32 = 0.9;

// ---- Reações ----
const LIGHT_ACT_P: f32 = 0.006;        // fotoativação por monómero gasto, luz plena
const CHEM_DECAY_P: f32 = 0.0002;      // hidrólise da ativação
const CHEM_SENSITIZE: f32 = 0.5;       // ativados na célula ajudam a ativar os gastos
const CHEM_SHIELD: f32 = 0.5;          // ativados juntos decaem menos
// Ativação térmica: água acima deste T reativa monómeros gastos.
const TEMP_ACT_THRESHOLD: f32 = 2.0;
const FUMAROLE_ACT_P: f32 = 0.15;

// Probabilidade de um grão de entulho saltar com a corrente (terreno).
// UNIDADES: v em células do fluido/s; saltos em células do ambiente.
fn parcel_hop_p(v: vec2<f32>) -> f32 {
    let env_per_fluid = f32(GRID_SIZE) / f32(FLUID_SIZE);
    let l1 = abs(v.x) + abs(v.y);
    return clamp(l1 * env_per_fluid * max(params.dt, 1e-3), 0.0, GRAIN_ADV_CAP);
}

@compute @workgroup_size(16, 16)
fn transport_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    let idx = y * GRID_SIZE + x;
    let src_total = chem_cell_total(idx);
    if (src_total == 0u) { return; }

    // Célula do ambiente -> posição no MUNDO (não passar coordenadas de
    // célula a funções de mundo: no v3 isso congelou todos os monómeros).
    let cell_w = f32(WORLD_UNITS_PER_CELL);
    let g_src = gamma_count(idx);
    var v = vec2<f32>(0.0);
    if (params.fluid_enabled != 0u && g_src == 0u) {
        // PONTO MÉDIO (Runge-Kutta 2): usa a velocidade a meio do caminho.
        // Com a velocidade só no início (Euler), cada passo segue a tangente
        // e atira os monómeros para fora dos remoinhos: os núcleos esvaziavam
        // (os pontos escuros na pluma).
        let c = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) * cell_w;
        let v0 = fluid_velocity_at_world(c);
        let world_per_fluid = SIM_SIZE / f32(FLUID_SIZE);
        v = fluid_velocity_at_world(c + v0 * world_per_fluid * max(params.dt, 0.0) * 0.5);
    }
    // Deslocamento por passo, em células do ambiente.
    let disp = v * (f32(GRID_SIZE) / f32(FLUID_SIZE)) * max(params.dt, 0.0);
    let agitation = clamp(length(v) / DIFF_AGITATION_SPEED, 0.0, 1.0);
    var p_diff = clamp(DIFF_HOP_P * mix(DIFF_HOP_FLOOR, 1.0, agitation) * max(params.diffusion, 0.0), 0.0, 0.5);
    if (g_src > 0u) {
        // Dentro do terreno não há correntes; só difusão lenta.
        p_diff = DIFF_HOP_P * BURIED_DIFF_FACTOR / (1.0 + GAMMA_POROSITY_K * f32(g_src));
    }
    if (src_total > chem_capacity(idx)) {
        p_diff = CHEM_SQUEEZE_P;
    }
    var p_settle = 0.0;
    if (g_src == 0u) {
        p_settle = MONOMER_SETTLE_P * max(params.settle, 0.0);
    }

    // Janela de destinos 4×4 à volta de floor(disp): o ponto de partida
    // (0..1) mais disp mais um salto de ±1 cai sempre dentro dela.
    let base = vec2<i32>(i32(floor(disp.x)) - 1, i32(floor(disp.y)) - 1);
    let light_t = uv_light_at_cell(x, y);

    for (var ch = 0u; ch < 4u; ch++) {
        let slot = idx * 4u + ch;
        let v_ch = atomicLoad(&chem_grid[slot]);
        var act_n = v_ch & CHEM_STATE_MASK;
        var spent_n = v_ch >> 16u;
        if (act_n + spent_n == 0u) { continue; }

        // ---- Reações locais (determinísticas: só dependem do estado antes) ----
        // FOTOATIVAÇÃO com sensibilização: os ativados são antenas; a
        // energia continua a vir só da luz.
        if (spent_n > 0u) {
            let sens = 1.0 + CHEM_SENSITIZE * f32(min(act_n, 8u));
            // + reativação uniforme (modo laboratório): cada gasto tem a mesma
            // probabilidade por passo, em qualquer lado.
            let exp_act = f32(min(spent_n, 8u)) * (LIGHT_ACT_P * max(params.uv_strength, 0.0) * light_t * sens
                + max(params.reactivation_rate, 0.0));
            var na = u32(floor(exp_act));
            if (rng_f4(slot, params.epoch, S_PHOTO).x < exp_act - f32(na)) { na += 1u; }
            na = min(na, min(spent_n, 8u));
            act_n += na;
            spent_n -= na;
        }
        // DECAIMENTO com blindagem.
        if (act_n > 0u) {
            let shield = 1.0 / (1.0 + CHEM_SHIELD * f32(min(act_n, 8u) - 1u));
            let exp_dec = f32(min(act_n, 8u)) * CHEM_DECAY_P * shield;
            if (rng_f4(slot, params.epoch, S_DECAY).x < exp_dec) {
                act_n -= 1u;
                spent_n += 1u;
            }
        }

        // COESÃO (por omissão desligada): os ativados difundem-se de
        // preferência para vizinhos ricos em ativados do mesmo tipo.
        var coh = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
        if (act_n > 0u && params.cohesion > 0.0) {
            if (x + 1u < GRID_SIZE) { coh[0] = 1.0 + params.cohesion * f32(min(chem_act_count(idx + 1u, ch), 8u)); }
            if (x > 0u) { coh[1] = 1.0 + params.cohesion * f32(min(chem_act_count(idx - 1u, ch), 8u)); }
            if (y + 1u < GRID_SIZE) { coh[2] = 1.0 + params.cohesion * f32(min(chem_act_count(idx + GRID_SIZE, ch), 8u)); }
            if (y > 0u) { coh[3] = 1.0 + params.cohesion * f32(min(chem_act_count(idx - GRID_SIZE, ch), 8u)); }
        }
        let coh_sum = coh[0] + coh[1] + coh[2] + coh[3];

        // ---- Movimento: histograma de destinos (ativados nos 16 bits baixos) ----
        var bins = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        var stay = 0u;
        let n = act_n + spent_n;
        for (var k = 0u; k < n; k++) {
            let unit = select(CHEM_SPENT_ONE, 1u, k < act_n);
            if (k >= MAX_MOVERS_PER_CH) {
                stay += unit;
                continue;
            }
            // 4 números: ponto de partida (x, y), evento, direção do salto.
            let rf = rng_f4(slot, params.epoch, S_MOVE + k);
            var p = rf.xy + disp;
            let r = rf.z;
            if (r < p_diff) {
                var d = min(u32(rf.w * 4.0), 3u);
                if (k < act_n && coh_sum > 4.0001) {
                    var u = rf.w * coh_sum;
                    d = 3u;
                    for (var cd = 0u; cd < 4u; cd++) {
                        if (u < coh[cd]) { d = cd; break; }
                        u -= coh[cd];
                    }
                }
                if (d == 0u) { p.x += 1.0; } else if (d == 1u) { p.x -= 1.0; }
                else if (d == 2u) { p.y += 1.0; } else { p.y -= 1.0; }
            } else if (r < p_diff + p_settle) {
                p.y -= 1.0;
            }
            let off = vec2<i32>(floor(p)) - base;
            let bi = u32(clamp(off.y, 0, 3)) * 4u + u32(clamp(off.x, 0, 3));
            bins[bi] += unit;
        }

        // ---- Emitir: uma soma atómica por destino ----
        for (var b = 0u; b < 16u; b++) {
            let cnt = bins[b];
            if (cnt == 0u) { continue; }
            // Paredes do aquário: o monómero fica na célula da borda.
            let tx = clamp(i32(x) + base.x + i32(b & 3u), 0, i32(GRID_SIZE) - 1);
            let ty = clamp(i32(y) + base.y + i32(b >> 2u), 0, i32(GRID_SIZE) - 1);
            let t_idx = u32(ty) * GRID_SIZE + u32(tx);
            // Rocha: a água não entra, o monómero também não. Dentro do
            // terreno só se passa por difusão lenta, com a porosidade.
            var blocked = false;
            if (t_idx != idx) {
                let g_tgt = gamma_count(t_idx);
                if (g_tgt > 0u) {
                    let gperm = 1.0 / (1.0 + GAMMA_POROSITY_K * f32(g_tgt));
                    blocked = g_src == 0u || rng_f4(slot, params.epoch, S_BLOCK + b).x >= gperm;
                }
            }
            if (blocked) {
                stay += cnt;
            } else {
                atomicAdd(&chem_next[t_idx * 4u + ch], cnt);
            }
        }
        if (stay > 0u) {
            atomicAdd(&chem_next[slot], stay);
        }
    }
}

// chem_grid <- chem_next; chem_next <- 0.
@compute @workgroup_size(256)
fn transport_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= GRID_SIZE * GRID_SIZE * 4u) { return; }
    atomicStore(&chem_grid[i], atomicLoad(&chem_next[i]));
    atomicStore(&chem_next[i], 0u);
}

// ATIVAÇÃO TÉRMICA (v3 `inject_fumarole_dye`, o nome era histórico): água
// acima de TEMP_ACT_THRESHOLD reativa monómeros gastos, no lugar.
@compute @workgroup_size(16, 16)
fn thermal_activation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    let fx = min((x * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
    let fy = min((y * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
    let t_loc = temp_in[fgrid(fx, fy)];
    if (t_loc <= TEMP_ACT_THRESHOLD) { return; }
    let p_act = clamp(FUMAROLE_ACT_P * (t_loc - TEMP_ACT_THRESHOLD) / TEMP_ACT_THRESHOLD, 0.0, 0.9);
    let idx = y * GRID_SIZE + x;
    for (var ch = 0u; ch < 4u; ch++) {
        let slot = idx * 4u + ch;
        if ((atomicLoad(&chem_grid[slot]) >> 16u) > 0u) {
            if (rng_f4(slot, params.epoch, S_THERMAL).x < p_act) {
                chem_activate_one(slot);
            }
        }
    }
}
