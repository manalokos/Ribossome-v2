// TRANSPORTE E REAÇÕES DOS MONÓMEROS (v3 simulation.wgsl `transport_quanta`).
// Tudo são trocas inteiras e atómicas: saltos de um quantum entre células
// vizinhas, ou mudanças de estado no lugar. Nada cria nem destrói matéria.

// ---- Transporte ----
// Difusão que depende da agitação: água calma só deixa um resíduo
// browniano (DIFF_HOP_FLOOR); água em movimento mistura à taxa toda. É o
// mecanismo anti-entropia: zonas calmas ficam reservatórios concentrados.
const DIFF_HOP_P: f32 = 0.01;
const DIFF_HOP_FLOOR: f32 = 0.15;
const DIFF_AGITATION_SPEED: f32 = 0.5;   // células do fluido / s
const MONOMER_ADV_CAP: f32 = 0.9;
const MAX_HOPS_PER_CELL_CH: u32 = 8u;
// Monómeros dentro de gamma rastejam à taxa base, atenuada pela ocupação.
const BURIED_DIFF_FACTOR: f32 = 1.0;
const GAMMA_POROSITY_K: f32 = 0.3;
// Célula acima da capacidade expulsa o excesso depressa (em todo o lado:
// no v3 só dentro do terreno; decidido com o Filipe na fase 2).
const CHEM_SQUEEZE_P: f32 = 0.25;
// Assentamento por passo (× slider "settle").
const MONOMER_SETTLE_P: f32 = 0.002;

// ---- Reações ----
const LIGHT_ACT_P: f32 = 0.006;        // fotoativação por monómero gasto, luz plena
const CHEM_DECAY_P: f32 = 0.0002;      // hidrólise da ativação
const CHEM_SENSITIZE: f32 = 0.5;       // ativados na célula ajudam a ativar os gastos
const CHEM_SHIELD: f32 = 0.5;          // ativados juntos decaem menos
// Ativação térmica: água acima deste T reativa monómeros gastos.
const TEMP_ACT_THRESHOLD: f32 = 2.0;
const FUMAROLE_ACT_P: f32 = 0.15;

// PARCEL-EXACT: deslocamento esperado = |v|·dt, a mesma distância que uma
// parcela de água percorre. Norma L1 (|vx|+|vy|): a direção do salto é
// repartida por |vx|/(|vx|+|vy|), por isso o fluxo por eixo é exatamente
// k·vx·n / k·vy·n, um esquema upwind conservativo em média que mantém
// uniforme um campo sem divergência. (Com L2 nasciam bolsas de vácuo nas
// cabeças das plumas.) UNIDADES: v em células do fluido/s; saltos em
// células do ambiente.
fn parcel_hop_p(v: vec2<f32>) -> f32 {
    let env_per_fluid = f32(GRID_SIZE) / f32(FLUID_SIZE);
    let l1 = abs(v.x) + abs(v.y);
    return clamp(l1 * env_per_fluid * max(params.dt, 1e-3), 0.0, MONOMER_ADV_CAP);
}

@compute @workgroup_size(16, 16)
fn transport_quanta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    let idx = y * GRID_SIZE + x;
    let rseed = params.seed * 1597334677u ^ params.epoch * 3812015801u;

    // Célula do ambiente -> posição no MUNDO (não passar coordenadas de
    // célula a funções de mundo: no v3 isso congelou todos os monómeros).
    let cell_w = f32(WORLD_UNITS_PER_CELL);
    var v = vec2<f32>(0.0);
    if (params.fluid_enabled != 0u) {
        v = fluid_velocity_at_world(vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) * cell_w);
    }
    let p_adv = parcel_hop_p(v);
    let agitation = clamp(length(v) / DIFF_AGITATION_SPEED, 0.0, 1.0);
    let p_diff = clamp(DIFF_HOP_P * mix(DIFF_HOP_FLOOR, 1.0, agitation) * max(params.diffusion, 0.0), 0.0, 0.5);

    let g_src = gamma_count(idx);
    let src_total = chem_cell_total(idx);
    let src_over = src_total > chem_capacity(idx);
    var p_adv_eff = p_adv;
    var p_diff_eff = p_diff;
    if (g_src > 0u) {
        // Dentro do terreno não há correntes; só difusão lenta.
        p_adv_eff = 0.0;
        p_diff_eff = DIFF_HOP_P * BURIED_DIFF_FACTOR / (1.0 + GAMMA_POROSITY_K * f32(g_src));
    }
    if (src_over) {
        p_diff_eff = CHEM_SQUEEZE_P;
    }

    let light_t = uv_light_at_cell(x, y);

    for (var ch = 0u; ch < 4u; ch++) {
        let slot = idx * 4u + ch;
        let v_ch = atomicLoad(&chem_grid[slot]);
        let act_n = v_ch & CHEM_STATE_MASK;
        let spent_n = v_ch >> 16u;

        // FOTOATIVAÇÃO (com sensibilização: os ativados são antenas; a
        // energia continua a vir só da luz).
        if (spent_n > 0u) {
            let sens = 1.0 + CHEM_SENSITIZE * f32(min(act_n, 8u));
            let exp_act = f32(min(spent_n, 8u)) * LIGHT_ACT_P * max(params.uv_strength, 0.0) * light_t * sens;
            let wa = u32(floor(exp_act));
            var na = wa;
            if (hash_f32(slot ^ rseed) < exp_act - f32(wa)) { na += 1u; }
            for (var ai = 0u; ai < min(na, 8u); ai++) {
                if (!chem_activate_one(slot)) { break; }
            }
        }
        // DECAIMENTO com BLINDAGEM.
        if (act_n > 0u) {
            let shield = 1.0 / (1.0 + CHEM_SHIELD * f32(min(act_n, 8u) - 1u));
            let exp_dec = f32(min(act_n, 8u)) * CHEM_DECAY_P * shield;
            if (hash_f32(slot ^ (params.seed * 2654435761u) ^ (params.epoch * 668265263u)) < exp_dec) {
                chem_spend_one(slot);
            }
        }

        // COESÃO: os ativados difundem-se de preferência para vizinhos ricos
        // em ativados do mesmo tipo (gotículas, coacervados).
        var coh = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
        if (act_n > 0u && params.cohesion > 0.0) {
            if (x + 1u < GRID_SIZE) { coh[0] = 1.0 + params.cohesion * f32(min(chem_act_count(idx + 1u, ch), 8u)); }
            if (x > 0u) { coh[1] = 1.0 + params.cohesion * f32(min(chem_act_count(idx - 1u, ch), 8u)); }
            if (y + 1u < GRID_SIZE) { coh[2] = 1.0 + params.cohesion * f32(min(chem_act_count(idx + GRID_SIZE, ch), 8u)); }
            if (y > 0u) { coh[3] = 1.0 + params.cohesion * f32(min(chem_act_count(idx - GRID_SIZE, ch), 8u)); }
        }
        let coh_sum = coh[0] + coh[1] + coh[2] + coh[3];

        let n = min(act_n + spent_n, MAX_HOPS_PER_CELL_CH);
        for (var k = 0u; k < n; k++) {
            let h = hash(slot ^ (k * 668265263u) ^ (params.seed * 2246822519u) ^ (params.epoch * 374761393u));
            let r = f32(h) / 4294967295.0;
            var dx = 0;
            var dy = 0;
            var is_adv = false;
            let pick_spent = hash_f32(h ^ 0x51ED270Bu) < f32(spent_n) / f32(max(act_n + spent_n, 1u));
            if (r < p_adv_eff) {
                // Salto a jusante: eixo escolhido pelo peso |vx| vs |vy|.
                is_adv = true;
                let ax = abs(v.x);
                let ay = abs(v.y);
                if (hash_f32(h ^ 0x85EBCA6Bu) < ax / max(ax + ay, 1e-6)) {
                    dx = select(-1, 1, v.x > 0.0);
                } else {
                    dy = select(-1, 1, v.y > 0.0);
                }
            } else if (r < p_adv_eff + p_diff_eff) {
                var d = hash(h ^ 0xC2B2AE35u) & 3u;
                if (!pick_spent && coh_sum > 4.0001) {
                    var u = hash_f32(h ^ 0x9E3779B9u) * coh_sum;
                    d = 3u;
                    for (var cd = 0u; cd < 4u; cd++) {
                        if (u < coh[cd]) { d = cd; break; }
                        u -= coh[cd];
                    }
                }
                if (d == 0u) { dx = 1; } else if (d == 1u) { dx = -1; }
                else if (d == 2u) { dy = 1; } else { dy = -1; }
            } else if (g_src == 0u && r < p_adv_eff + p_diff_eff + MONOMER_SETTLE_P * max(params.settle, 0.0)) {
                // ASSENTAMENTO para o fundo (-y). Não entra em rocha: o
                // sedimento pousa EM CIMA das pedras.
                dy = -1;
                if (y > 0u && gamma_count(idx - GRID_SIZE) > 0u) { continue; }
            } else {
                continue;
            }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) { continue; }
            let t_idx = u32(ny) * GRID_SIZE + u32(nx);
            let g_tgt = gamma_count(t_idx);
            let tgt_total = chem_cell_total(t_idx);
            if (tgt_total >= chem_capacity(t_idx)) {
                // Percolação por sobrepressão: uma célula acima da capacidade
                // pode empurrar para um vizinho não mais carregado (<=, não <:
                // com < um campo uniforme de entulho congelava).
                if (!(src_over && tgt_total <= src_total)) { continue; }
            }
            if (g_tgt > 0u) {
                // As correntes nunca empurram monómeros para dentro da rocha.
                if (is_adv) { continue; }
                let gperm = 1.0 / (1.0 + GAMMA_POROSITY_K * f32(g_tgt));
                if (hash_f32(h ^ 0x27D4EB2Fu) >= gperm) { continue; }
            }
            if (chem_take_state_one(slot, pick_spent)) {
                chem_add_state(t_idx, ch, 1u, pick_spent);
            }
        }
    }
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
            if (hash_f32(slot ^ (params.epoch * 0x85EBCA6Bu) ^ (params.seed * 0x51ED270Bu)) < p_act) {
                chem_activate_one(slot);
            }
        }
    }
}
