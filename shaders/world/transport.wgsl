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
// Saltos extra por unidade de pressão × queda de enchimento.
const PRESSURE_HOP_P: f32 = 0.01;
// Dispersão mecânica no entulho: desvio aleatório / deslocamento médio.
const RUBBLE_DISPERSION: f32 = 1.0;
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
// Hidrólise espontânea da ativação: params.activation_decay (era 0,0002 fixo).
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

// AGREGAÇÃO, passo 1: ativados por célula (todos os canais).
@compute @workgroup_size(16, 16)
fn agg_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= GRID_SIZE || gid.y >= GRID_SIZE) { return; }
    let idx = gid.y * GRID_SIZE + gid.x;
    var c = 0u;
    for (var ch = 0u; ch < 4u; ch++) { c += chem_act_count(idx, ch); }
    agg_act[idx] = c;
}

// AGREGAÇÃO, passo 2: ativados nas 8 células vizinhas.
@compute @workgroup_size(16, 16)
fn agg_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) { return; }
    var sum = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) { continue; }
            sum += agg_act[u32(ny) * GRID_SIZE + u32(nx)];
        }
    }
    agg_nb[y * GRID_SIZE + x] = sum;
}

// SORTEIO SUAVE para arredondar a deslocação dos grumos: ruído de valor
// (um valor por passo a cada SMOOTH_CELL células, interpolado suavemente) +
// um deslocamento global aleatório, módulo 1. Cada célula continua com um
// sorteio UNIFORME em [0, 1) (sem viés no transporte), mas células próximas
// sorteiam quase igual (o grumo move-se inteiro) e zonas afastadas não (sem
// sincronia no mundo todo). A costura (onde dá a volta 1 -> 0) muda de sítio a
// cada passo.
const SMOOTH_CELL: u32 = 16u;

fn lattice_rand(ix: u32, iy: u32) -> vec2<f32> {
    return rng_f4(ix * 73856093u ^ iy * 19349663u, params.epoch, S_MOVE + MAX_MOVERS_PER_CH + 1u).xy;
}

fn smooth_start(x: u32, y: u32) -> vec2<f32> {
    let gx = x / SMOOTH_CELL;
    let gy = y / SMOOTH_CELL;
    let t = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), (vec2<f32>(f32(x % SMOOTH_CELL), f32(y % SMOOTH_CELL)) + 0.5) / f32(SMOOTH_CELL));
    let n = mix(mix(lattice_rand(gx, gy), lattice_rand(gx + 1u, gy), t.x),
                mix(lattice_rand(gx, gy + 1u), lattice_rand(gx + 1u, gy + 1u), t.x), t.y);
    let g = rng_f4(0x9E3779B9u, params.epoch, S_MOVE + MAX_MOVERS_PER_CH).xy;
    return fract(n + g);
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
    if (params.fluid_enabled != 0u && g_src < GAMMA_SOLID_THRESHOLD) {
        // PONTO MÉDIO (Runge-Kutta 2): usa a velocidade a meio do caminho.
        // Com a velocidade só no início (Euler), cada passo segue a tangente
        // e atira os monómeros para fora dos remoinhos: os núcleos esvaziavam
        // (os pontos escuros na pluma).
        let c = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) * cell_w;
        let v0 = fluid_velocity_at_world(c);
        let world_per_fluid = SIM_SIZE / f32(FLUID_SIZE);
        v = fluid_velocity_at_world(c + v0 * world_per_fluid * max(params.dt, 0.0) * 0.5);
        // No ENTULHO (poroso) os monómeros andam com a água: o fluido já o
        // trava por atrito (rubble_drag), por isso esta é a velocidade real
        // da água nos poros. (Sem corrente no entulho ele era um filtro: a
        // água saía limpa e abria caudas vazias atrás; com v/φ esvaziava-se.)
    }
    // SEM PENETRAÇÃO nas paredes à resolução da química (o fluido corre a
    // metade dela: junto à rocha a velocidade interpolada ainda aponta para
    // dentro ou para fora da parede). Contra a rocha: a componente normal é
    // zero (o monómero desliza ao longo da parede em vez de se amontoar).
    // A afastar-se da rocha: metade (a velocidade média a meia célula de uma
    // parede), senão abria-se uma cauda vazia no lado de trás.
    if (any(v != vec2<f32>(0.0))) {
        let rock_r = x + 1u < GRID_SIZE && gamma_count(idx + 1u) >= GAMMA_SOLID_THRESHOLD;
        let rock_l = x > 0u && gamma_count(idx - 1u) >= GAMMA_SOLID_THRESHOLD;
        let rock_u = y + 1u < GRID_SIZE && gamma_count(idx + GRID_SIZE) >= GAMMA_SOLID_THRESHOLD;
        let rock_d = y > 0u && gamma_count(idx - GRID_SIZE) >= GAMMA_SOLID_THRESHOLD;
        if ((v.x > 0.0 && rock_r) || (v.x < 0.0 && rock_l)) { v.x = 0.0; }
        else if ((v.x > 0.0 && rock_l) || (v.x < 0.0 && rock_r)) { v.x *= 0.5; }
        if ((v.y > 0.0 && rock_u) || (v.y < 0.0 && rock_d)) { v.y = 0.0; }
        else if ((v.y > 0.0 && rock_d) || (v.y < 0.0 && rock_u)) { v.y *= 0.5; }
    }
    // Deslocamento por passo, em células do ambiente.
    let disp = v * (f32(GRID_SIZE) / f32(FLUID_SIZE)) * max(params.dt, 0.0);
    let agitation = clamp(length(v) / DIFF_AGITATION_SPEED, 0.0, 1.0);
    var p_diff = clamp(DIFF_HOP_P * mix(DIFF_HOP_FLOOR, 1.0, agitation) * max(params.diffusion, 0.0), 0.0, 0.5);
    if (g_src > 0u) {
        // Difusão no terreno: mais lenta (tortuosidade dos poros).
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

    // PRESSÃO OSMÓTICA: os saltos de difusão preferem a vizinha menos cheia,
    // ∝ à diferença de ENCHIMENTO (contagem / capacidade; assim o entulho,
    // com menos espaço, compara-se bem com a água). Fluxo ∝ gradiente (Fick)
    // que espalha as zonas densas e enche as vazias. Direções: +x, −x, +y, −y.
    var press = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
    var max_drop = 0.0;
    if (params.monomer_pressure > 0.0) {
        let f_self = f32(src_total) / f32(max(chem_capacity(idx), 1u));
        var nb = array<i32, 4>(1, -1, i32(GRID_SIZE), -i32(GRID_SIZE));
        var inside = array<bool, 4>(x + 1u < GRID_SIZE, x > 0u, y + 1u < GRID_SIZE, y > 0u);
        for (var d = 0u; d < 4u; d++) {
            if (!inside[d]) { continue; }
            let n = u32(i32(idx) + nb[d]);
            let cap_n = chem_capacity(n);
            if (cap_n == 0u) { continue; }
            let f_n = f32(chem_cell_total(n)) / f32(cap_n);
            press[d] = max(0.05, 1.0 + params.monomer_pressure * (f_self - f_n));
            max_drop = max(max_drop, f_self - f_n);
        }
    }
    // A pressão também aumenta o FLUXO: saltos extra ∝ à maior queda de
    // enchimento para uma vizinha (uma célula cheia ao lado de uma vazia
    // despeja mais).
    p_diff = clamp(p_diff + PRESSURE_HOP_P * params.monomer_pressure * max_drop, 0.0, 0.5);
    let light_t = uv_light_at_cell(x, y);

    // AGREGAÇÃO dos ativados: gás de rede com atração, dinâmica de KAWASAKI.
    // A energia de um ativado num sítio = −ε × ativados nas 8 células à
    // volta (todos os canais). Um salto de difusão para um sítio com MENOS
    // vizinhos só é aceite com probabilidade exp(−ε·Δ/T); para um com mais,
    // sempre. Assim os soltos vagueiam e são apanhados pelos aglomerados, que
    // crescem (só travar a saída congelava a sopa ao acaso, como um vidro).
    // T = temperatura local (o calor dissolve os grumos). Volume excluído: a
    // atração enfraquece com o enchimento da célula (a pressão espalha os
    // cheios). Os ativados com muitos vizinhos também são levados JUNTOS pela
    // corrente (o grumo viaja inteiro).
    var p_bound = 0.0;
    var accept = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
    if (params.aggregation > 0.0 && src_total <= chem_capacity(idx)) {
        // Vizinhos ativados aqui e nos 4 destinos (pré-calculados em agg_nb).
        var e = array<f32, 5>(f32(agg_nb[idx]), 0.0, 0.0, 0.0, 0.0);
        if (x + 1u < GRID_SIZE) { e[1] = f32(agg_nb[idx + 1u]); }
        if (x > 0u) { e[2] = f32(agg_nb[idx - 1u]); }
        if (y + 1u < GRID_SIZE) { e[3] = f32(agg_nb[idx + GRID_SIZE]); }
        if (y > 0u) { e[4] = f32(agg_nb[idx - GRID_SIZE]); }
        let fx = min((x * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
        let fy = min((y * FLUID_SIZE) / GRID_SIZE, FLUID_SIZE - 1u);
        // T = 1 na água à temperatura ambiente; no limiar da ativação térmica, 2.
        let t_rel = 1.0 + max(temp_in[fgrid(fx, fy)], 0.0) / TEMP_ACT_THRESHOLD;
        let room = clamp(1.0 - 2.0 * f32(src_total) / f32(max(chem_capacity(idx), 1u)), 0.0, 1.0);
        let k_e = params.aggregation * room / t_rel;
        for (var d = 0u; d < 4u; d++) {
            // No destino, a célula de origem (com o próprio) conta como vizinha: −1.
            let drop = e[0] - (e[d + 1u] - 1.0);
            if (drop > 0.0) { accept[d] = exp(-k_e * drop); }
        }
        p_bound = 1.0 - exp(-k_e * min(e[0], 48.0));
    }

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
            let exp_act = f32(min(spent_n, 8u)) * (LIGHT_ACT_P * max(params.uv_strength, 0.0) * max(params.direct_photoactivation, 0.0) * light_t * sens
                + max(params.reactivation_rate, 0.0));
            var na = u32(floor(exp_act));
            // (Sem probabilidade não se sorteia: o resultado seria o mesmo.)
            let frac = exp_act - f32(na);
            if (frac > 0.0 && rng_f4(slot, params.epoch, S_PHOTO).x < frac) { na += 1u; }
            na = min(na, min(spent_n, 8u));
            act_n += na;
            spent_n -= na;
        }
        // DECAIMENTO com blindagem.
        if (act_n > 0u) {
            let shield = 1.0 / (1.0 + CHEM_SHIELD * f32(min(act_n, 8u) - 1u));
            let exp_dec = f32(min(act_n, 8u)) * max(params.activation_decay, 0.0) * shield;
            if (exp_dec > 0.0 && rng_f4(slot, params.epoch, S_DECAY).x < exp_dec) {
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

        // ---- Movimento: histograma de destinos (ativados nos 16 bits baixos) ----
        var bins = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        var stay = 0u;
        let n = act_n + spent_n;
        // Acima de MAX_MOVERS_PER_CH os restantes ficam: conta-os de uma vez
        // (percorrê-los um a um custava O(n) e uma célula entupida com dezenas
        // de milhares de monómeros prendia o passo inteiro).
        let movers = min(n, MAX_MOVERS_PER_CH);
        let act_stay = act_n - min(act_n, movers);
        stay += act_stay + (n - movers - act_stay) * CHEM_SPENT_ONE;
        // Ponto de partida dos ativados presos: um sorteio SUAVE no espaço
        // (ver smooth_start). Com um sorteio por célula, células vizinhas do
        // mesmo grumo arredondavam a deslocação de forma diferente e o grumo
        // rasgava-se (difusão numérica); com um sorteio global, o mundo todo
        // andava em sincronia e abria fendas direitas.
        let bound_start = smooth_start(x, y);
        for (var k = 0u; k < movers; k++) {
            let unit = select(CHEM_SPENT_ONE, 1u, k < act_n);
            // 4 números: ponto de partida (x, y), evento, direção do salto.
            let rf = rng_f4(slot, params.epoch, S_MOVE + k);
            // Preso (só ativados): não difunde e parte do ponto comum.
            let bound = k < act_n && p_bound > 0.0 && fract(rf.z * 7.31 + rf.w * 3.17) < p_bound;
            var p = select(rf.xy, bound_start, bound) + disp;
            if (g_src > 0u && any(disp != vec2<f32>(0.0))) {
                // DISPERSÃO MECÂNICA no entulho: caminhos tortuosos entre os
                // grãos; cada monómero desvia-se ao acaso ∝ à velocidade.
                let rj = rng_f4(slot, params.epoch, S_DISPERSE + k);
                let jitter = (rj.xy * 2.0 - 1.0) * length(disp) * RUBBLE_DISPERSION;
                p += clamp(jitter, vec2<f32>(-0.9), vec2<f32>(0.9));
            }
            let r = rf.z;
            if (r < p_diff) {
                // Direção: pressão × coesão (esta só para os ativados).
                var w = press;
                if (k < act_n) { for (var cd = 0u; cd < 4u; cd++) { w[cd] *= coh[cd]; } }
                var u = rf.w * (w[0] + w[1] + w[2] + w[3]);
                var d = 3u;
                for (var cd = 0u; cd < 4u; cd++) {
                    if (u < w[cd]) { d = cd; break; }
                    u -= w[cd];
                }
                // Kawasaki: um ativado só sai para menos vizinhos com exp(−ε·Δ/T).
                let stay_put = k < act_n && fract(rf.w * 13.73 + rf.x * 5.31) >= accept[d];
                if (!stay_put) {
                    if (d == 0u) { p.x += 1.0; } else if (d == 1u) { p.x -= 1.0; }
                    else if (d == 2u) { p.y += 1.0; } else { p.y -= 1.0; }
                }
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
            // Rocha: a água não entra, o monómero também não. O entulho é
            // poroso (capacidade reduzida, entrada com a permeabilidade);
            // dentro dele não há corrente, só difusão lenta.
            var blocked = false;
            if (t_idx != idx) {
                let g_tgt = gamma_count(t_idx);
                if (g_tgt >= GAMMA_SOLID_THRESHOLD) {
                    // Rocha: não se entra de fora. Um monómero que já esteja
                    // DENTRO de rocha (um grão caiu-lhe em cima numa corrida)
                    // pode atravessá-la até sair: infiltração, para nunca
                    // ficar matéria presa para sempre.
                    blocked = g_src < GAMMA_SOLID_THRESHOLD;
                } else {
                    // Entulho: entra-se como na água (a água dos poros leva o
                    // soluto); só a capacidade, mais pequena, limita.
                    if (chem_cell_total(t_idx) > chem_capacity(t_idx)) {
                        // Volume excluído: uma célula CHEIA não aceita mais
                        // (o monómero fica, como contra a rocha). Sem isto,
                        // um beco de uma célula no terreno, mais fino que a
                        // grelha do fluido, onde a corrente aponta para
                        // dentro, enchia sem limite (150 mil em 10 mil passos).
                        blocked = true;
                    }
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
    let p_act = clamp(FUMAROLE_ACT_P * max(params.thermal_activation, 0.0) * (t_loc - TEMP_ACT_THRESHOLD) / TEMP_ACT_THRESHOLD, 0.0, 0.9);
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
