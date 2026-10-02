// Acessores da grelha de monómeros (portados do v3, shared.wgsl).
// Todas as mudanças de matéria são trocas inteiras e atómicas: um monómero
// existe ou não existe. Nada aqui cria nem destrói matéria; só a move ou
// muda de estado (ativado <-> gasto) no lugar.

const CHEM_STATE_MASK: u32 = 0xFFFFu;
const CHEM_SPENT_ONE: u32 = 0x10000u;

fn chem_act_count(idx: u32, ch: u32) -> u32 {
    return atomicLoad(&chem_grid[idx * 4u + ch]) & CHEM_STATE_MASK;
}

fn chem_spent_count(idx: u32, ch: u32) -> u32 {
    return atomicLoad(&chem_grid[idx * 4u + ch]) >> 16u;
}

fn chem_cell_total(cell: u32) -> u32 {
    var t = 0u;
    for (var c = 0u; c < 4u; c++) {
        let v = atomicLoad(&chem_grid[cell * 4u + c]);
        t += (v & CHEM_STATE_MASK) + (v >> 16u);
    }
    return t;
}

// Capacidade de uma célula (todos os canais e estados). A ROCHA (>=
// GAMMA_SOLID_THRESHOLD grãos) não guarda monómeros; o ENTULHO é poroso:
// cada grão tira 1/GAMMA_SOLID_THRESHOLD do espaço (1 grão: 2/3, 2: 1/3).
fn chem_capacity(cell: u32) -> u32 {
    let g = gamma_count(cell);
    if (g >= GAMMA_SOLID_THRESHOLD) { return 0u; }
    return CHEM_CELL_CAP * (GAMMA_SOLID_THRESHOLD - g) / GAMMA_SOLID_THRESHOLD;
}

// Troca atómica genérica: se `old` permitir, substitui por `old + delta`
// (aritmética modular em u32: -1 ativado = +0xFFFFFFFF; -1 gasto = +0xFFFF0000;
// gasto->ativado = +0xFFFF0001. Literais, porque overflow numa expressão
// constante é erro na especificação WGSL).
// `need_act` / `need_spent`: exige pelo menos um monómero nesse estado.
fn chem_cas(slot: u32, delta: u32, need_act: bool, need_spent: bool) -> bool {
    var done = false;
    loop {
        let old = atomicLoad(&chem_grid[slot]);
        if (need_act && (old & CHEM_STATE_MASK) == 0u) { break; }
        if (need_spent && (old >> 16u) == 0u) { break; }
        let res = atomicCompareExchangeWeak(&chem_grid[slot], old, old + delta);
        if (res.exchanged) { done = true; break; }
    }
    return done;
}

// ALIMENTAÇÃO: colhe a ativação de um monómero, que fica no lugar, gasto.
fn chem_spend_one(slot: u32) -> bool {
    return chem_cas(slot, CHEM_SPENT_ONE - 1u, true, false);
}

// REATIVAÇÃO (UV, calor): gasto -> ativado no lugar.
fn chem_activate_one(slot: u32) -> bool {
    return chem_cas(slot, 0xFFFF0001u, false, true);
}

// Transporte: tira um monómero num estado ESPECÍFICO (os saltos preservam o estado).
fn chem_take_state_one(slot: u32, spent: bool) -> bool {
    if (spent) {
        return chem_cas(slot, 0xFFFF0000u, false, true);
    }
    return chem_cas(slot, 0xFFFFFFFFu, true, false);
}

fn chem_add_state(idx: u32, ch: u32, n: u32, spent: bool) {
    if (n > 0u) {
        atomicAdd(&chem_grid[idx * 4u + ch], select(n, n * CHEM_SPENT_ONE, spent));
    }
}

// O terreno (entulho ou rocha) não guarda monómeros: um depósito apontado a
// uma célula com gamma vai para a célula de água livre mais próxima (anéis
// até raio 12, como no v3). Se estiver tudo cercado, fica na própria célula
// e o squeeze trata do excesso.
fn chem_open_cell(idx: u32) -> u32 {
    if (gamma_count(idx) == 0u) { return idx; }
    let gx = i32(idx % GRID_SIZE);
    let gy = i32(idx / GRID_SIZE);
    for (var r = 1; r <= 12; r++) {
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                if (abs(dx) != r && abs(dy) != r) { continue; }
                let qx = gx + dx;
                let qy = gy + dy;
                if (qx < 0 || qy < 0 || qx >= i32(GRID_SIZE) || qy >= i32(GRID_SIZE)) { continue; }
                let q = u32(qy) * GRID_SIZE + u32(qx);
                if (gamma_count(q) == 0u) { return q; }
            }
        }
    }
    return idx;
}
