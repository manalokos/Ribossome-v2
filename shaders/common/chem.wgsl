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

// Capacidade de uma célula (todos os canais e estados): QUALQUER gamma
// (rocha ou entulho) não guarda monómeros. O terreno é matéria sólida.
fn chem_capacity(cell: u32) -> u32 {
    if (gamma_count(cell) > 0u) { return 0u; }
    return CHEM_CELL_CAP;
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
