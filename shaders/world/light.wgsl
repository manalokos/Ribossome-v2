// LUZ UV COM SOMBRAS DO TERRENO E DOS AGENTES (v3 compute_uv_light).
// Regra do Filipe: do topo (y alto) para baixo, cada célula recebe a média
// das células vizinhas de cima, × atenuação da água × absorção pelo terreno
// (exp(-0,6·g) por grão), pelos MONÓMEROS (os nucleótidos absorvem UV,
// ativados ou gastos: zonas densas fazem sombra às de baixo) e pelos agentes
// (só os aromáticos W > Y > F e os fotossistemas: exp(-AGENT_UV_ABSORB) por
// triptofano equivalente). As sombras alargam e suavizam com a
// profundidade, como luz difusa.
// Calculada a 1/LIGHT_DIV da resolução (a varredura é sequencial nas
// linhas: a 2048² custava ~7 ms). Cada linha da luz cobre 2 linhas da
// grelha; a média de cima é a média simples de 7 vizinhos (light_above).
// Por omissão a luz PROPAGA-SE (light_propagate): em cada passo cada linha
// recebe a luz da linha de cima do passo ANTERIOR, todas em paralelo. O
// equilíbrio é o mesmo da varredura, mas a luz desce light_rows_per_step
// linhas por passo (uma "velocidade da luz" finita). A varredura inteira
// (compute_uv_light) só corre na sementeira e quando o terreno muda de vez.

const UV_SHADOW_ABSORB: f32 = 0.6;
// Profundidade ótica de um triptofano (ou fotossistema) de agente.
const AGENT_UV_ABSORB: f32 = 0.15;
const UV_SWEEP_THREADS: u32 = 256u;

// DUAS BANDAS DE LUZ, cada uma com o seu campo e a sua sombra (a segunda
// metade de light_grid, light_next e shade_grid: índice + banda × LIGHT_CELLS):
//  0 = a luz de CIMA (a de sempre, o "vermelho" do mar): absorvem-na a água,
//      os monómeros, os aromáticos dos corpos e os fotossistemas dela;
//  1 = a luz FUNDA (o "azul"): a água e os monómeros só lhe tiram a fração
//      DEEP_LIGHT_DEPTH do que tiram à outra, os aromáticos não a absorvem,
//      e só os fotossistemas DELA lhe fazem sombra. O terreno tapa as duas.
// Quem absorve uma cor faz sombra nessa cor e deixa passar a outra. A funda
// rende DEEP_LIGHT_YIELD da outra por unidade de luz (lifecycle.wgsl).
const LIGHT_CELLS: u32 = LIGHT_SIZE * LIGHT_SIZE;
const LIGHT_BANDS: u32 = 2u;
const DEEP_LIGHT_DEPTH: f32 = 0.35;
const DEEP_LIGHT_YIELD: f32 = 0.4;
// Atenuação pela água numa linha da luz, para cada banda.
fn row_water_of(band: u32) -> f32 {
    return exp(-max(params.uv_depth, 0.0) * select(1.0, DEEP_LIGHT_DEPTH, band == 1u) * f32(LIGHT_DIV) / f32(GRID_SIZE));
}

fn uv_light_at_cell(x: u32, y: u32) -> f32 {
    let lx = min(x, GRID_SIZE - 1u) / LIGHT_DIV;
    let ly = min(y, GRID_SIZE - 1u) / LIGHT_DIV;
    return light_grid[ly * LIGHT_SIZE + lx];
}

fn uv_light_at_idx(idx: u32) -> f32 {
    return uv_light_at_cell(idx % GRID_SIZE, idx / GRID_SIZE);
}

// Transmissão de uma célula da luz: terreno (média das linhas da grelha que
// cobre) e agentes.
fn light_transmit(lx: u32, ly: u32, band: u32) -> f32 {
    var g = 0.0;
    var m = 0u;
    for (var dy = 0u; dy < LIGHT_DIV; dy++) {
        for (var dx = 0u; dx < LIGHT_DIV; dx++) {
            let c = (ly * LIGHT_DIV + dy) * GRID_SIZE + lx * LIGHT_DIV + dx;
            g += f32(gamma_count(c));
            m += chem_cell_total(c);
        }
    }
    // Soma das LIGHT_DIV linhas atravessadas, média das LIGHT_DIV colunas.
    g /= f32(LIGHT_DIV);
    // Monómeros: densidade média por célula × fração da altura atravessada
    // (somada do topo ao fundo dá monomer_uv_absorb × densidade média).
    let dens = f32(m) / f32(LIGHT_DIV * LIGHT_DIV);
    let tau_m = max(params.monomer_uv_absorb, 0.0) * dens * f32(LIGHT_DIV) / f32(GRID_SIZE) * select(1.0, DEEP_LIGHT_DEPTH, band == 1u);
    // Sombra dos agentes NESTA banda, em "triptofanos equivalentes" (ponto
    // fixo SHADE_ONE).
    let shade = f32(atomicLoad(&shade_grid[band * LIGHT_CELLS + ly * LIGHT_SIZE + lx])) / f32(SHADE_ONE);
    return exp(-UV_SHADOW_ABSORB * g - AGENT_UV_ABSORB * shade - tau_m);
}

// Média SIMPLES dos LIGHT_TAPS vizinhos da linha `row` (x−3..x+3), com as
// paredes a refletir (borda repetida). 7 vizinhos a meia resolução: sombras
// difusas (o Filipe pediu mais difusas que com 5).
const LIGHT_TAPS: i32 = 7;
// A janela desloca-se params.sun_slope células por linha (o sol inclinado: a
// luz de (x, y) vem de (x − inclinação, y + 1)), com interpolação linear da
// parte fracionária: as sombras rodam com a hora do dia.
fn light_above(x: u32, row: u32, band: u32) -> f32 {
    let sh = -params.sun_slope;
    let base = i32(floor(sh));
    let f = sh - f32(base);
    var s = 0.0;
    for (var d = 0; d < LIGHT_TAPS; d++) {
        let x0 = clamp(i32(x) + d - LIGHT_TAPS / 2 + base, 0, i32(LIGHT_SIZE) - 1);
        let x1 = clamp(x0 + 1, 0, i32(LIGHT_SIZE) - 1);
        s += mix(light_grid[band * LIGHT_CELLS + row * LIGHT_SIZE + u32(x0)], light_grid[band * LIGHT_CELLS + row * LIGHT_SIZE + u32(x1)], f);
    }
    // As paredes do lado são opacas: se a luz viria de fora do mundo (sol
    // inclinado junto à parede), não entra. A média de 7 só usa o bordo.
    let cpos = f32(x) + sh;
    let inside = clamp(cpos + 1.0, 0.0, 1.0) * clamp(f32(LIGHT_SIZE) - cpos, 0.0, 1.0);
    return s / f32(LIGHT_TAPS) * inside;
}

@compute @workgroup_size(256)
fn clear_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i < LIGHT_CELLS) {
        atomicStore(&shade_grid[i], 0u);
        atomicStore(&shade_grid[LIGHT_CELLS + i], 0u);
    }
}

// Pré-passo PARALELO: escreve a transmissão de cada célula em light_grid;
// a varredura (sequencial nas linhas) só multiplica.
@compute @workgroup_size(256)
fn light_transmit_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= LIGHT_SIZE * LIGHT_SIZE) { return; }
    light_grid[i] = light_transmit(i % LIGHT_SIZE, i / LIGHT_SIZE, 0u);
    light_grid[LIGHT_CELLS + i] = light_transmit(i % LIGHT_SIZE, i / LIGHT_SIZE, 1u);
}

@compute @workgroup_size(256)
fn compute_uv_light(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    let cols = (LIGHT_SIZE + UV_SWEEP_THREADS - 1u) / UV_SWEEP_THREADS;
    // Ao longo da altura toda a luz cai exp(-uv_depth): independente da resolução.
    let top = LIGHT_SIZE - 1u;
    for (var c = 0u; c < cols; c++) {
        let x = lid * cols + c;
        if (x >= LIGHT_SIZE) { continue; }
        for (var band = 0u; band < LIGHT_BANDS; band++) {
            light_grid[band * LIGHT_CELLS + top * LIGHT_SIZE + x] *= row_water_of(band) * params.sun_now;
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var t = 1u; t < LIGHT_SIZE; t++) {
        let y = LIGHT_SIZE - 1u - t;
            for (var c = 0u; c < cols; c++) {
            let x = lid * cols + c;
            if (x >= LIGHT_SIZE) { continue; }
            for (var band = 0u; band < LIGHT_BANDS; band++) {
                let above = light_above(x, y + 1u, band);
                // light_grid ainda tem a transmissão desta célula (pré-passo).
                light_grid[band * LIGHT_CELLS + y * LIGHT_SIZE + x] *= above * row_water_of(band);
            }
        }
        storageBarrier();
        workgroupBarrier();
    }
}

// PROPAGAÇÃO (paralela): luz nova de cada célula = média binomial da linha
// de cima (do passo anterior) × água × transmissão desta célula.
@compute @workgroup_size(256)
fn light_propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= LIGHT_SIZE * LIGHT_SIZE) { return; }
    let x = i % LIGHT_SIZE;
    let y = i / LIGHT_SIZE;
    for (var band = 0u; band < LIGHT_BANDS; band++) {
        // No topo entra o sol desta hora (de noite, nada).
        var above = params.sun_now;
        if (y + 1u < LIGHT_SIZE) { above = light_above(x, y + 1u, band); }
        light_next[band * LIGHT_CELLS + i] = above * row_water_of(band) * light_transmit(x, y, band);
    }
}

@compute @workgroup_size(256)
fn light_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= LIGHT_SIZE * LIGHT_SIZE) { return; }
    light_grid[i] = light_next[i];
    light_grid[LIGHT_CELLS + i] = light_next[LIGHT_CELLS + i];
}

// Luz ABSORVIDA na célula da luz que contém a célula da grelha `env` (por
// célula da luz): a que chega de cima × (1 − transmissão) + a fração que a
// água absorve. Fonte de calor solar do fluido.
const SUN_WATER_ABSORB: f32 = 0.05;
fn light_absorbed_at(env: u32) -> f32 {
    let lx = (env % GRID_SIZE) / LIGHT_DIV;
    let ly = (env / GRID_SIZE) / LIGHT_DIV;
    var incoming = params.sun_now;
    if (ly + 1u < LIGHT_SIZE) { incoming = light_above(lx, ly + 1u, 0u); }
    incoming = max(incoming, 0.0);
    return incoming * ((1.0 - light_transmit(lx, ly, 0u)) + SUN_WATER_ABSORB);
}
