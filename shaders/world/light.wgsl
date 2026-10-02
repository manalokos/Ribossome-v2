// LUZ UV COM SOMBRAS DO TERRENO E DOS AGENTES (v3 compute_uv_light).
// Regra do Filipe: do topo (y alto) para baixo, cada célula recebe a média
// das células vizinhas de cima, × atenuação da água × absorção pelo terreno
// (exp(-0,6·g) por grão) e pelos agentes (o RNA e as proteínas absorvem UV:
// exp(-AGENT_UV_ABSORB) por resíduo). As sombras alargam e suavizam com a
// profundidade, como luz difusa.
// Calculada a 1/LIGHT_DIV da resolução (a varredura é sequencial nas
// linhas: a 2048² custava ~7 ms). Cada linha da luz cobre 2 linhas da
// grelha; a média de cima usa pesos binomiais (1 4 6 4 1)/16 sobre x−2..x+2,
// que espalha por linha da luz o mesmo que a média simples de 5 vizinhos em
// 2 linhas da grelha.
// Por omissão a luz PROPAGA-SE (light_propagate): em cada passo cada linha
// recebe a luz da linha de cima do passo ANTERIOR, todas em paralelo. O
// equilíbrio é o mesmo da varredura, mas a luz desce light_rows_per_step
// linhas por passo (uma "velocidade da luz" finita). A varredura inteira
// (compute_uv_light) só corre na sementeira e quando o terreno muda de vez.

const UV_SHADOW_ABSORB: f32 = 0.6;
// Profundidade ótica de um resíduo de agente.
const AGENT_UV_ABSORB: f32 = 0.15;
const UV_SWEEP_THREADS: u32 = 256u;

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
fn light_transmit(lx: u32, ly: u32) -> f32 {
    var g = 0.0;
    for (var dy = 0u; dy < LIGHT_DIV; dy++) {
        for (var dx = 0u; dx < LIGHT_DIV; dx++) {
            g += f32(gamma_count((ly * LIGHT_DIV + dy) * GRID_SIZE + lx * LIGHT_DIV + dx));
        }
    }
    // Soma das LIGHT_DIV linhas atravessadas, média das LIGHT_DIV colunas.
    g /= f32(LIGHT_DIV);
    let shade = f32(atomicLoad(&shade_grid[ly * LIGHT_SIZE + lx]));
    return exp(-UV_SHADOW_ABSORB * g - AGENT_UV_ABSORB * shade);
}

@compute @workgroup_size(256)
fn clear_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i < LIGHT_SIZE * LIGHT_SIZE) { atomicStore(&shade_grid[i], 0u); }
}

// Pré-passo PARALELO: escreve a transmissão de cada célula em light_grid;
// a varredura (sequencial nas linhas) só multiplica.
@compute @workgroup_size(256)
fn light_transmit_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= LIGHT_SIZE * LIGHT_SIZE) { return; }
    light_grid[i] = light_transmit(i % LIGHT_SIZE, i / LIGHT_SIZE);
}

@compute @workgroup_size(256)
fn compute_uv_light(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    let cols = (LIGHT_SIZE + UV_SWEEP_THREADS - 1u) / UV_SWEEP_THREADS;
    // Ao longo da altura toda a luz cai exp(-uv_depth): independente da resolução.
    let row_water = exp(-max(params.uv_depth, 0.5) * f32(LIGHT_DIV) / f32(GRID_SIZE));
    let top = LIGHT_SIZE - 1u;
    for (var c = 0u; c < cols; c++) {
        let x = lid * cols + c;
        if (x >= LIGHT_SIZE) { continue; }
        light_grid[top * LIGHT_SIZE + x] *= row_water;
    }
    storageBarrier();
    workgroupBarrier();
    var w = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    for (var t = 1u; t < LIGHT_SIZE; t++) {
        let y = LIGHT_SIZE - 1u - t;
        let up = (y + 1u) * LIGHT_SIZE;
        for (var c = 0u; c < cols; c++) {
            let x = lid * cols + c;
            if (x >= LIGHT_SIZE) { continue; }
            var above = 0.0;
            for (var d = 0u; d < 5u; d++) {
                // x−2..x+2, com as paredes a refletir (borda repetida).
                let xs = clamp(i32(x) + i32(d) - 2, 0, i32(LIGHT_SIZE) - 1);
                above += w[d] * light_grid[up + u32(xs)];
            }
            // light_grid ainda tem a transmissão desta célula (pré-passo).
            light_grid[y * LIGHT_SIZE + x] *= above * row_water;
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
    let row_water = exp(-max(params.uv_depth, 0.5) * f32(LIGHT_DIV) / f32(GRID_SIZE));
    var above = 1.0;
    if (y + 1u < LIGHT_SIZE) {
        var w = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
        above = 0.0;
        for (var d = 0u; d < 5u; d++) {
            let xs = clamp(i32(x) + i32(d) - 2, 0, i32(LIGHT_SIZE) - 1);
            above += w[d] * light_grid[(y + 1u) * LIGHT_SIZE + u32(xs)];
        }
    }
    light_next[i] = above * row_water * light_transmit(x, y);
}

@compute @workgroup_size(256)
fn light_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= LIGHT_SIZE * LIGHT_SIZE) { return; }
    light_grid[i] = light_next[i];
}
