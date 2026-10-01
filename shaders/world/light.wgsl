// LUZ UV COM SOMBRAS DO TERRENO (v3 compute_uv_light).
// Regra do Filipe: do topo (y alto) para baixo, cada célula recebe a MÉDIA
// das 3 células de cima (taps espaçados UV_SPREAD), × atenuação da água por
// linha × absorção pelo gamma (exp(-0,6·g)). As sombras alargam e suavizam
// com a profundidade, como luz difusa. Um só workgroup de 256 threads varre
// as linhas com barreiras entre elas.

const UV_SHADOW_ABSORB: f32 = 0.6;
const UV_SWEEP_THREADS: u32 = 256u;
const UV_SPREAD: u32 = 4u;

fn uv_light_at_cell(x: u32, y: u32) -> f32 {
    return light_grid[min(y, GRID_SIZE - 1u) * GRID_SIZE + min(x, GRID_SIZE - 1u)];
}

@compute @workgroup_size(256)
fn compute_uv_light(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    let cols = (GRID_SIZE + UV_SWEEP_THREADS - 1u) / UV_SWEEP_THREADS;
    // Ao longo da altura toda a luz cai exp(-uv_depth): independente da resolução.
    let row_water = exp(-max(params.uv_depth, 0.5) / f32(GRID_SIZE));
    let top = GRID_SIZE - 1u;
    for (var c = 0u; c < cols; c++) {
        let x = lid * cols + c;
        if (x >= GRID_SIZE) { continue; }
        let idx = top * GRID_SIZE + x;
        light_grid[idx] = row_water * exp(-UV_SHADOW_ABSORB * f32(gamma_count(idx)));
    }
    storageBarrier();
    workgroupBarrier();
    for (var t = 1u; t < GRID_SIZE; t++) {
        let y = GRID_SIZE - 1u - t;
        let up = (y + 1u) * GRID_SIZE;
        for (var c = 0u; c < cols; c++) {
            let x = lid * cols + c;
            if (x >= GRID_SIZE) { continue; }
            let xl = select(x - UV_SPREAD, 0u, x < UV_SPREAD);
            let xr = min(x + UV_SPREAD, GRID_SIZE - 1u);
            let above = (light_grid[up + xl] + light_grid[up + x] + light_grid[up + xr]) / 3.0;
            let idx = y * GRID_SIZE + x;
            light_grid[idx] = above * row_water * exp(-UV_SHADOW_ABSORB * f32(gamma_count(idx)));
        }
        storageBarrier();
        workgroupBarrier();
    }
}
