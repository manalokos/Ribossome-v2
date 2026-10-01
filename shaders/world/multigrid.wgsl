// PRESSÃO POR MULTIGRID (geométrico, centrado nas células).
//
// Resolve a mesma equação que o Jacobi: Σ_{vizinhos de água}(p_n − p_c) = div
// (Laplaciano de 5 pontos; paredes e rochas com Neumann, i.e. não contam),
// com uma ancoragem ε·p minúscula: com Neumann em todo o lado a solução só
// está definida a menos de uma constante, e sem ancoragem essa constante
// derivava até rebentar. O gradiente (que é o que move a água) não muda.
// O Jacobi só corrige bem os erros de pequena escala; o multigrid suaviza
// em cada resolução (1024, 512, …, 8) e corrige as escalas grandes no nível
// grosso, por isso converge em poucos ciclos.
//
// Um ciclo V: em cada nível, de fino para grosso: suaviza (Gauss-Seidel
// vermelho-preto), calcula o resíduo e passa-o ao nível de baixo (média das
// 4 filhas, ×2: operador de Galerkin). No nível mais grosso suaviza
// muitas vezes. De grosso para fino: soma a correção às 4 filhas e suaviza.
// Todos os níveis vivem nos mesmos buffers (mg_p, mg_rhs, mg_fluid), cada um
// no seu offset; o nível atual vem num uniforme com offset dinâmico.

@group(4) @binding(0) var<uniform> mg: MgLevel;
@group(4) @binding(1) var<storage, read_write> mg_p: array<f32>;
@group(4) @binding(2) var<storage, read_write> mg_rhs: array<f32>;
// 1 = célula de água, 0 = sólido (ou fora).
@group(4) @binding(3) var<storage, read_write> mg_fluid: array<u32>;

fn mg_idx(x: u32, y: u32) -> u32 {
    return mg.off + y * mg.n + x;
}

fn mg_is_fluid(x: i32, y: i32) -> bool {
    if (x < 0 || y < 0 || x >= i32(mg.n) || y >= i32(mg.n)) { return false; }
    return mg_fluid[mg_idx(u32(x), u32(y))] != 0u;
}

// Soma dos vizinhos de água e quantos são.
fn mg_neighbors(x: u32, y: u32) -> vec2<f32> {
    var s = 0.0;
    var k = 0.0;
    let xi = i32(x);
    let yi = i32(y);
    if (mg_is_fluid(xi - 1, yi)) { s += mg_p[mg_idx(x - 1u, y)]; k += 1.0; }
    if (mg_is_fluid(xi + 1, yi)) { s += mg_p[mg_idx(x + 1u, y)]; k += 1.0; }
    if (mg_is_fluid(xi, yi - 1)) { s += mg_p[mg_idx(x, y - 1u)]; k += 1.0; }
    if (mg_is_fluid(xi, yi + 1)) { s += mg_p[mg_idx(x, y + 1u)]; k += 1.0; }
    return vec2<f32>(s, k);
}

// Nível 0: máscara de água, lado direito = divergência, arranque quente com
// a pressão anterior (pressure_in = pressure_a).
@compute @workgroup_size(16, 16)
fn mg_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mg.n || gid.y >= mg.n) { return; }
    let i = mg_idx(gid.x, gid.y);
    let f = fgrid(gid.x, gid.y);
    let fluid = !is_effectively_solid(gid.x, gid.y);
    mg_fluid[i] = select(0u, 1u, fluid);
    mg_rhs[i] = select(0.0, divergence[f], fluid);
    mg_p[i] = select(0.0, pressure_in[f], fluid);
}

fn mg_smooth(gid: vec3<u32>, color: u32) {
    if (gid.x >= mg.n || gid.y >= mg.n) { return; }
    if (((gid.x + gid.y) & 1u) != color) { return; }
    let i = mg_idx(gid.x, gid.y);
    if (mg_fluid[i] == 0u) { return; }
    let nb = mg_neighbors(gid.x, gid.y);
    mg_p[i] = (nb.x - mg_rhs[i]) / (nb.y + mg.eps);
}

@compute @workgroup_size(16, 16)
fn mg_smooth_red(@builtin(global_invocation_id) gid: vec3<u32>) {
    mg_smooth(gid, 0u);
}

@compute @workgroup_size(16, 16)
fn mg_smooth_black(@builtin(global_invocation_id) gid: vec3<u32>) {
    mg_smooth(gid, 1u);
}

// Resíduo do nível atual, média das 4 filhas para o nível grosso (n_c, off_c).
// Uma thread por célula GROSSA. Também zera a correção grossa e faz a máscara
// (grossa = água se alguma filha for água).
@compute @workgroup_size(16, 16)
fn mg_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mg.n_c || gid.y >= mg.n_c) { return; }
    var sum = 0.0;
    var nf = 0u;
    for (var c = 0u; c < 4u; c++) {
        let fx = gid.x * 2u + (c & 1u);
        let fy = gid.y * 2u + (c >> 1u);
        let i = mg_idx(fx, fy);
        if (mg_fluid[i] == 0u) { continue; }
        let nb = mg_neighbors(fx, fy);
        // Resíduo: rhs − (Σ_água (p_n − p_c) − ε·p_c).
        sum += mg_rhs[i] - (nb.x - (nb.y + mg.eps) * mg_p[i]);
        nf += 1u;
    }
    let ci = mg.off_c + gid.y * mg.n_c + gid.x;
    mg_fluid[ci] = select(0u, 1u, nf > 0u);
    // Média das filhas de água, ×2: com média na restrição e cópia no
    // prolongamento, o operador grosso consistente (Galerkin) é metade do
    // Laplaciano sem escala, logo A_c·e = 2·r. (Com ×4, a correção grossa
    // era o dobro e um ciclo V divergia.)
    mg_rhs[ci] = select(0.0, 2.0 * sum / f32(max(nf, 1u)), nf > 0u);
    mg_p[ci] = 0.0;
}

// Soma a correção grossa às 4 filhas (uma thread por célula FINA).
@compute @workgroup_size(16, 16)
fn mg_prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mg.n || gid.y >= mg.n) { return; }
    let i = mg_idx(gid.x, gid.y);
    if (mg_fluid[i] == 0u) { return; }
    mg_p[i] += mg_p[mg.off_c + (gid.y / 2u) * mg.n_c + gid.x / 2u];
}

// Nível 0 -> pressão do fluido (pressure_out do bind group "ba" = pressure_a).
@compute @workgroup_size(16, 16)
fn mg_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mg.n || gid.y >= mg.n) { return; }
    pressure_out[fgrid(gid.x, gid.y)] = mg_p[mg_idx(gid.x, gid.y)];
}
