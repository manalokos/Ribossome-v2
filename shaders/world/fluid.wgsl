// FLUIDO — Stable Fluids em FLUID_SIZE², portado de v3 fluid.wgsl.
// Velocidade em células do fluido por segundo; cima no ecrã = +y.
//
// Ordem por resolução (src/world/fluid.rs):
//   update_temperature -> copy_temperature -> buoyancy -> gather_forces
//   -> add_forces (a->b) -> clear_forces -> diffuse_velocity (b->a)
//   -> advect_velocity (a->b) -> vorticity_confinement (b->a)
//   -> compute_divergence (a) -> jacobi_pressure × N par (pressão em arranque
//      quente: NUNCA se limpa) -> subtract_gradient (a->b)
//   -> enforce_boundaries (b->a). Resultado final em velocity_a.
// Os operadores são consistentes (divergência central, Laplaciano de 5
// pontos, gradiente central): com operadores diferentes ficavam vórtices
// permanentes em malha.

const MAX_DT: f32 = 0.07;
const MAX_FORCE: f32 = 200000.0;
const MAX_VEL: f32 = 2000.0;
const FORCE_SMOOTH_MIX: f32 = 0.25;
const SOLID_PERM_THRESHOLD: f32 = 0.02;

// ---- Temperatura ----
const TEMP_HEAT_RATE: f32 = 0.01;   // T por unidade de força da fumarola por s, no centro
const TEMP_COOL_RATE: f32 = 0.12;   // relaxação para o ambiente local (1/s)
const TEMP_MAX: f32 = 12.0;
const SUN_HEAT_RATE: f32 = 0.15;
// INFRAVERMELHO: a água é transparente à luz visível e ao UV mas absorve o
// infravermelho do sol logo à superfície (cai a 1/e em SUN_IR_DEPTH da altura
// do mundo). Não tira luz aos fotossistemas: só aquece. Com SUN_IR 1,2 a
// superfície fica ~1,5 acima do ambiente ao meio-dia.
const SUN_IR: f32 = 1.2;
const SUN_IR_DEPTH: f32 = 0.08;
const TEMP_BUOYANCY: f32 = 12.0;    // força por unidade de desvio ao ambiente
const TEMP_AMBIENT_SURFACE: f32 = 0.0;
const TEMP_AMBIENT_ATTEN: f32 = 4.0;
const TEMP_DIFFUSE: f32 = 0.08;
// Redutor: largado ∝ calor das fumarolas; ponto fixo do consumo; teto.
const REDOX_RATE: f32 = 0.01;
const REDOX_FP: f32 = 10000.0;
const REDOX_MAX: f32 = 50.0;

fn fgrid(x: u32, y: u32) -> u32 {
    return y * FLUID_SIZE + x;
}

fn temp_ambient_at(y: u32) -> f32 {
    return TEMP_AMBIENT_SURFACE * exp(-TEMP_AMBIENT_ATTEN * (1.0 - (f32(y) + 0.5) / f32(FLUID_SIZE)));
}

fn fluid_dt() -> f32 {
    return clamp(params.fluid_dt, 0.0, MAX_DT);
}

fn is_bad_f32(x: f32) -> bool {
    return (x != x) || (abs(x) > 1e20);
}

fn sanitize_vec2(v: vec2<f32>) -> vec2<f32> {
    if (is_bad_f32(v.x) || is_bad_f32(v.y)) { return vec2<f32>(0.0); }
    return v;
}

fn clamp_vec2_len(v: vec2<f32>, max_len: f32) -> vec2<f32> {
    let len = length(v);
    if (len > max_len) { return v * (max_len / max(len, 1e-12)); }
    return v;
}

// ---- Obstáculos (terreno) ----
// Célula do ambiente debaixo do centro de uma célula do fluido.
fn env_cell_for_fluid(x: u32, y: u32) -> u32 {
    let s = f32(GRID_SIZE) / f32(FLUID_SIZE);
    let gx = u32(clamp((f32(x) + 0.5) * s, 0.0, f32(GRID_SIZE - 1u)));
    let gy = u32(clamp((f32(y) + 0.5) * s, 0.0, f32(GRID_SIZE - 1u)));
    return gy * GRID_SIZE + gx;
}

// Fração de 4 amostras do ambiente que são rocha sólida.
fn gamma_solidity_at_fluid_cell(x: u32, y: u32) -> f32 {
    let scale = f32(GRID_SIZE) / f32(FLUID_SIZE);
    let gx = (f32(x) + 0.5) * scale;
    let gy = (f32(y) + 0.5) * scale;
    let h = scale * 0.25;
    var n_solid = 0u;
    for (var s = 0u; s < 4u; s++) {
        let ox = select(-h, h, (s & 1u) == 1u);
        let oy = select(-h, h, (s & 2u) == 2u);
        let sx = u32(clamp(gx + ox, 0.0, f32(GRID_SIZE - 1u)));
        let sy = u32(clamp(gy + oy, 0.0, f32(GRID_SIZE - 1u)));
        if (gamma_count(sy * GRID_SIZE + sx) >= GAMMA_SOLID_THRESHOLD) { n_solid += 1u; }
    }
    return f32(n_solid) * 0.25;
}

// 1 = água livre, 0 = sólido. Maioria sólida = parede (penínsulas finas
// também bloqueiam); declives fortes são pouco permeáveis:
// perm = 1 / (1 + k·|declive|).
// Só a ROCHA é parede. (O v3 também fazia parede dos declives fortes do
// terreno; com o entulho poroso isso punha "paredes salpicadas" no meio do
// entulho irregular e, à resolução da química, acumulava monómeros de um
// lado e abria caudas vazias do outro. O entulho agora trava por atrito:
// rubble_drag.)
fn permeability(x: u32, y: u32) -> f32 {
    let solidity = gamma_solidity_at_fluid_cell(x, y);
    if (solidity >= 0.5) { return 0.0; }
    return clamp(1.0 - solidity * 2.0, 0.0, 1.0);
}

// ENTULHO POROSO (Brinkman): a água atravessa-o com atrito proporcional aos
// grãos (média das células do ambiente cobertas). Fator por passo.
// (300: a água dos poros fica a ~3–5% da água livre. Mais forte quase não
// abranda: a projeção da pressão força a água a passar; abrandar a sério
// pede a permeabilidade dentro da própria pressão, lei de Darcy.)
const RUBBLE_DRAG: f32 = 300.0;
fn rubble_drag(x: u32, y: u32, dt: f32) -> f32 {
    let scale = GRID_SIZE / FLUID_SIZE;
    var g = 0.0;
    for (var dy = 0u; dy < scale; dy++) {
        for (var dx = 0u; dx < scale; dx++) {
            let c = gamma_count((y * scale + dy) * GRID_SIZE + x * scale + dx);
            if (c < GAMMA_SOLID_THRESHOLD) { g += f32(c); }
        }
    }
    g /= f32(scale * scale) * f32(GAMMA_SOLID_THRESHOLD);
    return 1.0 / (1.0 + RUBBLE_DRAG * g * dt);
}

// DESVIO PELO DECLIVE: mantém |v| e roda a direção para "declive abaixo",
// mais quanto mais desalinhada estiver (independente do dt).
fn slope_steer_velocity(x: u32, y: u32, v_in: vec2<f32>, dt: f32) -> vec2<f32> {
    let s = -sanitize_vec2(slope_grid[env_cell_for_fluid(x, y)]);
    let v_len = length(v_in);
    let s_len = length(s);
    if (v_len < 1e-5 || s_len < 1e-5) { return v_in; }
    let v_dir = v_in / v_len;
    let s_dir = s / s_len;
    let misalign = clamp(1.0 - dot(v_dir, s_dir), 0.0, 2.0);
    let t = clamp(1.0 - exp(-max(params.slope_steer_rate, 0.0) * misalign * dt), 0.0, 1.0);
    let dir_raw = v_dir + (s_dir - v_dir) * t;
    let dir_len = length(dir_raw);
    if (dir_len < 1e-5) { return v_in; }
    return (dir_raw / dir_len) * v_len;
}

fn is_effectively_solid(x: u32, y: u32) -> bool {
    return solid_mask[fgrid(x, y)] != 0u;
}

// Refaz a máscara das paredes (no início de cada passo do fluido).
@compute @workgroup_size(16, 16)
fn build_solid_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    solid_mask[fgrid(gid.x, gid.y)] = select(0u, 1u, permeability(gid.x, gid.y) < SOLID_PERM_THRESHOLD);
}

fn raw_velocity_cell(x: u32, y: u32) -> vec2<f32> {
    if (is_effectively_solid(x, y)) { return vec2<f32>(0.0); }
    return sanitize_vec2(velocity_in[fgrid(x, y)]);
}

// Normal aproximada, para fora, a partir dos vizinhos sólidos (8 vizinhos).
fn solid_normal_from_neighbors(x: u32, y: u32) -> vec2<f32> {
    let xm = select(x - 1u, x, x == 0u);
    let xp = select(x + 1u, x, x + 1u >= FLUID_SIZE);
    let ym = select(y - 1u, y, y == 0u);
    let yp = select(y + 1u, y, y + 1u >= FLUID_SIZE);
    let s_l = select(0.0, 1.0, (x > 0u) && is_effectively_solid(xm, y));
    let s_r = select(0.0, 1.0, (x + 1u < FLUID_SIZE) && is_effectively_solid(xp, y));
    let s_b = select(0.0, 1.0, (y > 0u) && is_effectively_solid(x, ym));
    let s_t = select(0.0, 1.0, (y + 1u < FLUID_SIZE) && is_effectively_solid(x, yp));
    let s_bl = select(0.0, 1.0, (x > 0u) && (y > 0u) && is_effectively_solid(xm, ym));
    let s_tl = select(0.0, 1.0, (x > 0u) && (y + 1u < FLUID_SIZE) && is_effectively_solid(xm, yp));
    let s_br = select(0.0, 1.0, (x + 1u < FLUID_SIZE) && (y > 0u) && is_effectively_solid(xp, ym));
    let s_tr = select(0.0, 1.0, (x + 1u < FLUID_SIZE) && (y + 1u < FLUID_SIZE) && is_effectively_solid(xp, yp));
    let diag = 0.70710678;
    let nx = (s_l - s_r) + diag * ((s_bl + s_tl) - (s_br + s_tr));
    let ny = (s_b - s_t) + diag * ((s_bl + s_br) - (s_tl + s_tr));
    let n = vec2<f32>(nx, ny);
    let n_len = length(n);
    return select(vec2<f32>(0.0), n / n_len, n_len > 1e-6);
}

// Reflete a componente que aponta para dentro de uma parede (como no v3).
fn reflect_if_into_solid(x: u32, y: u32, v_in: vec2<f32>) -> vec2<f32> {
    let n = solid_normal_from_neighbors(x, y);
    let d = dot(v_in, n);
    if (length(n) > 1e-6 && d < 0.0) { return v_in - n * d; }
    return v_in;
}

fn is_solid_at_pos(pos: vec2<f32>) -> bool {
    let ix = i32(floor(pos.x - 0.5));
    let iy = i32(floor(pos.y - 0.5));
    if (ix < 0 || ix >= i32(FLUID_SIZE) || iy < 0 || iy >= i32(FLUID_SIZE)) { return true; }
    return is_effectively_solid(u32(ix), u32(iy));
}

fn clamp_coords(x: i32, y: i32) -> vec2<u32> {
    return vec2<u32>(u32(clamp(x, 0, i32(FLUID_SIZE) - 1)), u32(clamp(y, 0, i32(FLUID_SIZE) - 1)));
}

// Bilinear; fora do domínio vale zero (parede).
fn sample_velocity(pos: vec2<f32>) -> vec2<f32> {
    let min_pos = 0.5;
    let max_pos = f32(FLUID_SIZE) - 0.5;
    if (pos.x < min_pos || pos.x > max_pos || pos.y < min_pos || pos.y > max_pos) { return vec2<f32>(0.0); }
    let x = pos.x - 0.5;
    let y = pos.y - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = fract(x);
    let fy = fract(y);
    let c00 = clamp_coords(x0, y0);
    let c10 = clamp_coords(x0 + 1, y0);
    let c01 = clamp_coords(x0, y0 + 1);
    let c11 = clamp_coords(x0 + 1, y0 + 1);
    let v0 = mix(raw_velocity_cell(c00.x, c00.y), raw_velocity_cell(c10.x, c10.y), fx);
    let v1 = mix(raw_velocity_cell(c01.x, c01.y), raw_velocity_cell(c11.x, c11.y), fx);
    return mix(v0, v1, fy);
}

// Velocidade (células do fluido / s) num ponto do MUNDO, bilinear, com os
// centros das células em +0.5 (sem isto há uma deriva diagonal global).
// Lê velocity_in: os passes de transporte ligam o bind group "ab" (final em a).
fn fluid_velocity_at_world(pos_world: vec2<f32>) -> vec2<f32> {
    let g = f32(FLUID_SIZE);
    let gx = clamp(pos_world.x / SIM_SIZE * g, 0.5, g - 0.5) - 0.5;
    let gy = clamp(pos_world.y / SIM_SIZE * g, 0.5, g - 0.5) - 0.5;
    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));
    let x1 = min(x0 + 1u, FLUID_SIZE - 1u);
    let y1 = min(y0 + 1u, FLUID_SIZE - 1u);
    let tx = fract(gx);
    let ty = fract(gy);
    let v0 = mix(sanitize_vec2(velocity_in[fgrid(x0, y0)]), sanitize_vec2(velocity_in[fgrid(x1, y0)]), tx);
    let v1 = mix(sanitize_vec2(velocity_in[fgrid(x0, y1)]), sanitize_vec2(velocity_in[fgrid(x1, y1)]), tx);
    return mix(v0, v1, ty);
}

// ---- Forças ----
// Soma de forças em PONTO FIXO (inteiro de 32 bits em complemento para 2
// sobre o u32 atómico): uma só soma atómica, sem laço de tentativas (com
// muitos agentes juntos o laço em vírgula flutuante ficava à espera), e a
// soma inteira não depende da ordem (determinística).
const FORCE_FP: f32 = 256.0;
fn atomic_add_force(i: u32, v: f32) {
    let q = i32(round(clamp(v, -1.0e5, 1.0e5) * FORCE_FP));
    if (q != 0) { atomicAdd(&force_vectors[i], bitcast<u32>(q)); }
}

fn force_at_fp(i: u32) -> f32 {
    return f32(bitcast<i32>(atomicLoad(&force_vectors[i]))) / FORCE_FP;
}

@compute @workgroup_size(16, 16)
fn clear_force_vectors(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    let idx = fgrid(gid.x, gid.y);
    atomicStore(&force_vectors[idx * 2u], 0u);
    atomicStore(&force_vectors[idx * 2u + 1u], 0u);
}

// TEMPERATURA: advecção semi-Lagrangiana BILINEAR (com arredondamento à
// célula o calor nunca se movia) + difusão + fumarolas + sol + arrefecimento.
@compute @workgroup_size(16, 16)
fn update_temperature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    let dt = fluid_dt();

    let vel = raw_velocity_cell(x, y);
    let src = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) - vel * dt;
    let sxf = clamp(src.x - 0.5, 0.0, f32(FLUID_SIZE - 1u));
    let syf = clamp(src.y - 0.5, 0.0, f32(FLUID_SIZE - 1u));
    let x0 = u32(floor(sxf));
    let y0 = u32(floor(syf));
    let x1 = min(x0 + 1u, FLUID_SIZE - 1u);
    let y1 = min(y0 + 1u, FLUID_SIZE - 1u);
    let fx = fract(sxf);
    let fy = fract(syf);
    var t = mix(mix(temp_in[fgrid(x0, y0)], temp_in[fgrid(x1, y0)], fx),
                mix(temp_in[fgrid(x0, y1)], temp_in[fgrid(x1, y1)], fx), fy);

    let tl = temp_in[fgrid(u32(max(i32(x) - 1, 0)), y)];
    let tr = temp_in[fgrid(min(x + 1u, FLUID_SIZE - 1u), y)];
    let tb = temp_in[fgrid(x, u32(max(i32(y) - 1, 0)))];
    let tt = temp_in[fgrid(x, min(y + 1u, FLUID_SIZE - 1u))];
    t = mix(t, (tl + tr + tb + tt) * 0.25, TEMP_DIFFUSE);

    // Fumarolas: mapa de calor por célula (CPU: pontuais + píxeis vermelhos).
    t += heat_src[idx] * TEMP_HEAT_RATE * dt;

    // SOL: o calor que entra é a luz ABSORVIDA nesta célula (energia
    // conservada): a que chega de cima × (1 − transmissão), com terreno,
    // monómeros e agentes, mais um pouco pela água. Aquecer por igual de cima
    // estratifica (estável); a CONVECÇÃO nasce onde a absorção é desigual na
    // horizontal: rocha iluminada, manchas densas de monómeros, colónias.
    let env = env_cell_for_fluid(x, y);
    let depth = 1.0 - (f32(y) + 0.5) / f32(FLUID_SIZE);
    let ir = params.sun_now * SUN_IR * exp(-depth / SUN_IR_DEPTH);
    t += max(params.sun_heat, 0.0) * SUN_HEAT_RATE * (light_absorbed_at(env) + ir) * dt;

    let amb = temp_ambient_at(y);
    t = amb + (t - amb) * exp(-TEMP_COOL_RATE * dt);
    temp_out[idx] = clamp(t, 0.0, TEMP_MAX);

    // REDUTOR: a mesma advecção e difusão; nasce nas fumarolas, perde o
    // que os quimiossintéticos comeram e oxida-se devagar (mais devagar do
    // que o calor arrefece: chega mais longe do que a zona que mata).
    var r = mix(mix(redox_in[fgrid(x0, y0)], redox_in[fgrid(x1, y0)], fx),
                mix(redox_in[fgrid(x0, y1)], redox_in[fgrid(x1, y1)], fx), fy);
    let rl = redox_in[fgrid(u32(max(i32(x) - 1, 0)), y)];
    let rr = redox_in[fgrid(min(x + 1u, FLUID_SIZE - 1u), y)];
    let rb = redox_in[fgrid(x, u32(max(i32(y) - 1, 0)))];
    let rt = redox_in[fgrid(x, min(y + 1u, FLUID_SIZE - 1u))];
    r = mix(r, (rl + rr + rb + rt) * 0.25, TEMP_DIFFUSE);
    r += heat_src[idx] * REDOX_RATE * dt;
    r = max(r - f32(atomicLoad(&redox_eaten[idx])) / REDOX_FP, 0.0);
    r *= exp(-max(params.redox_decay, 0.0) * dt);
    redox_out[idx] = clamp(r, 0.0, REDOX_MAX);
}

@compute @workgroup_size(16, 16)
fn copy_temperature(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    let idx = fgrid(gid.x, gid.y);
    temp_in[idx] = temp_out[idx];
    redox_in[idx] = redox_out[idx];
    atomicStore(&redox_eaten[idx], 0u);
}

// FLUTUAÇÃO: força = cima × BUOYANCY × desvio ao ambiente local, com uma
// pequena oscilação lateral. A força morre com a pluma quando esta arrefece.
@compute @workgroup_size(16, 16)
fn buoyancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x == 0u || y == 0u || x >= FLUID_SIZE - 1u || y >= FLUID_SIZE - 1u) { return; }
    let idx = fgrid(x, y);
    let dev = temp_in[idx] - temp_ambient_at(y);
    if (abs(dev) <= 1e-3) { return; }
    let time_bucket = params.epoch / 16u;
    let side = (rng_f4(idx, time_bucket, S_BUOYANCY).x * 2.0 - 1.0) * 0.25 * abs(dev) / max(abs(dev), 1.0);
    let f = vec2<f32>(side * TEMP_BUOYANCY * abs(dev), TEMP_BUOYANCY * dev);
    atomic_add_force(idx * 2u, f.x);
    atomic_add_force(idx * 2u + 1u, f.y);
}

// Junta as forças atómicas no buffer de forças (com limite).
@compute @workgroup_size(16, 16)
fn gather_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    let idx = fgrid(gid.x, gid.y);
    let f = vec2<f32>(force_at_fp(idx * 2u), force_at_fp(idx * 2u + 1u));
    fluid_forces[idx] = clamp_vec2_len(sanitize_vec2(f), MAX_FORCE);
}

@compute @workgroup_size(16, 16)
fn clear_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    fluid_forces[fgrid(gid.x, gid.y)] = vec2<f32>(0.0);
}

fn force_at(x: u32, y: u32) -> vec2<f32> {
    return sanitize_vec2(fluid_forces[fgrid(x, y)]);
}

@compute @workgroup_size(16, 16)
fn add_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    let dt = fluid_dt();
    let xm = select(x, x - 1u, x > 0u);
    let xp = select(x, x + 1u, x + 1u < FLUID_SIZE);
    let ym = select(y, y - 1u, y > 0u);
    let yp = select(y, y + 1u, y + 1u < FLUID_SIZE);
    // Desfoque isotrópico de 9 pontos das forças injetadas (evita vincos nos eixos).
    let f_c = force_at(x, y);
    let f_avg = (4.0 * (force_at(xm, y) + force_at(xp, y) + force_at(x, ym) + force_at(x, yp))
        + (force_at(xm, ym) + force_at(xp, ym) + force_at(xm, yp) + force_at(xp, yp)) + 20.0 * f_c) * (1.0 / 36.0);
    let f_user = clamp_vec2_len(mix(f_c, f_avg, FORCE_SMOOTH_MIX), MAX_FORCE);
    var v = sanitize_vec2(sanitize_vec2(velocity_in[idx]) + f_user * dt);
    v = slope_steer_velocity(x, y, v, dt);
    v *= rubble_drag(x, y, dt);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

// Viscosidade explícita (Laplaciano de 9 pontos): amortece o "ruído de TV".
@compute @workgroup_size(16, 16)
fn diffuse_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    if (x == 0u || x == FLUID_SIZE - 1u || y == 0u || y == FLUID_SIZE - 1u) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    let v_c = raw_velocity_cell(x, y);
    let axis_sum = raw_velocity_cell(x - 1u, y) + raw_velocity_cell(x + 1u, y)
        + raw_velocity_cell(x, y - 1u) + raw_velocity_cell(x, y + 1u);
    let diag_sum = raw_velocity_cell(x - 1u, y - 1u) + raw_velocity_cell(x + 1u, y - 1u)
        + raw_velocity_cell(x - 1u, y + 1u) + raw_velocity_cell(x + 1u, y + 1u);
    let lap = (4.0 * axis_sum + diag_sum - 20.0 * v_c) * (1.0 / 6.0);
    let a = max(params.fluid_viscosity, 0.0) * fluid_dt();
    var v = sanitize_vec2(v_c + a * lap);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

@compute @workgroup_size(16, 16)
fn advect_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    let dt = fluid_dt();
    let vel = reflect_if_into_solid(x, y, clamp_vec2_len(raw_velocity_cell(x, y), MAX_VEL));
    var trace_pos = pos - vel * dt;
    if (is_solid_at_pos(trace_pos)) {
        trace_pos = pos - reflect_if_into_solid(x, y, vel) * dt;
    }
    // Amortecimento independente do frame rate: decay por frame a 60 fps.
    let decay_factor = pow(clamp(params.fluid_decay, 0.0, 1.0), dt * 60.0);
    var out_v = sanitize_vec2(sample_velocity(trace_pos) * decay_factor);
    out_v = reflect_if_into_solid(x, y, out_v);
    velocity_out[idx] = clamp_vec2_len(out_v, MAX_VEL);
}

fn curl_at(x: u32, y: u32) -> f32 {
    let xm = u32(clamp(i32(x) - 1, 0, i32(FLUID_SIZE) - 1));
    let xp = u32(clamp(i32(x) + 1, 0, i32(FLUID_SIZE) - 1));
    let ym = u32(clamp(i32(y) - 1, 0, i32(FLUID_SIZE) - 1));
    let yp = u32(clamp(i32(y) + 1, 0, i32(FLUID_SIZE) - 1));
    let v_l = reflect_if_into_solid(xm, y, raw_velocity_cell(xm, y));
    let v_r = reflect_if_into_solid(xp, y, raw_velocity_cell(xp, y));
    let v_b = reflect_if_into_solid(x, ym, raw_velocity_cell(x, ym));
    let v_t = reflect_if_into_solid(x, yp, raw_velocity_cell(x, yp));
    return 0.5 * (v_r.y - v_l.y) - 0.5 * (v_t.x - v_b.x);
}

@compute @workgroup_size(16, 16)
fn vorticity_confinement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    let border = 2u;
    let epsilon = clamp(params.fluid_vorticity, 0.0, 10.0);
    if (x < border || x >= FLUID_SIZE - border || y < border || y >= FLUID_SIZE - border || epsilon < 1e-6) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    let w = curl_at(x, y);
    if (abs(w) < 5e-4) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    let grad = vec2<f32>(abs(curl_at(x + 1u, y)) - abs(curl_at(x - 1u, y)),
                         abs(curl_at(x, y + 1u)) - abs(curl_at(x, y - 1u))) * 0.5;
    let n = grad / max(length(grad), 1e-5);
    let f = vec2<f32>(n.y, -n.x) * (w * epsilon);
    // Limite por frame: o confinamento acrescenta caracóis, não energia.
    let dv = clamp_vec2_len(f * fluid_dt(), 2.0);
    var v = sanitize_vec2(velocity_in[idx] + dv);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

// Vizinho para a divergência. Se for parede (fora do domínio) ou sólido,
// usa uma célula-fantasma que ESPELHA a componente normal da célula atual:
// a velocidade na face da parede fica exatamente zero (impermeável). Assim
// a projeção "vê" as paredes; antes a divergência era forçada a zero nas
// bordas e uma deriva uniforme atravessava o teto sem ser corrigida.
fn div_neighbor(v_c: vec2<f32>, nx: i32, ny: i32, axis_x: bool) -> vec2<f32> {
    let outside = nx < 0 || ny < 0 || nx >= i32(FLUID_SIZE) || ny >= i32(FLUID_SIZE);
    if (outside || is_effectively_solid(u32(nx), u32(ny))) {
        return select(vec2<f32>(v_c.x, -v_c.y), vec2<f32>(-v_c.x, v_c.y), axis_x);
    }
    return raw_velocity_cell(u32(nx), u32(ny));
}

@compute @workgroup_size(16, 16)
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) {
        divergence[idx] = 0.0;
        return;
    }
    let v_c = raw_velocity_cell(x, y);
    let xi = i32(x);
    let yi = i32(y);
    let v_l = div_neighbor(v_c, xi - 1, yi, true);
    let v_r = div_neighbor(v_c, xi + 1, yi, true);
    let v_b = div_neighbor(v_c, xi, yi - 1, false);
    let v_t = div_neighbor(v_c, xi, yi + 1, false);
    divergence[idx] = 0.5 * ((v_r.x - v_l.x) + (v_t.y - v_b.y));
}

// JACOBI com tile em memória partilhada (16×16 + halo de 1).
// Fronteira de Neumann (dp/dn = 0) por espelhamento de p_c.
var<workgroup> jacobi_p_tile: array<f32, 324u>;
var<workgroup> jacobi_s_tile: array<u32, 324u>;

fn jacobi_load(tile_i: u32, gx: i32, gy: i32) {
    let ok = gx >= 0 && gy >= 0 && gx < i32(FLUID_SIZE) && gy < i32(FLUID_SIZE);
    if (ok) {
        let gxu = u32(gx);
        let gyu = u32(gy);
        jacobi_p_tile[tile_i] = pressure_in[fgrid(gxu, gyu)];
        jacobi_s_tile[tile_i] = select(0u, 1u, is_effectively_solid(gxu, gyu));
    } else {
        jacobi_p_tile[tile_i] = 0.0;
        jacobi_s_tile[tile_i] = 1u;
    }
}

@compute @workgroup_size(16, 16)
fn jacobi_pressure(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let stride = 18u;
    let lx = lid.x + 1u;
    let ly = lid.y + 1u;
    let tile_idx = ly * stride + lx;

    jacobi_load(tile_idx, x, y);
    if (lid.x == 0u) { jacobi_load(ly * stride, x - 1, y); }
    if (lid.x == 15u) { jacobi_load(ly * stride + 17u, x + 1, y); }
    if (lid.y == 0u) { jacobi_load(lx, x, y - 1); }
    if (lid.y == 15u) { jacobi_load(17u * stride + lx, x, y + 1); }
    workgroupBarrier();

    if (gid.x >= FLUID_SIZE || gid.y >= FLUID_SIZE) { return; }
    let idx = fgrid(gid.x, gid.y);
    if (jacobi_s_tile[tile_idx] != 0u) {
        pressure_out[idx] = 0.0;
        return;
    }
    let p_c = jacobi_p_tile[tile_idx];
    let p_l = select(jacobi_p_tile[tile_idx - 1u], p_c, jacobi_s_tile[tile_idx - 1u] != 0u);
    let p_r = select(jacobi_p_tile[tile_idx + 1u], p_c, jacobi_s_tile[tile_idx + 1u] != 0u);
    let p_b = select(jacobi_p_tile[tile_idx - stride], p_c, jacobi_s_tile[tile_idx - stride] != 0u);
    let p_t = select(jacobi_p_tile[tile_idx + stride], p_c, jacobi_s_tile[tile_idx + stride] != 0u);
    let div = divergence[idx];
    if (is_bad_f32(div) || is_bad_f32(p_c)) {
        pressure_out[idx] = 0.0;
        return;
    }
    let p_new = (p_l + p_r + p_b + p_t - div) * 0.25;
    pressure_out[idx] = select(p_new, 0.0, is_bad_f32(p_new));
}

@compute @workgroup_size(16, 16)
fn subtract_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    let p_c = pressure_in[idx];
    var p_l = p_c;
    if (x > 0u && !is_effectively_solid(x - 1u, y)) { p_l = pressure_in[fgrid(x - 1u, y)]; }
    var p_r = p_c;
    if (x + 1u < FLUID_SIZE && !is_effectively_solid(x + 1u, y)) { p_r = pressure_in[fgrid(x + 1u, y)]; }
    var p_b = p_c;
    if (y > 0u && !is_effectively_solid(x, y - 1u)) { p_b = pressure_in[fgrid(x, y - 1u)]; }
    var p_t = p_c;
    if (y + 1u < FLUID_SIZE && !is_effectively_solid(x, y + 1u)) { p_t = pressure_in[fgrid(x, y + 1u)]; }
    let grad = vec2<f32>(p_r - p_l, p_t - p_b) * 0.5;
    var v = sanitize_vec2(velocity_in[idx]) - grad;
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL);
}

// Paredes do aquário: impermeáveis e com deslizamento livre. A componente
// normal é ZERO na parede; a tangencial mantém-se. (O v3 invertia a normal
// a cada resolução, um "ressalto" que uma parede real não faz, e a borda
// oscilava. Mudado com o Filipe na fase 2.)
@compute @workgroup_size(16, 16)
fn enforce_boundaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let idx = fgrid(x, y);
    if (is_effectively_solid(x, y)) { velocity_out[idx] = vec2<f32>(0.0); return; }
    var v = sanitize_vec2(velocity_in[idx]);
    if (x == 0u || x == FLUID_SIZE - 1u) { v.x = 0.0; }
    if (y == 0u || y == FLUID_SIZE - 1u) { v.y = 0.0; }
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL);
}

// Suaviza a velocidade final (lida em velocity_in, a "a") para os agentes.
@compute @workgroup_size(16, 16)
fn smooth_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_SIZE || y >= FLUID_SIZE) { return; }
    let xm = select(x, x - 1u, x > 0u);
    let xp = min(x + 1u, FLUID_SIZE - 1u);
    let ym = select(y, y - 1u, y > 0u);
    let yp = min(y + 1u, FLUID_SIZE - 1u);
    let v = sanitize_vec2(velocity_in[fgrid(x, y)]) + sanitize_vec2(velocity_in[fgrid(xm, y)])
        + sanitize_vec2(velocity_in[fgrid(xp, y)]) + sanitize_vec2(velocity_in[fgrid(x, ym)])
        + sanitize_vec2(velocity_in[fgrid(x, yp)]);
    velocity_smooth[fgrid(x, y)] = v * 0.2;
}

// Velocidade suavizada num ponto do mundo (bilinear).
fn fluid_smooth_at_world(pos_world: vec2<f32>) -> vec2<f32> {
    let g = f32(FLUID_SIZE);
    let gx = clamp(pos_world.x / SIM_SIZE * g, 0.5, g - 0.5) - 0.5;
    let gy = clamp(pos_world.y / SIM_SIZE * g, 0.5, g - 0.5) - 0.5;
    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));
    let x1 = min(x0 + 1u, FLUID_SIZE - 1u);
    let y1 = min(y0 + 1u, FLUID_SIZE - 1u);
    let tx = fract(gx);
    let ty = fract(gy);
    let v0 = mix(velocity_smooth[fgrid(x0, y0)], velocity_smooth[fgrid(x1, y0)], tx);
    let v1 = mix(velocity_smooth[fgrid(x0, y1)], velocity_smooth[fgrid(x1, y1)], tx);
    return mix(v0, v1, ty);
}
