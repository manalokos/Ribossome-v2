// JUNTAS DO CORPO.
//
// A cadeia é descrita pelos ângulos de viragem θ_k em cada resíduo:
// direção do segmento k = Σ_{j<=k} θ_j; o segmento k tem o comprimento do
// resíduo k (residue_len: aminoácido × órgão).
// A forma de repouso é o ângulo de repouso de cada aminoácido (como no v3):
// o corpo nasce já com ela, sem fase de dobragem. (A dobragem por contactos
// Miyazawa-Jernigan foi retirada: era O(n²) por agente nos primeiros
// passos de vida e, num modelo 2D com um segmento por resíduo, pouco
// acrescentava à forma.)
//
// Dinâmica sobreamortecida: cada junta é uma mola K_k (= chain_stiffness /
// flexibilidade², Vihinen 1994) para o alvo = repouso + motor catalítico +
// desvio pelos sinais α/β, com ruído térmico ∝ temperatura local.

// Mobilidade das juntas (rad por passo por RT de binário) e limite por passo
// (só proteção numérica: com 0.05 o limite atrasava as juntas e era esse
// atraso artificial que fazia nadar os relógios lentos).
const JOINT_MOBILITY: f32 = 0.01;
const JOINT_MAX_STEP: f32 = 0.2;
const S_JOINT: u32 = 5u << 16u;     // + índice da junta

fn joint_stiffness(aa: u32) -> f32 {
    let f = aa_props[aa].flex;
    return params.chain_stiffness / (f * f);
}

// Reconstrói as posições locais (centradas no centro de massa) a partir dos ângulos.
fn rebuild_body(slot: u32, n: u32) {
    var p = vec2<f32>(0.0);
    var ang = 0.0;
    var com = vec2<f32>(0.0);
    var mass = 0.0;
    for (var k = 0u; k < n; k++) {
        body_pos[slot * MAX_BODY + k] = p;
        let m = residue_mass(slot, k);
        com += p * m;
        mass += m;
        ang += joint_angle[slot * MAX_BODY + k];
        p += vec2<f32>(cos(ang), sin(ang)) * residue_len(slot, k);
    }
    com /= max(mass, 1e-6);
    for (var k = 0u; k < n; k++) {
        body_pos[slot * MAX_BODY + k] -= com;
    }
}

// NATAÇÃO POR FORÇAS RESISTIVAS (RFT), baixo Reynolds.
// Cada resíduo que se move sente arrasto anisotrópico: ξ⊥ = 2·ξ∥ em
// relação à tangente da cadeia (corpo esbelto). O corpo move-se como um
// todo (V, Ω) de modo a que a força e o binário totais sejam ZERO (sem
// inércia). A velocidade de cada resíduo no mundo é V + Ω×r + u, onde u é a
// mudança de forma. É um sistema linear 3×3 em (Vx, Vy, Ω). Um movimento
// recíproco não desloca nada (teorema da vieira).
const RFT_PERP_RATIO: f32 = 2.0;

// Tangente da cadeia no resíduo k, a MEIO do passo (média das posições
// antigas e novas alinhadas).
// Contexto do RFT de um agente. As posições NOVAS (alinhadas), a água e o
// arrasto de cada resíduo calculam-se na hora a partir da memória global:
// tabelas locais de 64 posições por agente (eram 4) não cabem nos registos
// e iam para memória lenta (o RFT era ~75% do custo dos agentes).
struct RftCtx {
    base: u32,
    n: u32,
    cp: f32,  // cos/sin de φ (forma nova -> alinhada)
    sp: f32,
    cr: f32,  // cos/sin da orientação (corpo -> mundo)
    sr: f32,
    pos: vec2<f32>,
    soft: u32,
}

// Multiplicador de arrasto do resíduo k (órgãos volumosos arrastam mais).
fn residue_drag_mult(slot: u32, k: u32) -> f32 {
    let o = organ_get(slot, k);
    return select(1.0, max(organ_cost(o).drag_mult, 0.05), o != 0u);
}

// Posição nova (alinhada) do resíduo k.
fn cur_at(c: RftCtx, k: u32) -> vec2<f32> {
    let q = body_pos[c.base + k];
    return vec2<f32>(c.cp * q.x - c.sp * q.y, c.sp * q.x + c.cp * q.y);
}

// Água (referencial do corpo, por passo, já ÷ arrasto) e arrasto do resíduo
// k, no ponto médio do passo.
fn env_at(c: RftCtx, old: ptr<function, array<vec2<f32>, 64>>, k: u32) -> vec3<f32> {
    let r = 0.5 * (cur_at(c, k) + (*old)[k]);
    let rw = c.pos + vec2<f32>(c.cr * r.x - c.sr * r.y, c.sr * r.x + c.cr * r.y);
    let drag = anchor_drag(rw);
    var uf = vec2<f32>(0.0);
    if (params.fluid_enabled != 0u) {
        let w = water_at(rw, c.soft != 0u);
        // Dois arrastos: o da água (relativo à água) e o dos GRÃOS do
        // entulho (relativo a zero: os grãos estão parados). Somados dão
        // arrasto total `drag` com velocidade de referência água / drag:
        // um resíduo bem preso fica parado mesmo com a água dos poros a
        // mexer (antes seguia-a e vibrava com ela).
        let wb = vec2<f32>(c.cr * w.x + c.sr * w.y, -c.sr * w.x + c.cr * w.y); // mundo -> corpo
        uf = wb / drag;
    }
    return vec3<f32>(uf, drag);
}

fn rft_tangent(c: RftCtx, k: u32, old: ptr<function, array<vec2<f32>, 64>>) -> vec2<f32> {
    let n = c.n;
    let ka = select(k - 1u, k, k == 0u);
    let kb = select(k + 1u, k, k + 1u >= n);
    let a = 0.5 * (cur_at(c, ka) + (*old)[ka]);
    let b = 0.5 * (cur_at(c, kb) + (*old)[kb]);
    let t = b - a;
    let l = length(t);
    return select(vec2<f32>(1.0, 0.0), t / l, l > 1e-5);
}

// Resultado do RFT (referencial alinhado do corpo, por passo): NATAÇÃO (o
// movimento rígido que a mudança de forma provoca) e TRANSPORTE pela água
// (o que a corrente nos resíduos provoca). Ambos com rotação.
struct RftOut {
    swim: vec3<f32>,
    flow: vec3<f32>,
}

// Arrasto extra de um resíduo dentro de terreno: o entulho prende (e junto
// à rocha ainda mais). CONTÍNUO: usa os grãos interpolados na posição exata
// do resíduo (com valores por célula, o arrasto saltava de 1 para 31 ao
// cruzar uma fronteira e os agentes no entulho vibravam).
const RUBBLE_ANCHOR_PER_GRAIN: f32 = 10.0;
const ROCK_ANCHOR: f32 = 20.0;

fn grains_at(pw: vec2<f32>) -> f32 {
    let w = f32(WORLD_UNITS_PER_CELL);
    let g = pw / w - 0.5;
    let c0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    var v = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var q = 0u; q < 4u; q++) {
        let c = clamp(c0 + vec2<i32>(i32(q & 1u), i32(q >> 1u)), vec2<i32>(0), vec2<i32>(i32(GRID_SIZE) - 1));
        v[q] = f32(min(gamma_count(u32(c.y) * GRID_SIZE + u32(c.x)), 4u));
    }
    return mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
}

fn anchor_drag(world_pos: vec2<f32>) -> f32 {
    let g = grains_at(world_pos);
    let rock = max(g - 2.0, 0.0);
    return 1.0 + RUBBLE_ANCHOR_PER_GRAIN * g + ROCK_ANCHOR * rock * rock;
}

// Força-livre e binário-livre: M·x = −Σ Dᵀ·R·(u − u_f), com M = Σ Dᵀ·R·D.
// A parte de u (forma) é a natação; a de u_f (água) é o transporte.
fn rft_solve(slot: u32, c: RftCtx, old: ptr<function, array<vec2<f32>, 64>>) -> RftOut {
    let n = c.n;
    var m = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var cs = vec3<f32>(0.0);
    var cf = vec3<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        // Regra do PONTO MÉDIO: geometria avaliada a meio do passo. Com a
        // geometria do fim do passo, o ruído das juntas era retificado numa
        // deriva espúria (∝ ruído²) que fazia "nadar" sem motor nenhum.
        let ck = cur_at(c, k);
        let r = 0.5 * (ck + (*old)[k]);
        let u = ck - (*old)[k];
        let t = rft_tangent(c, k, old);
        let env = env_at(c, old, k);
        // R = ξ∥·t·tᵀ + ξ⊥·(I − t·tᵀ), com ξ∥ = 1.
        let tt = mat2x2<f32>(vec2<f32>(t.x * t.x, t.x * t.y), vec2<f32>(t.y * t.x, t.y * t.y));
        let id = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        // Arrasto ∝ comprimento do segmento (corpo esbelto): o da água é
        // anisotrópico (⊥ = 2 × ∥, o que permite nadar); o dos GRÃOS do
        // entulho é igual em todas as direções (env.z − 1), por isso dilui a
        // anisotropia e a propulsão perde eficiência no sedimento.
        let lw = residue_len(slot, k) / SEGMENT_LEN * residue_drag_mult(slot, k);
        let rr = (tt + RFT_PERP_RATIO * (id - tt) + (env.z - 1.0) * id) * lw;
        // Colunas de D: ∂v/∂Vx = (1,0), ∂v/∂Vy = (0,1), ∂v/∂Ω = (−r.y, r.x).
        let d0 = vec2<f32>(1.0, 0.0);
        let d1 = vec2<f32>(0.0, 1.0);
        let d2 = vec2<f32>(-r.y, r.x);
        let rd0 = rr * d0;
        let rd1 = rr * d1;
        let rd2 = rr * d2;
        m[0] += vec3<f32>(dot(d0, rd0), dot(d1, rd0), dot(d2, rd0));
        m[1] += vec3<f32>(dot(d0, rd1), dot(d1, rd1), dot(d2, rd1));
        m[2] += vec3<f32>(dot(d0, rd2), dot(d1, rd2), dot(d2, rd2));
        let ru = rr * u;
        cs += vec3<f32>(dot(d0, ru), dot(d1, ru), dot(d2, ru));
        let rf = rr * env.xy;
        cf += vec3<f32>(dot(d0, rf), dot(d1, rf), dot(d2, rf));
    }
    var out: RftOut;
    out.swim = vec3<f32>(0.0);
    out.flow = vec3<f32>(0.0);
    let det = determinant(m);
    if (abs(det) < 1e-6) { return out; }
    // Regra de Cramer, para os dois lados direitos.
    let bs = -cs;
    out.swim = vec3<f32>(determinant(mat3x3<f32>(bs, m[1], m[2])), determinant(mat3x3<f32>(m[0], bs, m[2])), determinant(mat3x3<f32>(m[0], m[1], bs))) / det;
    let bf = cf;
    out.flow = vec3<f32>(determinant(mat3x3<f32>(bf, m[1], m[2])), determinant(mat3x3<f32>(m[0], bf, m[2])), determinant(mat3x3<f32>(m[0], m[1], bf))) / det;
    return out;
}

// Velocidade da água num ponto do mundo, em unidades do mundo por passo,
// SUAVIZADA (média de 5 pontos a ±1 célula do fluido) para que o dipolo do
// próprio agente pese pouco na água que o leva.
fn water_at(p: vec2<f32>, soft: bool) -> vec2<f32> {
    let h = SIM_SIZE / f32(FLUID_SIZE);
    // Suavizada: o campo pré-calculado em smooth_velocity (uma leitura).
    let v = select(fluid_velocity_at_world(p), fluid_smooth_at_world(p), soft);
    return v * h * max(params.dt, 0.0);
}

// Saída de joints_step: natação e transporte (referencial alinhado, por
// passo) e φ, a rotação do referencial guardado face ao alinhado.
struct JointsOut {
    swim: vec3<f32>,
    flow: vec3<f32>,
    phi: f32,
    // Grãos de entulho empurrados pelas juntas neste passo (custam energia).
    pushed: u32,
}

// Um passo da dinâmica das juntas do agente `slot`. kT = agitação térmica local.
// Devolve a natação e o transporte pela água (movimentos rígidos no
// referencial do corpo) e φ, a rotação do referencial guardado (preso ao 1.º
// segmento) face ao alinhado em que o RFT resolve. a.rot avança Ω + φ.
fn joints_step(slot: u32, a: Agent, kt: f32) -> JointsOut {
    var none: JointsOut;
    none.swim = vec3<f32>(0.0);
    none.flow = vec3<f32>(0.0);
    none.phi = 0.0;
    none.pushed = 0u;
    let n = a.body_len;
    if (n < 2u) { return none; }
    let base = slot * MAX_BODY;
    var old: array<vec2<f32>, 64>;
    for (var k = 0u; k < n; k++) { old[k] = body_pos[base + k]; }

    let cr0 = cos(a.rot);
    let sr0 = sin(a.rot);
    var pushed = 0u;
    for (var k = 1u; k < n; k++) { // θ_0 é a orientação global (o corpo roda livre)
        let aa = body_get(slot, k);
        // No sedimento a junta dobra mais devagar: tem de empurrar os grãos.
        // Mobilidade ÷ √arrasto (com ÷ arrasto inteiro, ~1/50, a junta ficava
        // praticamente imóvel e o agente preso para sempre).
        let ok = old[k];
        let pw_k = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr0 * ok.x - sr0 * ok.y, sr0 * ok.x + cr0 * ok.y);
        let drag_here = anchor_drag(pw_k);
        // Alvo: forma de repouso + a deformação ATIVA da junta (estado
        // catalítico, propagado N->C com atraso) + o desvio pelos sinais.
        let goal = joint_base[base + k] + joint_active[base + k] + signal_deflection(slot, k);
        let theta = joint_angle[base + k];
        let tau = -joint_stiffness(aa) * (theta - goal);
        // Ruído térmico (Langevin sobreamortecido): σ = √(2·μ·kT).
        let q = rng_f4(a.id, params.epoch, S_JOINT + k);
        let bm = sqrt(-2.0 * log(max(q.x, 1e-7))) * cos(6.2831853 * q.y);
        let dth = clamp(JOINT_MOBILITY * tau / sqrt(drag_here), -JOINT_MAX_STEP, JOINT_MAX_STEP)
            + bm * sqrt(2.0 * JOINT_MOBILITY * max(kt, 0.0));
        joint_angle[base + k] = theta + dth;

        // ESCAVAR: o segmento seguinte, ao dobrar, empurra o grão de entulho
        // onde está para a célula vizinha (o sedimento cede e abre uma
        // galeria). A probabilidade segue o deslocamento que a junta TENTA
        // (o binário, sem a resistência dos grãos), não o que consegue:
        // assim um agente preso também cava. Os grãos conservam-se.
        if (params.bioturbation > 0.0 && k + 1u < n) {
            let seg = old[k + 1u] - ok;
            let want = clamp(JOINT_MOBILITY * tau, -JOINT_MAX_STEP, JOINT_MAX_STEP) * length(seg);
            let p_push = clamp(params.bioturbation * abs(want) / f32(WORLD_UNITS_PER_CELL), 0.0, 1.0);
            if (q.z < p_push && a.energy > params.bioturbation_cost * f32(pushed + 1u)) {
                // Velocidade do segmento k+1 ao rodar à volta da junta k: ⊥ seg.
                let vb = sign(want) * vec2<f32>(-seg.y, seg.x);
                let vw = vec2<f32>(cr0 * vb.x - sr0 * vb.y, sr0 * vb.x + cr0 * vb.y);
                let ob = old[k + 1u];
                let src = world_to_cell(vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr0 * ob.x - sr0 * ob.y, sr0 * ob.x + cr0 * ob.y));
                let g = gamma_count(src);
                if (g > 0u && g < GAMMA_SOLID_THRESHOLD) {
                    var d = vec2<i32>(select(-1, 1, vw.x > 0.0), 0);
                    if (abs(vw.y) > abs(vw.x)) { d = vec2<i32>(0, select(-1, 1, vw.y > 0.0)); }
                    let c = vec2<i32>(i32(src % GRID_SIZE), i32(src / GRID_SIZE)) + d;
                    if (all(c >= vec2<i32>(0)) && all(c < vec2<i32>(i32(GRID_SIZE)))) {
                        let dst = u32(c.y) * GRID_SIZE + u32(c.x);
                        if (gamma_count(dst) + 1u < GAMMA_SOLID_THRESHOLD && gamma_move_one(src, dst)) {
                            pushed += 1u;
                        }
                    }
                }
            }
        }
    }
    // ACOPLAMENTO ATIVO (para o passo seguinte): a atividade da junta k é o
    // seu próprio motor mais uma fração da atividade da junta k−1 AGORA, que
    // só chega à k no passo seguinte. Propaga-se só a deformação paga pela
    // hidrólise; o ruído térmico não (senão o calor faria nadar: 2.ª lei).
    // A direção N->C é a polaridade da cadeia polipeptídica.
    var prev_active = 0.0;
    for (var k = 0u; k < n; k++) {
        let st = joint_state[base + k];
        let motor = select(select(0.0, -params.motor_amplitude, st == 2u), params.motor_amplitude, st == 1u);
        let here = joint_active[base + k];
        joint_active[base + k] = motor + params.joint_coupling * prev_active;
        prev_active = here;
    }
    rebuild_body(slot, n);
    if (params.rft_enabled == 0u && params.fluid_swim_only == 0u) {
        none.pushed = pushed;
        return none;
    }
    // As posições guardadas estão num referencial preso ao 1.º segmento: cada
    // batida aparece lá como uma rotação RÍGIDA grande do resto do corpo, que
    // o RFT linearizado (Ω×r) só cancela até 1.ª ordem; o resto (∝ Ω²) dava
    // uma rotação espúria sempre no mesmo sentido. Por isso alinha-se primeiro
    // a forma nova à antiga (Procrustes 2D, à volta do centro de massa): o
    // RFT vê só a deformação verdadeira e a rotação do referencial (φ) é
    // contabilizada de forma exata.
    var sc = 0.0;
    var sd = 0.0;
    for (var k = 0u; k < n; k++) {
        let q = body_pos[base + k];
        let o = old[k];
        sd += dot(q, o);
        sc += q.x * o.y - q.y * o.x;
    }
    let phi = atan2(sc, sd); // roda a forma nova para a antiga
    var c: RftCtx;
    c.base = base;
    c.n = n;
    c.cp = cos(phi);
    c.sp = sin(phi);
    c.cr = cos(a.rot);
    c.sr = sin(a.rot);
    c.pos = vec2<f32>(a.pos_x, a.pos_y);
    c.soft = select(0u, 1u, params.fluid_swim_only == 0u);
    var out = rft_solve(slot, c, &old);
    if (params.fluid_swim_only != 0u || params.rft_enabled == 0u) {
        // EXPERIÊNCIA "só pelo fluido": a forma empurra a água, que leva o
        // corpo (o transporte); sem a natação do RFT.
        out.swim = vec3<f32>(0.0);
    }
    push_fluid(slot, a, c, &old, out.swim + out.flow);
    var res: JointsOut;
    res.swim = out.swim;
    res.flow = out.flow;
    // Mundo = R(rot)·R(Ω)·R(φ)·forma guardada nova  =>  rot avança Ω + φ.
    res.phi = phi;
    res.pushed = pushed;
    return res;
}

// OS AGENTES EMPURRAM A ÁGUA: cada resíduo devolve ao fluido a reação do seu
// arrasto, f = R·v (v = velocidade do resíduo na água: V + Ω×r + u − u_água). Para
// um nadador a soma é zero (força total nula, baixo Reynolds): a água recebe
// um DIPOLO, não impulso líquido. Acumula-se em force_vectors entre dois
// passos do fluido. (A deriva do próprio agente usa a água à sua volta, onde
// o seu dipolo é ~simétrico; não se desconta à parte.)
fn push_fluid(slot: u32, a: Agent, c: RftCtx, old: ptr<function, array<vec2<f32>, 64>>, s: vec3<f32>) {
    if (params.fluid_enabled == 0u || (params.agent_fluid_push == 0.0 && params.fluid_swim_only == 0u)) { return; }
    let rot_mid = a.rot + 0.5 * s.z;
    let cr = cos(rot_mid);
    let sr = sin(rot_mid);
    let world_per_fluid = SIM_SIZE / f32(FLUID_SIZE);
    // Velocidade por passo (mundo) -> força do fluido (células do fluido / s²).
    // (Na experiência "só pelo fluido" o empurrão é sempre 1.)
    let push = select(params.agent_fluid_push, 1.0, params.fluid_swim_only != 0u);
    let scale = push / (world_per_fluid * max(params.dt, 1e-4) * max(params.dt, 1e-4));
    let id = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let n = c.n;
    for (var k = 0u; k < n; k++) {
        let ck = cur_at(c, k);
        let r = 0.5 * (ck + (*old)[k]);
        let u = ck - (*old)[k];
        let t = rft_tangent(c, k, old);
        let env = env_at(c, old, k);
        let tt = mat2x2<f32>(vec2<f32>(t.x * t.x, t.x * t.y), vec2<f32>(t.y * t.x, t.y * t.y));
        // Só o arrasto com a ÁGUA a empurra (o dos grãos fica no entulho);
        // a água real no resíduo é uf × drag (ver joints_step).
        let rr = (tt + RFT_PERP_RATIO * (id - tt)) * (residue_len(slot, k) / SEGMENT_LEN * residue_drag_mult(slot, k));
        // Velocidade do resíduo RELATIVA à água (quem só é levado não empurra).
        let v = s.xy + s.z * vec2<f32>(-r.y, r.x) + u - env.xy * env.z;
        let f_body = rr * v;
        let f = vec2<f32>(cr * f_body.x - sr * f_body.y, sr * f_body.x + cr * f_body.y) * scale;
        let rw = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * r.x - sr * r.y, sr * r.x + cr * r.y);
        let fi = fluid_index_at_world(rw);
        atomic_add_force(fi * 2u, f.x);
        atomic_add_force(fi * 2u + 1u, f.y);
    }
}
