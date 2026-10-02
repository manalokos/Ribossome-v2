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

// Mobilidade das juntas (rad por passo por RT de binário) e limite por passo.
const JOINT_MOBILITY: f32 = 0.01;
const JOINT_MAX_STEP: f32 = 0.05;
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
fn rft_tangent(n: u32, k: u32, old: ptr<function, array<vec2<f32>, 64>>, cur: ptr<function, array<vec2<f32>, 64>>) -> vec2<f32> {
    let ka = select(k - 1u, k, k == 0u);
    let kb = select(k + 1u, k, k + 1u >= n);
    let a = 0.5 * ((*cur)[ka] + (*old)[ka]);
    let b = 0.5 * ((*cur)[kb] + (*old)[kb]);
    let t = b - a;
    let l = length(t);
    return select(vec2<f32>(1.0, 0.0), t / l, l > 1e-5);
}

// Devolve (Vx, Vy, Ω) no referencial do corpo, por passo.
fn rft_solve(slot: u32, n: u32, old: ptr<function, array<vec2<f32>, 64>>, cur: ptr<function, array<vec2<f32>, 64>>) -> vec3<f32> {
    // M·[Vx,Vy,Ω] = −c, com M = Σ Dᵀ·R·D e c = Σ Dᵀ·R·u, onde R é o tensor de
    // arrasto do resíduo e D mapeia (Vx,Vy,Ω) para a velocidade do resíduo.
    var m = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var c = vec3<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        // Regra do PONTO MÉDIO: geometria avaliada a meio do passo. Com a
        // geometria do fim do passo, o ruído das juntas era retificado numa
        // deriva espúria (∝ ruído²) que fazia "nadar" sem motor nenhum.
        let r = 0.5 * ((*cur)[k] + (*old)[k]);
        let u = (*cur)[k] - (*old)[k];
        let t = rft_tangent(n, k, old, cur);
        // R = ξ∥·t·tᵀ + ξ⊥·(I − t·tᵀ), com ξ∥ = 1.
        let tt = mat2x2<f32>(vec2<f32>(t.x * t.x, t.x * t.y), vec2<f32>(t.y * t.x, t.y * t.y));
        let id = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        // Arrasto ∝ comprimento do segmento (corpo esbelto).
        let rr = (tt + RFT_PERP_RATIO * (id - tt)) * (residue_len(slot, k) / SEGMENT_LEN);
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
        c += vec3<f32>(dot(d0, ru), dot(d1, ru), dot(d2, ru));
    }
    let det = determinant(m);
    if (abs(det) < 1e-6) { return vec3<f32>(0.0); }
    // Regra de Cramer.
    let b = -c;
    let mx = mat3x3<f32>(b, m[1], m[2]);
    let my = mat3x3<f32>(m[0], b, m[2]);
    let mz = mat3x3<f32>(m[0], m[1], b);
    return vec3<f32>(determinant(mx), determinant(my), determinant(mz)) / det;
}

// Um passo da dinâmica das juntas do agente `slot`. kT = agitação térmica local.
// Devolve (Vx, Vy, Ω, φ): o movimento rígido de natação no referencial do
// corpo e φ, a rotação do referencial guardado (preso ao 1.º segmento) face
// ao referencial alinhado em que o RFT resolve. a.rot avança Ω + φ.
fn joints_step(slot: u32, a: Agent, kt: f32) -> vec4<f32> {
    let n = a.body_len;
    if (n < 2u) { return vec4<f32>(0.0); }
    let base = slot * MAX_BODY;
    var old: array<vec2<f32>, 64>;
    for (var k = 0u; k < n; k++) { old[k] = body_pos[base + k]; }

    for (var k = 1u; k < n; k++) { // θ_0 é a orientação global (o corpo roda livre)
        let aa = body_get(slot, k);
        // Alvo: forma de repouso + a deformação ATIVA da junta (estado
        // catalítico, propagado N->C com atraso) + o desvio pelos sinais.
        let goal = joint_base[base + k] + joint_active[base + k] + signal_deflection(slot, k);
        let theta = joint_angle[base + k];
        let tau = -joint_stiffness(aa) * (theta - goal);
        // Ruído térmico (Langevin sobreamortecido): σ = √(2·μ·kT).
        let q = rng_f4(a.id, params.epoch, S_JOINT + k);
        let bm = sqrt(-2.0 * log(max(q.x, 1e-7))) * cos(6.2831853 * q.y);
        let dth = clamp(JOINT_MOBILITY * tau, -JOINT_MAX_STEP, JOINT_MAX_STEP)
            + bm * sqrt(2.0 * JOINT_MOBILITY * max(kt, 0.0));
        joint_angle[base + k] = theta + dth;
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
    if (params.rft_enabled == 0u) { return vec4<f32>(0.0); }
    // As posições guardadas estão num referencial preso ao 1.º segmento: cada
    // batida aparece lá como uma rotação RÍGIDA grande do resto do corpo, que
    // o RFT linearizado (Ω×r) só cancela até 1.ª ordem; o resto (∝ Ω²) dava
    // uma rotação espúria sempre no mesmo sentido. Por isso alinha-se primeiro
    // a forma nova à antiga (Procrustes 2D, à volta do centro de massa): o
    // RFT vê só a deformação verdadeira e a rotação do referencial (φ) é
    // contabilizada de forma exata.
    var cur: array<vec2<f32>, 64>;
    var sc = 0.0;
    var sd = 0.0;
    for (var k = 0u; k < n; k++) {
        let q = body_pos[base + k];
        let o = old[k];
        sd += dot(q, o);
        sc += q.x * o.y - q.y * o.x;
    }
    let phi = atan2(sc, sd); // roda a forma nova para a antiga
    let cp = cos(phi);
    let sp = sin(phi);
    for (var k = 0u; k < n; k++) {
        let q = body_pos[base + k];
        cur[k] = vec2<f32>(cp * q.x - sp * q.y, sp * q.x + cp * q.y);
    }
    let s = rft_solve(slot, n, &old, &cur);
    push_fluid(slot, a, n, &old, &cur, s);
    // Mundo = R(rot)·R(Ω)·R(φ)·forma guardada nova  =>  rot avança Ω + φ.
    return vec4<f32>(s, phi);
}

// OS AGENTES EMPURRAM A ÁGUA: cada resíduo devolve ao fluido a reação do seu
// arrasto, f = R·v (v = velocidade do resíduo na água: V + Ω×r + u). Para
// um nadador a soma é zero (força total nula, baixo Reynolds): a água recebe
// um DIPOLO, não impulso líquido. Acumula-se em force_vectors entre dois
// passos do fluido. (A deriva do próprio agente usa a água à sua volta, onde
// o seu dipolo é ~simétrico; não se desconta à parte.)
fn push_fluid(slot: u32, a: Agent, n: u32, old: ptr<function, array<vec2<f32>, 64>>, cur: ptr<function, array<vec2<f32>, 64>>, s: vec3<f32>) {
    if (params.fluid_enabled == 0u || params.agent_fluid_push == 0.0) { return; }
    let rot_mid = a.rot + 0.5 * s.z;
    let cr = cos(rot_mid);
    let sr = sin(rot_mid);
    let world_per_fluid = SIM_SIZE / f32(FLUID_SIZE);
    // Velocidade por passo (mundo) -> força do fluido (células do fluido / s²).
    let scale = params.agent_fluid_push / (world_per_fluid * max(params.dt, 1e-4) * max(params.dt, 1e-4));
    let id = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    for (var k = 0u; k < n; k++) {
        let r = 0.5 * ((*cur)[k] + (*old)[k]);
        let u = (*cur)[k] - (*old)[k];
        let t = rft_tangent(n, k, old, cur);
        let tt = mat2x2<f32>(vec2<f32>(t.x * t.x, t.x * t.y), vec2<f32>(t.y * t.x, t.y * t.y));
        let rr = (tt + RFT_PERP_RATIO * (id - tt)) * (residue_len(slot, k) / SEGMENT_LEN);
        let v = s.xy + s.z * vec2<f32>(-r.y, r.x) + u;
        let f_body = rr * v;
        let f = vec2<f32>(cr * f_body.x - sr * f_body.y, sr * f_body.x + cr * f_body.y) * scale;
        let rw = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * r.x - sr * r.y, sr * r.x + cr * r.y);
        let fi = fluid_index_at_world(rw);
        atomic_add_force(fi * 2u, f.x);
        atomic_add_force(fi * 2u + 1u, f.y);
    }
}
