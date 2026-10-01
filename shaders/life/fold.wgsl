// DOBRAGEM E JUNTAS (fase 5 + início da fase 7).
//
// A cadeia é descrita pelos ângulos de viragem θ_k em cada resíduo:
// direção do segmento k = Σ_{j<=k} θ_j; posições a SEGMENT_LEN de distância.
//
// Energia (em unidades de RT, as do MJ):
//   E = Σ_{|a−b|>=3} e_MJ(a,b)·s(d)            contactos (Miyazawa-Jernigan)
//     + Σ_{|a−b|>=2} K_REP·(R_EXCL − d)²       volume excluído (d < R_EXCL)
//     + Σ_k ½·K_k·(θ_k − θ0_k)²                tendência local de cada junta
// s(d) = 1 até R_CONTACT_IN e cai suavemente a 0 em R_CONTACT_OUT (a escala
// do contacto MJ, ~6,5 Å entre centros; 11 unidades = 3,8 Å entre Cα).
// K_k = params.chain_stiffness / flexibilidade² (Vihinen 1994).
//
// Dinâmica sobreamortecida: cada junta roda pelo binário das forças sobre o
// resto da cadeia (o gradiente exato da energia em relação a θ_k), com
// ruído térmico ∝ temperatura local.
//   - Primeiros FOLD_STEPS passos de vida: energia completa, θ0 = tendência
//     local (Chou-Fasman). No fim, a forma dobrada passa a ser θ_base.
//   - Depois: cada junta treme à volta de θ_base (só mola + ruído).

const FOLD_STEPS: u32 = 120u;
const R_CONTACT_IN: f32 = 13.0;
const R_CONTACT_OUT: f32 = 19.0;
const R_EXCL: f32 = 10.0;
const K_REP: f32 = 0.5;
// Mobilidade das juntas (rad por passo por RT de binário) e limite por passo.
const JOINT_MOBILITY: f32 = 0.01;
const JOINT_MAX_STEP: f32 = 0.05;
const S_JOINT: u32 = 5u << 16u;     // + índice da junta

fn mj(a: u32, b: u32) -> f32 {
    return AA_MJ[a * 20u + b];
}

fn joint_stiffness(aa: u32) -> f32 {
    return params.chain_stiffness / (AA_FLEX[aa] * AA_FLEX[aa]);
}

// Reconstrói as posições locais (centradas no centro de massa) a partir dos ângulos.
fn rebuild_body(slot: u32, n: u32) {
    var p = vec2<f32>(0.0);
    var ang = 0.0;
    var com = vec2<f32>(0.0);
    var mass = 0.0;
    for (var k = 0u; k < n; k++) {
        body_pos[slot * MAX_BODY + k] = p;
        let m = residue_mass(body_get(slot, k));
        com += p * m;
        mass += m;
        ang += joint_angle[slot * MAX_BODY + k];
        p += vec2<f32>(cos(ang), sin(ang)) * SEGMENT_LEN;
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
fn rft_solve(n: u32, old: ptr<function, array<vec2<f32>, 64>>, cur: ptr<function, array<vec2<f32>, 64>>) -> vec3<f32> {
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
        let rr = tt + RFT_PERP_RATIO * (id - tt);
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
    let folding = a.age < FOLD_STEPS;
    var old: array<vec2<f32>, 64>;
    for (var k = 0u; k < n; k++) { old[k] = body_pos[base + k]; }

    // Forças sobre cada resíduo (só durante a dobragem: O(n²)).
    var force: array<vec2<f32>, 64>;
    for (var k = 0u; k < n; k++) { force[k] = vec2<f32>(0.0); }
    if (folding) {
        for (var i = 0u; i < n; i++) {
            let ri = body_pos[base + i];
            let ai = body_get(slot, i);
            for (var j = i + 2u; j < n; j++) {
                let d = ri - body_pos[base + j];
                let dist = max(length(d), 1e-3);
                var du = 0.0; // dE/dd
                if (j >= i + 3u && dist < R_CONTACT_OUT) {
                    if (dist > R_CONTACT_IN) {
                        let t = (dist - R_CONTACT_IN) / (R_CONTACT_OUT - R_CONTACT_IN);
                        du += mj(ai, body_get(slot, j)) * (-0.5 * 3.14159265 * sin(3.14159265 * t)) / (R_CONTACT_OUT - R_CONTACT_IN);
                    }
                }
                if (dist < R_EXCL) {
                    du += -2.0 * K_REP * (R_EXCL - dist);
                }
                let f = -du * d / dist; // força sobre i
                force[i] += f;
                force[j] -= f;
            }
        }
    }

    // Binário em cada junta k sobre a parte distal (resíduos > k):
    // τ_k = Σ_{b>k} (r_b − r_k) × F_b, por somas de sufixo.
    var sum_f = vec2<f32>(0.0);
    var sum_rxf = 0.0;
    for (var kk = 0u; kk < n; kk++) {
        let k = n - 1u - kk;
        let rk = body_pos[base + k];
        let tau_contacts = sum_rxf - (rk.x * sum_f.y - rk.y * sum_f.x);
        let fk = force[k];
        sum_f += fk;
        sum_rxf += rk.x * fk.y - rk.y * fk.x;
        if (k == 0u) { continue; } // θ_0 é a orientação global (o corpo roda livre)

        let aa = body_get(slot, k);
        // Alvo: forma base (ou tendência local durante a dobragem) + a
        // deformação ATIVA da junta: o desvio do seu estado catalítico
        // (ligado +A, produto −A) mais a atividade da junta anterior no passo
        // anterior (propagação N->C com atraso: uma onda de atividade).
        let goal = select(joint_base[base + k], residue_bend(aa), folding) + joint_active[base + k]
            + signal_deflection(slot, k);
        let theta = joint_angle[base + k];
        let tau = tau_contacts - joint_stiffness(aa) * (theta - goal);
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
    // Fim da dobragem: a forma atual passa a ser a forma base.
    if (a.age + 1u == FOLD_STEPS) {
        for (var k = 0u; k < n; k++) { joint_base[base + k] = joint_angle[base + k]; }
    }
    rebuild_body(slot, n);
    // Durante a dobragem não se nada (a cadeia está a assentar).
    if (folding || params.rft_enabled == 0u) { return vec4<f32>(0.0); }
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
    let s = rft_solve(n, &old, &cur);
    // Mundo = R(rot)·R(Ω)·R(φ)·forma guardada nova  =>  rot avança Ω + φ.
    return vec4<f32>(s, phi);
}
