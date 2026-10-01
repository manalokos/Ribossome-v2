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
const JOINT_MOBILITY: f32 = 0.0005;
const JOINT_MAX_STEP: f32 = 0.05;
const S_JOINT: u32 = 5u << 16u;     // + índice da junta

fn mj(a: u32, b: u32) -> f32 {
    var m = AA_MJ;
    return m[a * 20u + b];
}

fn joint_stiffness(aa: u32) -> f32 {
    var f = AA_FLEX;
    return params.chain_stiffness / (f[aa] * f[aa]);
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

// Um passo da dinâmica das juntas do agente `slot`. kT = agitação térmica local.
fn joints_step(slot: u32, a: Agent, kt: f32) {
    let n = a.body_len;
    if (n < 2u) { return; }
    let base = slot * MAX_BODY;
    let folding = a.age < FOLD_STEPS;

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
        let goal = select(joint_base[base + k], residue_bend(aa), folding);
        let theta = joint_angle[base + k];
        let tau = tau_contacts - joint_stiffness(aa) * (theta - goal);
        // Ruído térmico (Langevin sobreamortecido): σ = √(2·μ·kT).
        let q = rng_f4(a.id, params.epoch, S_JOINT + k);
        let bm = sqrt(-2.0 * log(max(q.x, 1e-7))) * cos(6.2831853 * q.y);
        let dth = clamp(JOINT_MOBILITY * tau, -JOINT_MAX_STEP, JOINT_MAX_STEP)
            + bm * sqrt(2.0 * JOINT_MOBILITY * max(kt, 0.0));
        joint_angle[base + k] = theta + dth;
    }
    // Fim da dobragem: a forma atual passa a ser a forma base.
    if (a.age + 1u == FOLD_STEPS) {
        for (var k = 0u; k < n; k++) { joint_base[base + k] = joint_angle[base + k]; }
    }
    rebuild_body(slot, n);
}
