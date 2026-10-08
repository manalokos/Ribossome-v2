// PISTA DE CORRIDAS (banco de ensaio da natação; params.track_mode != 0).
// Um circuito fechado à volta do centro do mundo: o eixo é uma curva em
// coordenadas polares, r(θ) = R0·(1 + onda·sin(lóbulos·θ)), que vira para os
// dois lados; a meia-largura também varia com θ. Tudo o que fica fora do
// corredor é rocha (o terreno é gerado em Rust com as MESMAS fórmulas:
// src/track.rs). Não há comida nem luz do sol: a única energia é AVANÇAR no
// sentido contrário ao dos ponteiros do relógio (θ a crescer), e tocar nas
// paredes custa energia. As paredes são "luminosas": a luz é só uma função
// da distância à parede mais próxima, para os sensores de luz as verem.
const TRACK_R0: f32 = 0.30;
const TRACK_WAVE: f32 = 0.14;
const TRACK_LOBES: f32 = 5.0;
const TRACK_HALF: f32 = 0.022;
const TRACK_HALF_WAVE: f32 = 0.3;
// Recuo (unidades do mundo) que não se paga: o vaivém de nadar.
const TRACK_SLACK: f32 = 150.0;
// Genoma de referência do custo de um filho (ver o emparelhamento).
const TRACK_REF_BASES: f32 = 33.0;

fn track_axis_r(theta: f32) -> f32 {
    return SIM_SIZE * TRACK_R0 * (1.0 + TRACK_WAVE * sin(TRACK_LOBES * theta));
}

fn track_half_width(theta: f32) -> f32 {
    return SIM_SIZE * TRACK_HALF * (1.0 + TRACK_HALF_WAVE * sin(2.0 * theta + 1.0));
}

// Distância (aproximada, medida ao longo do raio) à parede mais próxima:
// positiva dentro do corredor, negativa dentro da rocha.
fn track_wall_dist(p: vec2<f32>) -> f32 {
    let d = p - vec2<f32>(0.5 * SIM_SIZE);
    let theta = atan2(d.y, d.x);
    return track_half_width(theta) - abs(length(d) - track_axis_r(theta));
}

// Avanço ao longo da pista entre dois pontos, em unidades do mundo (arco
// sobre o eixo): positivo no sentido de θ a crescer, negativo a recuar.
fn track_advance(p0: vec2<f32>, p1: vec2<f32>) -> f32 {
    let c = vec2<f32>(0.5 * SIM_SIZE);
    let a = p0 - c;
    let b = p1 - c;
    // Ângulo de a para b, sem o salto de ±π do atan2 de cada um.
    let dtheta = atan2(a.x * b.y - a.y * b.x, dot(a, b));
    return dtheta * track_axis_r(atan2(b.y, b.x));
}

// Luz das paredes num ponto: 1 encostado à parede, a cair para o meio.
fn track_light(p: vec2<f32>) -> f32 {
    let d = p - vec2<f32>(0.5 * SIM_SIZE);
    let theta = atan2(d.y, d.x);
    let half = track_half_width(theta);
    let wd = half - abs(length(d) - track_axis_r(theta));
    // (também cai para dentro da rocha, para só a borda brilhar no desenho)
    return exp(-abs(wd) / (0.3 * half));
}
