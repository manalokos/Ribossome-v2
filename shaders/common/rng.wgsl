// GERADOR ALEATÓRIO baseado em contador: PCG4D (Jarzynski & Olano, "Hash
// Functions for GPU Rendering", JCGT 2020 — dos melhores em qualidade e
// velocidade). Cada número sai de 4 entradas SEPARADAS:
//   (chave, passo, fluxo, semente)
// chave = célula/slot/agente; fluxo = qual o uso (constantes S_* abaixo,
// mais um sub-índice). Nada de misturar entradas com XOR à mão, o que
// criava correlações entre sequências que deviam ser independentes.
// Determinista: as mesmas entradas dão sempre o mesmo número.

fn pcg4d(v_in: vec4<u32>) -> vec4<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// Fluxos (o sub-índice soma-se ao fluxo base; os blocos de 2^16 não se sobrepõem).
const S_PHOTO: u32 = 1u;
const S_DECAY: u32 = 2u;
const S_THERMAL: u32 = 3u;
const S_BUOYANCY: u32 = 4u;
const S_RELAX: u32 = 16u;           // + fase
const S_SAND: u32 = 32u;            // + fase
const S_MOVE: u32 = 1u << 16u;      // + índice do monómero
const S_BLOCK: u32 = 2u << 16u;     // + destino (0..15)

// 4 números independentes de 32 bits para (chave, passo, fluxo).
fn rng_u4(key: u32, epoch: u32, stream: u32) -> vec4<u32> {
    return pcg4d(vec4<u32>(key, epoch, stream, params.seed));
}

// Os mesmos, como floats uniformes em [0, 1) (24 bits de mantissa).
fn rng_f4(key: u32, epoch: u32, stream: u32) -> vec4<f32> {
    return vec4<f32>(rng_u4(key, epoch, stream) >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

// Mistura determinista (desempates; não é uma fonte de aleatoriedade).
fn hash(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u);
    return x;
}
