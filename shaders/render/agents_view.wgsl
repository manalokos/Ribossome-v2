// Agentes: DUAS instâncias por resíduo. Primeiro um TUBO (cápsula do
// resíduo k até ao k+1, com a cor da classe química do v3 e sombreado de
// cilindro); depois, só nos órgãos, a forma do órgão por cima (SDF no
// fragmento, orientada pela cadeia):
//   boca        disco com uma abertura escura virada para fora;
//   músculo     elipse ao longo da cadeia, com estrias;
//   sensores    TOTAIS: coroa de 6 antenas; DIRECIONAIS: 2 antenas, uma de
//               cada lado da cadeia (comida verde, luz amarela);
//   energia     disco com anel interior;
//   relógio     mostrador com um ponteiro que roda com o período;
//   relé        losango;
//   armazenamento disco grande com anéis.
// Vista de sinais (view.signal_view): a cor passa a ser o sinal α e/ou β.
// Instâncias sem resíduo viram triângulos degenerados. RNA nu = disco cinzento.

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> agents_view: array<Agent>;
@group(0) @binding(2) var<storage, read> bodies_view: array<u32>;
@group(0) @binding(3) var<storage, read> body_pos_view: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> draw_list_view: array<u32>;
@group(0) @binding(5) var<storage, read> organs_view: array<u32>;
@group(0) @binding(6) var<storage, read> signals_view: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> aa_props_view: array<AaProps, 20>;
@group(0) @binding(8) var<storage, read> genomes_view: array<u32>;
@group(0) @binding(9) var<storage, read> rna_tail_view: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> organ_variants_view: array<OrganVariant>;
// Ligações entre agentes (shaders/life/bonds.wgsl): 5 vec4<u32> por slot.
@group(0) @binding(11) var<storage, read> bonds_view: array<vec4<u32>>;
// Semelhança genética de cada slot com o selecionado (−1 = sem dados).
@group(0) @binding(12) var<storage, read> kin_view: array<f32>;
// Mordidas do último passo por agente: .x = energia que lhe tiraram, .z = a
// que ganhou a morder (shaders/life/contact.wgsl).
@group(0) @binding(13) var<storage, read> bite_view: array<vec4<f32>>;
// SPRITES (aspeto "microscopia eletrónica"): atlas cinzento + máscara
// (assets/sprites.png, feito por scripts/sprites_atlas.py a partir de
// imagens geradas). SPRITE_COLS variantes por linha; linhas 0..21 = tipo de
// órgão, depois o troço de aminoácido, um espigão e o corpo da protease.
// A cor é a do órgão (ou a da classe do aminoácido).
@group(0) @binding(14) var sprites_tex: texture_2d<f32>;
@group(0) @binding(15) var sprites_samp: sampler;
// SOMBRAS DE CONTACTO (como no microscópio eletrónico: sem direção, um
// escurecimento à volta de cada peça sobre o que está por baixo). Cada peça
// desenha uma auréola preta e transparente fora da sua máscara; o quadrado
// leva uma margem para ela caber. A ordem do desenho (tubos, depois órgãos)
// decide quem escurece quem.
const SHADOW: f32 = 0.5;
const SHADOW_MARGIN: f32 = 1.3;
const TUBE_SHADOW: f32 = 1.5;
// Espessura do troço por baixo de uma protease, em fração da normal.
const PROTEASE_STALK: f32 = 0.4;
// Auréola à volta de um disco de raio 1 (x = distância ao centro).
fn halo(x: f32) -> f32 {
    return SHADOW * (1.0 - smoothstep(0.8, SHADOW_MARGIN, x));
}
const SPRITE_COLS: f32 = 9.0;
const SPRITE_COLS_U: u32 = 9u;
const SPRITE_ROWS: f32 = 32.0;
// Aminoácidos: um troço por tipo (i na linha AMINO + i/9, coluna i%9), com
// a cápsula no meio do mosaico e AMINO_MARGIN de folga para as deformações
// (BODY_MARGIN no script do atlas).
const SPRITE_ROW_AMINO: f32 = 29.0;
const AMINO_MARGIN: f32 = 1.3;
const SPRITE_ROW_SPIKE: f32 = 23.0;
const SPRITE_ROW_PROTEASE: f32 = 24.0;
// Raio da esfera nos sprites dos sensores de um lado, em fração de meio
// mosaico (STALK_BODY no script do atlas).
const SPRITE_STALK_BODY: f32 = 0.3;
// (luminância, máscara) do mosaico (linha, coluna) em q (-1..1, y para
// cima); qx, qy são as derivadas de q no ecrã (escolhem o mipmap sem
// depender do ramo em que se está).
// ALTURA da peça neste fragmento (unidades do mundo), para o relevo do
// microscópio 3D: cada sítio que lê um sprite deixa-a aqui (a altura do
// atlas vezes o tamanho da peça).
var<private> g_h: f32 = 0.0;
// ...e quanto esse ponto está levantado em relação ao meio do corpo.
var<private> g_lift: f32 = 0.0;
// Cota do meio dos corpos acima do chão, no microscópio 3D.
const AGENT_Z: f32 = 11.0;
// Altura dos sensores em relação à de uma bola do mesmo contorno.
const SENSOR_FLAT: f32 = 0.45;
// Altura dos troços do corpo em relação à forma insuflada do seu sprite.
const AMINO_RELIEF: f32 = 1.8;
// ...e a altura do CENTRO da peça acima do chão: o volume vai de g_c - g_h
// a g_c + g_h (a mesma forma para cima e para baixo: uma cúpula fica uma
// bola, meio tubo um tubo).
var<private> g_c: f32 = 0.0;
// Devolve também (em .z) a altura da forma insuflada, em meios mosaicos.
fn sprite(row: f32, col: f32, q: vec2<f32>, qx: vec2<f32>, qy: vec2<f32>) -> vec3<f32> {
    let sc = vec2<f32>(0.5 / SPRITE_COLS, -0.5 / SPRITE_ROWS);
    let qc = clamp(q, vec2<f32>(-0.98), vec2<f32>(0.98));
    let s = textureSampleGrad(sprites_tex, sprites_samp, vec2<f32>((col + 0.5) / SPRITE_COLS, (row + 0.5) / SPRITE_ROWS) + qc * sc, qx * sc, qy * sc);
    let inside = step(max(abs(q.x), abs(q.y)), 0.995);
    return vec3<f32>(s.r, s.g * inside, s.b);
}
// Máscara DESFOCADA do mosaico (um nível baixo do mipmap, escolhido à mão):
// serve de sombra de contacto com a forma do sprite.
const SPRITE_SOFT_LEVEL: f32 = 3.4;
fn sprite_soft(row: f32, col: f32, q: vec2<f32>) -> f32 {
    let sc = vec2<f32>(0.5 / SPRITE_COLS, -0.5 / SPRITE_ROWS);
    let qc = clamp(q, vec2<f32>(-0.94), vec2<f32>(0.94));
    let far = max(abs(q.x), abs(q.y));
    return textureSampleLevel(sprites_tex, sprites_samp, vec2<f32>((col + 0.5) / SPRITE_COLS, (row + 0.5) / SPRITE_ROWS) + qc * sc, SPRITE_SOFT_LEVEL).g * (1.0 - smoothstep(0.94, 1.15, far));
}
// Cinzento do microscópio pintado com a cor do órgão, com as arestas claras.
fn sem_color(tint: vec3<f32>, lum: f32) -> vec3<f32> {
    let l = pow(clamp(lum, 0.0, 1.0), 0.7);
    return clamp(tint * (0.25 + 1.1 * l) + vec3<f32>(0.35) * pow(l, 4.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

// FLASH DAS PROTEASES: quem está a atacar fica com as proteases ligadas
// amarelo-claras (só a cor, a forma é a do momento); a vítima fica vermelha.
const BITE_ATTACK_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.45);
const BITE_VICTIM_COLOR: vec3<f32> = vec3<f32>(1.0, 0.12, 0.08);

// Fios de RNA nas pontas (as zonas não traduzidas): bases desenhadas por
// agente (metade para cada ponta), distância entre bases e ondulação.
const RNA_PER_END: u32 = 32u;
const RNA_SPACING: f32 = 5.0;
const RNA_RADIUS: f32 = 1.6;
const RNA_WIGGLE: f32 = 0.35;

fn base_color(b: u32) -> vec3<f32> {
    switch b {
        case 0u: { return vec3<f32>(1.0, 0.25, 0.2); }   // A
        case 1u: { return vec3<f32>(1.0, 0.85, 0.2); }   // U
        case 2u: { return vec3<f32>(0.25, 0.9, 0.3); }   // G
        default: { return vec3<f32>(0.3, 0.55, 1.0); }   // C
    }
}

fn genome_base(slot: u32, i: u32) -> u32 {
    return (genomes_view[slot * 16u + i / 16u] >> ((i % 16u) * 2u)) & 3u;
}

const MAX_BODY_V: u32 = 64u;
// Ligações por agente e vec4 por slot (iguais a MAX_BONDS / BOND_STRIDE).
const BONDS_V: u32 = 4u;
const BOND_STRIDE_V: u32 = 5u;
// Instâncias por agente: tubos, órgãos, bases de RNA, ligações e a bola do
// parentesco.
const AGENT_INSTANCES: u32 = 3u * MAX_BODY_V + BONDS_V + 1u;
// NÍVEIS DE DETALHE (view.lod), para o aspeto de um agente não mudar com o
// zoom: cada nível é a MÉDIA do anterior, em vez de engordar as coisas
// pequenas até um mínimo de píxeis (os órgãos engordavam mais do que o corpo
// e tapavam-se uns aos outros pela ordem de desenho: a cor mudava com o zoom).
//   0 = detalhe (tubos, órgãos desenhados, fios de RNA, ligações);
//   1 = um troço liso por resíduo, com a cor que ele tem de perto;
//   2 = um troço por cada 4 resíduos, com a média das cores dos 4.
// Em 1 e 2 cada agente leva os seus troços e a bola de marcação
// (65 e 17 instâncias: os valores em drawlist.wgsl).
// ÓRGÃOS REFORÇADOS: na média, um resíduo com órgão pesa LOD_ORGAN_WEIGHT
// (ao perto o órgão é um disco bem maior do que o tubo); assim as manchas
// de cor que distinguem os tipos de agente continuam a ver-se de longe.
const LOD_ORGAN_WEIGHT: f32 = 4.0;
// E o seu troço é mais grosso (ao perto o disco do órgão tem ~3× o raio do
// tubo), com um mínimo de píxeis um pouco maior.
const LOD_ORGAN_THICK: f32 = 2.2;
// Raio da bola do parentesco, em píxeis do ecrã (igual para todos).
const KIN_DOT_PX: f32 = 5.0;
const NO_ORGAN: u32 = 0xFFu;
// Tamanho de um órgão em relação a um resíduo estrutural.
const ORGAN_SCALE: f32 = 1.15;
const TUBE_FAT: f32 = 1.5;

struct AgentVsOut {
    @builtin(position) pos: vec4<f32>,
    // 0 = tubo (segmento do resíduo k até ao k+1), 1 = órgão por cima.
    @location(5) @interpolate(flat) mode: u32,
    // Coordenadas no quadrado (−1..1), eixos do mundo. Interpoladas POR
    // AMOSTRA: o shader de fragmentos corre uma vez por amostra (4 por
    // píxel) e a forma de um órgão pequeno fica bem recortada.
    @location(0) @interpolate(perspective, sample) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    // Tipo do órgão (NO_ORGAN se não houver).
    @location(2) @interpolate(flat) organ: u32,
    // Tangente da cadeia (mundo, unitária).
    @location(3) @interpolate(flat) tangent: vec2<f32>,
    // Raio do disco do resíduo em fração do quadrado; ângulo do relógio.
    @location(4) @interpolate(flat) core_phase: vec2<f32>,
    // Coluna do atlas de sprites (variante do aspeto); bit 8 = tubo com textura.
    @location(6) @interpolate(flat) sprite: u32,
    // Meio lado do quadrado de um órgão, em unidades do mundo (para a altura).
    @location(7) @interpolate(flat) size: f32,
    // Quanto este resíduo está levantado do fundo (ondulação do corpo).
    @location(8) @interpolate(flat) lift: f32,
};

// Classes (v3): alifáticos A I L M V, aromáticos F W Y, polares S T N Q,
// C à parte, positivos K R H, negativos D E, G e P especiais.
// TOM PELA ENERGIA (vista química): energia por resíduo. Com pouca o agente
// escurece (ate 0,65, para continuar a ver-se); a partir de 1 por residuo vai
// clareando ate 1,25 (com 4 por residuo).
fn energy_tone(a: Agent) -> f32 {
    let e = a.energy / max(f32(a.body_len), 1.0);
    return mix(0.65, 1.0, clamp(e, 0.0, 1.0)) + 0.25 * clamp((e - 1.0) / 3.0, 0.0, 1.0);
}

// TOM DA ESPÉCIE: cada agente tem um ângulo de cor tirado do seu genoma
// (species_hue em drawlist.wgsl; o mesmo para as duas formas da linhagem e
// quase igual entre parentes próximos). Aqui passa a uma cor suave.
@group(0) @binding(18) var<storage, read> tint_view: array<f32>;
fn hue_color(h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.0, 2.0943951, 4.1887902);
    return vec3<f32>(0.70) + 0.26 * cos(vec3<f32>(h) - k);
}
// Cor de um TROÇO do corpo: a da classe do aminoácido puxada (AMINO_MUTE)
// para o tom da espécie, para as cores dos órgãos sobressaírem e cada
// espécie ter a sua cor.
const AMINO_MUTE: f32 = 0.65;
fn amino_color(aa: u32, hue: f32) -> vec3<f32> {
    return mix(class_color(aa), hue_color(hue), AMINO_MUTE);
}

fn class_color(aa: u32) -> vec3<f32> {
    switch aa {
        case 0u, 7u, 9u, 10u, 17u: { return vec3<f32>(0.72, 0.72, 0.62); } // A I L M V
        case 4u, 18u, 19u: { return vec3<f32>(0.70, 0.45, 0.95); }        // F W Y
        case 15u, 16u, 11u, 13u: { return vec3<f32>(0.40, 0.85, 0.45); }  // S T N Q
        case 1u: { return vec3<f32>(0.95, 0.90, 0.30); }                  // C
        case 8u, 14u, 6u: { return vec3<f32>(0.35, 0.55, 1.00); }         // K R H
        case 2u, 3u: { return vec3<f32>(1.00, 0.35, 0.30); }              // D E
        case 5u: { return vec3<f32>(0.95, 0.95, 0.95); }                  // G
        default: { return vec3<f32>(1.00, 0.60, 0.20); }                  // P
    }
}

// ENGROSSAR POR IGUAL: quando um tubo típico (DETAIL_REF_R de raio) ficaria
// com menos de DETAIL_MIN_PX no ecrã, o corpo INTEIRO (tubos e órgãos)
// engrossa pelo mesmo fator. Antes cada coisa tinha o seu mínimo de píxeis
// (os órgãos mais do que os tubos) e as proporções, e com elas a cor do
// agente, mudavam com o zoom.
const DETAIL_REF_R: f32 = 3.0;
const DETAIL_MIN_PX: f32 = 0.3;
// Tamanho de um resíduo no ecrã (píxeis) abaixo do qual o desenho passa aos
// troços lisos (LOD_MID_PX em render/mod.rs) e comprimento de um resíduo.
const LOD_SWITCH_PX: f32 = 0.35;
const RESIDUE_UNITS: f32 = 11.0;
fn detail_fat() -> f32 {
    return max(1.0, DETAIL_MIN_PX / (DETAIL_REF_R * view.zoom));
}

// Cor de um órgão visto de longe: a cor dominante do seu desenho de perto.
fn organ_lod_color(organ: u32, oc: u32, base: vec3<f32>) -> vec3<f32> {
    let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
    let v0 = organ_variants_view[organ * ORGAN_VARIANTS + p].p0;
    switch organ {
        case ORGAN_MOUTH: { return mix(base, vec3<f32>(1.0), 0.3); }
        case ORGAN_MUSCLE: { return mix(base, vec3<f32>(0.85, 0.25, 0.25), 0.55); }
        case ORGAN_FOOD_SENSOR, ORGAN_FOOD_SENSOR_DIR: { return vec3<f32>(0.45, 1.0, 0.45); }
        case ORGAN_LIGHT_SENSOR, ORGAN_LIGHT_SENSOR_DIR: { return vec3<f32>(1.0, 0.95, 0.4); }
        case ORGAN_ENERGY_SENSOR: { return vec3<f32>(1.0, 0.85, 0.2); }
        case ORGAN_CLOCK: { return vec3<f32>(0.85, 0.9, 1.0); }
        case ORGAN_RELAY: { return vec3<f32>(0.6, 0.9, 1.0); }
        case ORGAN_PHOTOSYSTEM: { return vec3<f32>(0.35, 0.95, 0.35); }
        case ORGAN_PROTEASE: { return vec3<f32>(0.9, 0.2, 0.2); }
        case ORGAN_ANCHOR: { return select(vec3<f32>(0.25, 0.5, 1.0), vec3<f32>(0.15, 0.9, 0.8), v0 >= 0.0); }
        case ORGAN_BIAS, ORGAN_AGE_BIAS: { return select(vec3<f32>(1.0, 0.55, 0.15), vec3<f32>(0.35, 0.95, 0.35), v0 >= 0.5); }
        case ORGAN_CHEMO: { return vec3<f32>(0.9, 0.78, 0.15); }
        case ORGAN_DORMANCY: { return vec3<f32>(0.72, 0.82, 1.0); }
        case ORGAN_INHIBITOR: { return vec3<f32>(0.95, 0.8, 0.9); }
        case ORGAN_HOLDFAST: { return vec3<f32>(0.85, 0.6, 0.3); }
        case ORGAN_CHIRAL: { return vec3<f32>(0.95, 0.35, 0.85); }
        default: { return base * 0.8; }
    }
}

// Cor (rgb) e peso (a) do resíduo q nos níveis de detalhe 1 e 2.
fn lod_residue_color(slot: u32, q: u32) -> vec4<f32> {
    let aa = (bodies_view[slot * 16u + q / 4u] >> ((q % 4u) * 8u)) & 0xFFu;
    // (De longe também: os troços sem órgão levam o tom da espécie.)
    var col = amino_color(aa, tint_view[slot]);
    var wgt = 1.0;
    let oc = (organs_view[slot * 32u + q / 2u] >> ((q % 2u) * 16u)) & 0xFFFFu;
    if (oc != 0u) {
        col = organ_lod_color((oc & 0x1Fu) - 1u, oc, class_color(aa));
        wgt = LOD_ORGAN_WEIGHT;
    }
    // Nas vistas de sinais a cor é o sinal do resíduo (sem reforço).
    let sg = signals_view[slot * MAX_BODY_V + q];
    switch view.signal_view {
        case 1u: { col = signed_color(sg.x, vec3<f32>(1.0, 0.45, 0.1), vec3<f32>(0.1, 0.6, 1.0)); wgt = 1.0; }
        case 2u: { col = signed_color(sg.y, vec3<f32>(0.3, 1.0, 0.3), vec3<f32>(0.95, 0.3, 0.9)); wgt = 1.0; }
        case 3u: { col = vec3<f32>(0.5 + 0.5 * tanh(sg.x), 0.5 + 0.5 * tanh(sg.y), 0.35); wgt = 1.0; }
        case 5u: { col = vec3<f32>(0.5 + 0.5 * tanh(sg.z), 0.5 + 0.5 * tanh(sg.w), 0.35); wgt = 1.0; }
        default: {}
    }
    return vec4<f32>(col, wgt);
}

// Cor de um sinal com sinal: positivo -> `pos`, negativo -> `neg`, zero -> cinzento escuro.
fn signed_color(v: f32, pos: vec3<f32>, neg: vec3<f32>) -> vec3<f32> {
    let t = tanh(abs(v));
    return mix(vec3<f32>(0.18), select(neg, pos, v >= 0.0), t);
}

// Centro da câmara. Com um agente em foco (imagem do inspetor) é a posição
// ATUAL desse agente mais um desvio fixo: a posição que o CPU conhece vem de
// uma leitura com um ou dois frames de atraso, e um agente pequeno levado
// depressa pela corrente já tinha saído do enquadramento.
fn cam_center() -> vec2<f32> {
    var c = vec2<f32>(view.center_x, view.center_y);
    // (No microscópio 3D o agente em foco desenha-se no SEU sítio, com a
    // câmara da zona: é a camada que diz que peças são dele.)
    if (view.focus_slot != 0xFFFFFFFFu && view.relief == 0u) {
        let f = agents_view[view.focus_slot];
        c = vec2<f32>(f.pos_x + view.focus_dx, f.pos_y + view.focus_dy);
    }
    return c;
}

// Deposito: proporcao do oval (comprimento / largura) ate a meia largura
// STORAGE_MAX_HALF_WIDTH (unidades do mundo); dai para cima a largura so cresce
// com a raiz do comprimento, para uma fila comprida ficar um fuso gordo e nao
// uma bola que tapa o corpo.
const STORAGE_ASPECT: f32 = 1.3;
const STORAGE_MAX_HALF_WIDTH: f32 = 14.0;
fn is_storage(slot: u32, k: u32) -> bool {
    let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
    return oc != 0u && (oc & 0x1Fu) - 1u == ORGAN_STORAGE;
}
// Tamanho do quadrado em múltiplos do raio do disco, por tipo de órgão.
fn organ_extent(t: u32) -> f32 {
    switch t {
        // (Com sprites o mosaico enche o quadrado: isto é o raio do desenho
        // em raios do órgão. Nos sensores a esfera fica com ~1,2.)
        case ORGAN_FOOD_SENSOR: { return 2.0; }
        case ORGAN_LIGHT_SENSOR: { return 1.6; }
        case ORGAN_FOOD_SENSOR_DIR, ORGAN_LIGHT_SENSOR_DIR: { return 1.2 / SPRITE_STALK_BODY; }
        case ORGAN_MUSCLE: { return 1.6; }
        case ORGAN_STORAGE: { return 1.9; }
        case ORGAN_PHOTOSYSTEM: { return 1.45; }
        case ORGAN_HOLDFAST: { return 1.4; }
        case ORGAN_PROTEASE: { return 2.8; }
        case ORGAN_ANCHOR: { return 1.4; }
        case ORGAN_BIAS, ORGAN_AGE_BIAS: { return 1.1; }
        case NO_ORGAN: { return 1.0; }
        default: { return 1.25; }
    }
}

// RESTOS (só para o desenho): quando um agente morre À VISTA da câmara,
// guarda-se aqui uma cópia do corpo; o desenho mostra as peças a separarem-se
// e a flutuar durante uns 60 frames (vs_ghost em agents_view.wgsl). Não é
// matéria: a do morto já voltou à sopa. Palavra 0 = contador (anel de
// GHOST_MAX registos de GHOST_WORDS palavras, a partir de GHOST_HEAD):
//   0 pos_x, 1 pos_y, 2 rot (bits de f32), 3 epoch da morte, 4 resíduos,
//   5 id; 8.. corpo (16 palavras); 24.. órgãos (32); 56.. posições (128).
const GHOST_MAX: u32 = 2048u;
const GHOST_WORDS: u32 = 192u;
const GHOST_HEAD: u32 = 16u;
@group(0) @binding(16) var<storage, read> ghosts_view: array<u32>;
// Velocidade da água (grelha do fluido), para os restos irem na corrente.
@group(0) @binding(17) var<storage, read> velocity_view: array<vec2<f32>>;
// Passo de tempo da simulação (params.dt por omissão): a água leva uma peça
// velocidade × GHOST_DT por passo, como leva um agente.
const GHOST_DT: f32 = 0.017;
fn water_at(w: vec2<f32>) -> vec2<f32> {
    let f = clamp(floor(w / SIM_SIZE * f32(FLUID_SIZE)), vec2<f32>(0.0), vec2<f32>(f32(FLUID_SIZE - 1u)));
    let v = velocity_view[u32(f.y) * FLUID_SIZE + u32(f.x)];
    // A grelha guarda CÉLULAS DO FLUIDO por segundo: passa a unidades do mundo
    // (como water_at em fold.wgsl). Um valor estragado na grelha não pode
    // atirar a peça para o infinito.
    return select(vec2<f32>(0.0), clamp(v, vec2<f32>(-50.0), vec2<f32>(50.0)), v == v) * (SIM_SIZE / f32(FLUID_SIZE));
}
// Instâncias por registo: 64 tubos e 64 órgãos (GHOST_INSTANCES em render/mod.rs).
const GHOST_INSTANCES: u32 = 2u * MAX_BODY_V;

fn ghost_f(i: u32) -> f32 {
    return bitcast<f32>(ghosts_view[i]);
}

@vertex
fn vs_ghost(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let glyph = (inst % GHOST_INSTANCES) >= MAX_BODY_V;
    let k = inst % MAX_BODY_V;
    let g = GHOST_HEAD + (inst / GHOST_INSTANCES) * GHOST_WORDS;
    let n = min(ghosts_view[g + 4u], MAX_BODY_V);
    let died = ghosts_view[g + 3u];
    let age = f32(view.epoch - died);
    if (view.ghost_steps <= 0.0 || view.lod != 0u || view.focus_slot != 0xFFFFFFFFu || k >= n || view.epoch < died || age >= view.ghost_steps) {
        return o;
    }
    let t = age / view.ghost_steps;
    let aa = (ghosts_view[g + 8u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
    let oc = (ghosts_view[g + 24u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
    var organ = NO_ORGAN;
    if (oc != 0u) { organ = (oc & 0x1Fu) - 1u; }
    if (glyph && (organ == NO_ORGAN || organ == ORGAN_LINKER)) { return o; }
    let rot = ghost_f(g + 2u);
    let cr = cos(rot);
    let sr = sin(rot);
    let origin = vec2<f32>(ghost_f(g), ghost_f(g + 1u));
    let lp = vec2<f32>(ghost_f(g + 56u + 2u * k), ghost_f(g + 57u + 2u * k));
    let k1 = min(k + 1u, n - 1u);
    let lp1 = vec2<f32>(ghost_f(g + 56u + 2u * k1), ghost_f(g + 57u + 2u * k1));
    // Cada peça solta-se, roda um pouco e é LEVADA PELA CORRENTE; não encolhe:
    // desaparece de repente, cada uma na sua altura (ao acaso, entre 35% e
    // 100% da duração). Na corrente: segue a água desde o sítio onde estava (em
    // quatro troços, com a velocidade de agora em cada ponto), mais um
    // pequeno desvio próprio (um hash do agente e do resíduo) para as peças
    // se separarem mesmo em água parada.
    var h = (ghosts_view[g + 5u] * 64u + k) * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    let ang = f32(h & 0xFFFu) * (6.2831853 / 4096.0);
    let ease = 1.0 - (1.0 - t) * (1.0 - t);
    var off = vec2<f32>(cos(ang), sin(ang)) * (5.0 + 14.0 * f32((h >> 12u) & 0xFFu) / 255.0) * ease;
    let p_rest = origin + vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
    for (var i = 0; i < 4; i++) {
        off += water_at(p_rest + off) * (0.25 * age * GHOST_DT);
    }
    let spin = (f32((h >> 20u) & 0xFFu) / 255.0 - 0.5) * 3.0 * t;
    let fade = 1.0;
    if (t > 0.35 + 0.65 * f32((h >> 4u) & 0xFFu) / 255.0) { return o; }
    let cs = cos(spin);
    let sn = sin(spin);
    let seg_l = lp1 - lp;
    var seg = vec2<f32>(cr * seg_l.x - sr * seg_l.y, sr * seg_l.x + cr * seg_l.y);
    seg = vec2<f32>(seg.x * cs - seg.y * sn, seg.x * sn + seg.y * cs);
    let pk = origin + vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y) + off;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let r_base = TUBE_FAT * (0.9 + 3.0 * pow(aa_props_view[aa].volume / 130.0, 1.4));
    let tone = mix(1.0, 0.75, t);
    let l = length(seg);
    let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
    var w = pk;
    if (!glyph) {
        let r_tube = select(r_base, 0.9, organ == ORGAN_LINKER) * select(1.0, PROTEASE_STALK, organ == ORGAN_PROTEASE) * fade * detail_fat();
        let nn = vec2<f32>(-e.y, e.x);
        let rq = r_tube * TUBE_SHADOW;
        w = pk + e * select(-rq, l + rq, c.x > 0.0) + nn * (c.y * rq);
        o.mode = 0u;
        o.local = w - pk;
        o.tangent = seg;
        o.core_phase = vec2<f32>(r_tube, 0.0);
        o.organ = organ;
        o.sprite = select(0u, 0x100u | aa, organ != ORGAN_LINKER);
        o.color = amino_color(aa, ghost_f(g + 6u)) * tone;
    } else {
        var phase = 0.0;
        if (organ == ORGAN_PROTEASE) { phase = 8.99; }
        if (organ == ORGAN_STORAGE) { phase = 1.5; }
        if (organ == ORGAN_FOOD_SENSOR_DIR || organ == ORGAN_LIGHT_SENSOR_DIR) { phase = 1.0; }
        var ext = organ_extent(organ);
        if (organ == ORGAN_STORAGE) { ext = phase * SHADOW_MARGIN; }
        if (organ != ORGAN_STORAGE && organ != ORGAN_PROTEASE) { ext *= SHADOW_MARGIN; }
        w = pk + c * (r_base * ORGAN_SCALE * fade * ext * detail_fat());
        o.size = r_base * ORGAN_SCALE * fade * ext * detail_fat();
        o.mode = 1u;
        o.local = c;
        o.tangent = e;
        o.core_phase = vec2<f32>(1.0 / ext, phase);
        o.organ = organ;
        o.sprite = ((oc >> 5u) * 7u + organ) % SPRITE_COLS_U;
        o.color = organ_lod_color(organ, oc, class_color(aa)) * tone;
    }
    let px = (w - cam_center()) * view.zoom;
    // Abaixo de todos os agentes vivos (ver agent_z); os órgãos por cima dos tubos.
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), select(0.010, 0.011, glyph), 1.0);
    return o;
}

// ALTURA DE CADA PEÇA (profundidade): a lista de desenho sai da GPU por uma
// ordem que muda de frame para frame, e dois agentes sobrepostos trocavam de
// lugar (cintilavam). Cada agente tem uma altura fixa (um hash do slot) e,
// dentro dele, os órgãos ficam por cima dos tubos e cada troço por cima do
// anterior; o teste de profundidade decide quem tapa quem, e as sombras só
// caem sobre o que está mais baixo. camada: 0 = fios, ligações e troços de
// longe; 1 + k = tubo k; 80 + k = órgão k.
fn agent_z(slot: u32, layer: u32) -> f32 {
    var x = slot * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return 0.02 + 0.96 * (f32(x & 0xFFFu) + f32(layer) / 256.0) / 4096.0;
}

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> AgentVsOut {
    var o = agent_vertex(vi, inst);
    // (Os vértices postos fora do ecrã, com z = 2, ficam como estão.)
    if (o.pos.z < 1.5) {
        let far = view.lod != 0u;
        let per = select(AGENT_INSTANCES, MAX_BODY_V / select(4u, 1u, view.lod == 1u) + 1u, far);
        var slot = view.focus_slot;
        if (slot == 0xFFFFFFFFu) { slot = draw_list_view[inst / per]; }
        let local_i = inst % per;
        var layer = 0u;
        if (!far && local_i < MAX_BODY_V) { layer = 1u + local_i; }
        if (!far && local_i >= MAX_BODY_V && local_i < 2u * MAX_BODY_V) { layer = 80u + local_i - MAX_BODY_V; }
        o.pos.z = agent_z(slot, layer);
    }
    return o;
}

// ONDULAÇÃO VERTICAL (só no relevo do microscópio 3D): cada resíduo fica um
// pouco levantado do fundo, numa onda ao longo do corpo que avança com a
// idade do agente, como uma fita a nadar perto do substrato.
const BODY_LIFT: f32 = 9.0;
fn body_lift(a: Agent, k: u32) -> f32 {
    return BODY_LIFT * (0.5 + 0.5 * sin(f32(k) * 0.38 + f32(a.age) * 0.03 + f32(a.id % 97u)));
}
// Os ÓRGÃOS ficam desencontrados em altura (um em baixo, o seguinte mais
// acima, o outro mais ainda), para dois órgãos seguidos não se atravessarem.
// Só para CIMA: quando um descia, um órgão grande ficava enterrado no chão.
// (Pouco: com mais, os órgãos grandes e achatados ficavam a voar por cima do
// corpo, e via-se o vazio por baixo deles. Quem tapa quem já é decidido pela
// altura real, por isso dois órgãos seguidos fundem-se em vez de se cortarem.)
const ORGAN_STAGGER: f32 = 0.08;
fn organ_stagger(k: u32, radius: f32) -> f32 {
    return f32(k % 3u) * ORGAN_STAGGER * radius;
}

fn agent_vertex(vi: u32, inst: u32) -> AgentVsOut {
    var o: AgentVsOut;
    // Instâncias por agente: 0..63 tubos, 64..127 órgãos (por cima),
    // 128..191 bases de RNA não traduzidas nas pontas.
    let far = view.lod != 0u;
    // Resíduos por troço e troços por agente nos níveis 1 e 2.
    let stride = select(4u, 1u, view.lod == 1u);
    let tubes = MAX_BODY_V / stride;
    let per = select(AGENT_INSTANCES, tubes + 1u, far);
    // Com um agente em foco (imagem do inspetor) desenha-se SÓ esse, com
    // as suas instâncias (o draw não percorre a lista dos vivos).
    var slot = view.focus_slot;
    if (slot == 0xFFFFFFFFu) { slot = draw_list_view[inst / per]; }
    var local_i = inst % per;
    let a = agents_view[slot];
    if (far) {
        if (local_i == tubes) { return kin_vertex(vi, slot, a); }
        if (a.body_len > 0u) {
            // Um troço liso do resíduo k0 ao k1, com a média pesada das
            // cores dos seus resíduos e a espessura média deles.
            var gone: AgentVsOut;
            gone.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
            let k0 = local_i * stride;
            if (a.alive == 0u || k0 >= a.body_len) { return gone; }
            let k1 = min(k0 + stride, a.body_len - 1u);
            var sum = vec4<f32>(0.0);
            var thick = 0.0;
            var cnt = 0.0;
            var organ_w = 0.0;
            var armed = false;
            for (var q = k0; q < min(k0 + stride, a.body_len); q++) {
                let c = lod_residue_color(slot, q);
                sum += vec4<f32>(c.rgb * c.a, c.a);
                let aq = (bodies_view[slot * 16u + q / 4u] >> ((q % 4u) * 8u)) & 0xFFu;
                let oq = (organs_view[slot * 32u + q / 2u] >> ((q % 2u) * 16u)) & 0xFFFFu;
                let has_organ = oq != 0u;
                if (has_organ && (oq & 0x1Fu) - 1u == ORGAN_PROTEASE) { armed = true; }
                thick += (0.9 + 3.0 * pow(aa_props_view[aq].volume / 130.0, 1.4)) * select(1.0, LOD_ORGAN_THICK, has_organ);
                organ_w += select(0.0, c.a, has_organ);
                cnt += 1.0;
            }
            var colr = sum.rgb / max(sum.a, 1e-6);
            // Pouca energia = mais escuro, e a vítima de uma protease a
            // vermelho, como no desenho de perto.
            if (view.signal_view == 0u || view.signal_view == 4u) {
                colr *= energy_tone(a);
            }
            // O mínimo de espessura cresce aos poucos a partir do limiar
            // (onde o desenho detalhado acaba com tubos finos) até 1 píxel
            // de raio (1,5 com órgãos) a metade desse zoom: sem salto de
            // espessura na passagem e sem desaparecer ao longe.
            let px = view.zoom * RESIDUE_UNITS;
            let grow = clamp((LOD_SWITCH_PX - px) / (0.5 * LOD_SWITCH_PX), 0.0, 1.0);
            let floor_px = grow * (1.0 + 0.5 * organ_w / max(sum.a, 1e-6));
            var r_lod = max(thick / max(cnt, 1.0) * detail_fat(), floor_px / view.zoom);
            // MORDIDAS (como de perto): a vítima a vermelho; o troço com a
            // protease que ataca a amarelo e só um pouco mais grosso. Ao
            // longe é uma mudança de cor do próprio agente, não uma bola
            // por cima (com mínimos de píxeis os clarões tapavam o mapa).
            let bite = bite_view[slot];
            if (view.signal_view == 0u) {
                if (bite.x > 0.0) { colr = mix(colr, BITE_VICTIM_COLOR, 0.8); }
                if (bite.z > 0.0 && armed) {
                    colr = BITE_ATTACK_COLOR;
                }
            }
            return capsule_vertex(vi, residue_world_v(slot, a, k0), residue_world_v(slot, a, k1), r_lod, colr);
        }
        // RNA nu: o ponto do costume (só a instância 0 desenha).
    }
    if (local_i == 3u * MAX_BODY_V + BONDS_V) {
        return kin_vertex(vi, slot, a);
    }
    if (local_i >= 3u * MAX_BODY_V) {
        return bond_vertex(vi, slot, a, local_i - 3u * MAX_BODY_V);
    }
    if (local_i >= 2u * MAX_BODY_V) {
        return rna_vertex(vi, slot, a, local_i - 2u * MAX_BODY_V);
    }
    let glyph = local_i >= MAX_BODY_V;
    let k = local_i % MAX_BODY_V;
    let naked = a.body_len == 0u;
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot;
    if (a.alive == 0u || hidden || (k >= a.body_len && !(naked && k == 0u))) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    var centre = vec2<f32>(a.pos_x, a.pos_y);
    var r_world = 6.0;
    var col = vec3<f32>(0.55, 0.55, 0.55);
    var organ = NO_ORGAN;
    var tangent = vec2<f32>(1.0, 0.0);
    var phase = 0.0;
    var skip_glyph = false;
    var sprite_col = 0u;
    let bite = bite_view[slot];
    var flash = vec3<f32>(-1.0);
    if (bite.x > 0.0) { flash = BITE_VICTIM_COLOR; }
    // Comprimento dos espigões de uma protease de alcance (unidades do mundo).
    var spike = 0.0;
    if (!naked) {
        let base = slot * MAX_BODY_V;
        let aa = (bodies_view[slot * 16u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
        let lp = body_pos_view[base + k];
        let cr = cos(a.rot);
        let sr = sin(a.rot);
        centre += vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
        // Tangente da cadeia (vizinhos), rodada para o mundo.
        let ka = select(k - 1u, k, k == 0u);
        let kb = select(k + 1u, k, k + 1u >= a.body_len);
        let tl = body_pos_view[base + kb] - body_pos_view[base + ka];
        let tn = select(vec2<f32>(1.0, 0.0), tl / length(tl), length(tl) > 1e-5);
        tangent = vec2<f32>(cr * tn.x - sr * tn.y, sr * tn.x + cr * tn.y);
        // Espessura (v3): 4·√(volume/130), mais folga para os discos se tocarem.
        // Estruturais finos (a cadeia); os órgãos destacam-se (ORGAN_SCALE).
        // (Contraste forte de propósito: o volume é a capacidade de energia
        // do resíduo; glicina ~2, alanina ~2,6, médio ~4, triptofano ~7,5.)
        // (TUBE_FAT: tubos 1,5x mais grossos para os bichos se verem melhor;
        // ORGAN_SCALE desceu na mesma proporção, os órgãos ficam do mesmo tamanho.)
        r_world = TUBE_FAT * (0.9 + 3.0 * pow(aa_props_view[aa].volume / 130.0, 1.4));
        col = class_color(aa);
        // (O tubo, na vista química, leva a cor esbatida; o órgão troca-a mais abaixo.)
        if (!glyph && (view.signal_view == 0u || view.signal_view == 4u)) { col = amino_color(aa, tint_view[slot]); }
        sprite_col = aa;
        let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
        if (oc != 0u) {
            organ = (oc & 0x1Fu) - 1u;
            r_world *= ORGAN_SCALE;
            if (glyph) {
                // Variante do aspeto pelo código do órgão (igual na linhagem)
                // e a cor do órgão, com que o sprite cinzento é pintado.
                sprite_col = ((oc >> 5u) * 7u + organ) % SPRITE_COLS_U;
                col = organ_lod_color(organ, oc, col);
            }
            // FIO entre dois genes: um tubo fino e cinzento, sem desenho de órgão.
            if (organ == ORGAN_LINKER) {
                r_world = 0.9;
                col = vec3<f32>(0.55, 0.6, 0.7);
            }
            if (organ == ORGAN_PROTEASE) {
                col = vec3<f32>(0.9, 0.2, 0.2);
                // ESPIGÕES: do comprimento do alcance da variante, abertos
                // na medida em que a protease está ligada (sempre, ou pelo
                // sinal do seu canal); recolhidos ficam os dentes curtos.
                let pv = organ_variants_view[ORGAN_PROTEASE * ORGAN_VARIANTS + min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u)];
                var drive = 1.0;
                if (pv.p2 >= 0.0) { drive = clamp(signals_view[base + k][u32(clamp(pv.p2, 0.0, 3.0))], 0.0, 1.0); }
                // O mesmo limiar da simulacao (PROTEASE_MIN_DRIVE em contact.wgsl).
                drive = select(0.0, drive, drive >= 0.25);
                spike = max(pv.p0, 0.0) * drive;
                // Para o desenho: numero de espigoes pela FORCA (variante x
                // intensidade) na parte inteira, abertura na fracionaria.
                let force = max(pv.p1, 0.0) * exp2((f32(oc >> 8u) - 32.0) / 8.0);
                phase = clamp(round(4.0 + 3.5 * force), 6.0, 24.0) + drive * 0.99;
                // Flash: so a cor, com a forma que a protease tem no momento.
                if (bite.z > 0.0 && glyph && drive > 0.0) { flash = BITE_ATTACK_COLOR; }
            }
            if (organ == ORGAN_FOOD_SENSOR_DIR || organ == ORGAN_LIGHT_SENSOR_DIR) {
                // SENSOR DE UM LADO: o lado que le (como em organs.wgsl):
                // indice de intensidade par = esquerda, impar = direita, e
                // troca por cada orgao quiral antes dele na cadeia.
                var side = select(1.0, -1.0, ((oc >> 8u) & 1u) == 1u);
                for (var j = 0u; j < k; j++) {
                    let oj = (organs_view[slot * 32u + j / 2u] >> ((j % 2u) * 16u)) & 0xFFFFu;
                    if (oj != 0u && (oj & 0x1Fu) - 1u == ORGAN_CHIRAL) { side = -side; }
                }
                phase = side;
            }
            if (organ == ORGAN_STORAGE && glyph) {
                // DEPOSITO: UM oval por fila de depositos seguidos, a todo o
                // comprimento REAL da fila (o residuo k ocupa o troco k -> k+1,
                // e residue_len em body.wgsl cresce com o que o deposito
                // guarda). So o primeiro da fila desenha; dois depositos
                // justapostos ficam uma peca com o dobro do comprimento.
                var ks = k;
                var ke = k;
                for (var j = 0u; j < 16u; j++) {
                    if (ks > 0u && is_storage(slot, ks - 1u)) { ks -= 1u; }
                    if (ke + 1u < a.body_len && is_storage(slot, ke + 1u)) { ke += 1u; }
                }
                skip_glyph = k != ks;
                let pa = body_pos_view[base + ks];
                var pb = body_pos_view[base + ke] + tn * RESIDUE_UNITS * 3.0;
                if (ke + 1u < a.body_len) { pb = body_pos_view[base + ke + 1u]; }
                let mid = 0.5 * (pa + pb);
                let ax = pb - pa;
                let al = max(length(ax), 1e-3);
                let half = max(0.5 * al, r_world);
                centre = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * mid.x - sr * mid.y, sr * mid.x + cr * mid.y);
                tangent = vec2<f32>(cr * ax.x - sr * ax.y, sr * ax.x + cr * ax.y) / al;
                r_world = min(half / STORAGE_ASPECT, STORAGE_MAX_HALF_WIDTH * sqrt(half / (STORAGE_MAX_HALF_WIDTH * STORAGE_ASPECT)));
                phase = half / r_world;
            }
            if (organ == ORGAN_INHIBITOR) { col = vec3<f32>(0.95, 0.8, 0.9); }
            if (organ == ORGAN_HOLDFAST) { col = vec3<f32>(0.85, 0.6, 0.3); }
            if (organ == ORGAN_CHIRAL) { col = vec3<f32>(0.95, 0.35, 0.85); }
            if (organ == ORGAN_AGE_BIAS) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                let beta = organ_variants_view[ORGAN_AGE_BIAS * ORGAN_VARIANTS + p].p0 >= 0.5;
                col = select(vec3<f32>(1.0, 0.55, 0.15), vec3<f32>(0.35, 0.95, 0.35), beta);
            }
            if (organ == ORGAN_BIAS) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                let beta = organ_variants_view[ORGAN_BIAS * ORGAN_VARIANTS + p].p0 >= 0.5;
                col = select(vec3<f32>(1.0, 0.55, 0.15), vec3<f32>(0.35, 0.95, 0.35), beta);
            }
            if (organ == ORGAN_ANCHOR) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                let plus = organ_variants_view[ORGAN_ANCHOR * ORGAN_VARIANTS + p].p0 >= 0.0;
                col = select(vec3<f32>(0.25, 0.5, 1.0), vec3<f32>(0.15, 0.9, 0.8), plus);
            }
            if (organ == ORGAN_CLOCK) {
                let p = min((oc >> 5u) & 0x7u, ORGAN_VARIANTS - 1u);
                // Período da variante (o ponteiro ignora a modulação por α/β).
                let period = max(organ_variants_view[ORGAN_CLOCK * ORGAN_VARIANTS + p].p1, 2.0);
                phase = 6.2831853 * f32(a.age) / period;
            }
        }
        let s = signals_view[base + k];
        switch view.signal_view {
            case 1u: { col = signed_color(s.x, vec3<f32>(1.0, 0.45, 0.1), vec3<f32>(0.1, 0.6, 1.0)); }
            case 2u: { col = signed_color(s.y, vec3<f32>(0.3, 1.0, 0.3), vec3<f32>(0.95, 0.3, 0.9)); }
            case 3u: { col = vec3<f32>(0.5 + 0.5 * tanh(s.x), 0.5 + 0.5 * tanh(s.y), 0.35); }
            case 5u: { col = vec3<f32>(0.5 + 0.5 * tanh(s.z), 0.5 + 0.5 * tanh(s.w), 0.35); }
            default: {}
        }
    }
    // Pouca energia = mais escuro (só na vista química).
    let dim = energy_tone(a);
    o.color = select(col, col * dim, view.signal_view == 0u || view.signal_view == 4u);
    if (flash.x >= 0.0 && view.signal_view == 0u) { o.color = flash; }
    if (!glyph && !naked) {
        // TUBO: cápsula do resíduo k até ao k+1 (o último só tem a ponta).
        // A espessura é a do resíduo k sem o aumento dos órgãos.
        var r_tube = r_world;
        if (organ != NO_ORGAN) { r_tube /= ORGAN_SCALE; }
        // O troço por baixo de uma protease é fino (um pedúnculo): o que se
        // vê é a bola com os espigões, não um chouriço vermelho.
        if (organ == ORGAN_PROTEASE) { r_tube *= PROTEASE_STALK; }
        r_tube *= detail_fat();
        var b = centre;
        if (k + 1u < a.body_len) {
            let lp1 = body_pos_view[slot * MAX_BODY_V + k + 1u];
            let cr = cos(a.rot);
            let sr = sin(a.rot);
            b = vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * lp1.x - sr * lp1.y, sr * lp1.x + cr * lp1.y);
        }
        let seg = b - centre;
        let l = length(seg);
        let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
        let nn = vec2<f32>(-e.y, e.x);
        // Quadrado orientado que cobre a cápsula.
        // (Com a margem da sombra, menos no fio entre genes.)
        let rq = r_tube * select(TUBE_SHADOW, 1.0, organ == ORGAN_LINKER);
        let along = select(-rq, l + rq, c.x > 0.0);
        let w = centre + e * along + nn * (c.y * rq);
        let px = (w - cam_center()) * view.zoom;
        o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
        o.mode = 0u;
        o.local = w - centre;
        o.tangent = seg;
        o.core_phase = vec2<f32>(r_tube, 0.0);
        o.organ = organ;
        o.sprite = select(0u, 0x100u | sprite_col, organ != ORGAN_LINKER);
        o.lift = body_lift(a, k);
        return o;
    }
    if (glyph && !naked && (organ == NO_ORGAN || organ == ORGAN_LINKER || skip_glyph)) {
        // Resíduo estrutural: só o tubo.
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    if (glyph && naked) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    var ext = organ_extent(organ) + spike / max(r_world, 1e-3);
    // O quadrado do deposito tem de cobrir o oval ao comprido.
    if (organ == ORGAN_STORAGE) { ext = phase * SHADOW_MARGIN; }
    // Margem para a sombra (a protease já tem o quadrado dos espigões).
    if (organ != ORGAN_STORAGE && organ != ORGAN_PROTEASE) { ext *= SHADOW_MARGIN; }
    let r = r_world * ext * detail_fat();
    let w = centre + c * r;
    let px = (w - cam_center()) * view.zoom;
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.local = c;
    o.organ = organ;
    o.tangent = tangent;
    o.core_phase = vec2<f32>(1.0 / ext, phase);
    o.sprite = sprite_col;
    o.size = r;
    o.lift = body_lift(a, k) + organ_stagger(k, r / SHADOW_MARGIN);
    o.mode = 1u;
    return o;
}

// Quadrado (vi) que cobre a cápsula a–b de raio r, em modo tubo.
fn capsule_vertex(vi: u32, a_w: vec2<f32>, b_w: vec2<f32>, r: f32, col: vec3<f32>) -> AgentVsOut {
    var o: AgentVsOut;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    let seg = b_w - a_w;
    let l = length(seg);
    let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
    let nn = vec2<f32>(-e.y, e.x);
    let along = select(-r, l + r, c.x > 0.0);
    let w = a_w + e * along + nn * (c.y * r);
    let px = (w - cam_center()) * view.zoom;
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.mode = 0u;
    o.local = w - a_w;
    o.tangent = seg;
    o.core_phase = vec2<f32>(r, 0.0);
    o.organ = NO_ORGAN;
    o.color = col;
    return o;
}

// Base j das pontas: j < RNA_PER_END = 5' UTR (antes do AUG, a sair da
// ponta N); senão 3' UTR (depois do stop, a sair da ponta C). RNA nu: o
// genoma todo, a partir do centro. O fio ondula devagar com a idade.
fn residue_world_v(slot: u32, a: Agent, k: u32) -> vec2<f32> {
    let lp = body_pos_view[slot * MAX_BODY_V + k];
    let cr = cos(a.rot);
    let sr = sin(a.rot);
    return vec2<f32>(a.pos_x, a.pos_y) + vec2<f32>(cr * lp.x - sr * lp.y, sr * lp.x + cr * lp.y);
}

// LIGAÇÃO i do agente: tubo fino entre os dois resíduos. Só a
// desenha o lado de slot menor (o outro tem a mesma ligação ao contrário).
fn bond_vertex(vi: u32, slot: u32, a: Agent, i: u32) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let b = bonds_view[slot * BOND_STRIDE_V + i];
    // (Em foco só este agente é desenhado: desenha as suas ligações todas.)
    if (a.alive == 0u || b.x == 0xFFFFFFFFu || (b.x <= slot && view.focus_slot == 0xFFFFFFFFu)) { return o; }
    let other = agents_view[b.x];
    if (other.alive == 0u || other.id != b.y) { return o; }
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot && b.x != view.focus_slot;
    let mine = b.z & 0xFFu;
    let theirs = (b.z >> 8u) & 0xFFu;
    if (hidden || mine >= a.body_len || theirs >= other.body_len) { return o; }
    let p0 = residue_world_v(slot, a, mine);
    let p1 = residue_world_v(b.x, other, theirs);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let c = corners[vi];
    // Nunca menos de 1,5 píxeis: vê-se a qualquer zoom.
    let r_tube = max(2.0, 1.5 / view.zoom);
    let seg = p1 - p0;
    let l = length(seg);
    let e = select(vec2<f32>(1.0, 0.0), seg / l, l > 1e-4);
    let nn = vec2<f32>(-e.y, e.x);
    let along = select(-r_tube, l + r_tube, c.x > 0.0);
    let w = p0 + e * along + nn * (c.y * r_tube);
    let px = (w - cam_center()) * view.zoom;
    o.pos = vec4<f32>(px.x / (0.5 * view.screen_w), px.y / (0.5 * view.screen_h), 0.0, 1.0);
    o.mode = 0u;
    o.local = w - p0;
    o.tangent = seg;
    o.core_phase = vec2<f32>(r_tube, 0.0);
    o.organ = NO_ORGAN;
    // Dourado = ligada ao tocar; azul-claro = de nascimento (pai e filho).
    o.color = select(vec3<f32>(1.0, 0.82, 0.35), vec3<f32>(0.45, 0.85, 1.0), (b.z >> 16u) != 0u);
    return o;
}

// PARENTESCO (vista 4): bola por cima do agente, do mesmo tamanho no ecrã
// para todos. Verde = genoma próximo do selecionado, amarelo, vermelho = distante.
const MARK_BONDED: u32 = 255u;
const KIN_DOTS: u32 = 0u;
fn kin_vertex(vi: u32, slot: u32, a: Agent) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    // MARCAR ÓRGÃO: bola ciano por cima de quem tem o tipo escolhido (a mesma
    // bola do parentesco; tem prioridade sobre ele).
    if (view.mark_organ == MARK_BONDED) {
        // MARCAR LIGADOS: bola dourada em quem tem uma ligação viva.
        if (a.alive == 0u) { return o; }
        var tied = false;
        for (var i = 0u; i < BONDS_V; i++) {
            let b = bonds_view[slot * BOND_STRIDE_V + i];
            if (b.x != 0xFFFFFFFFu && agents_view[b.x].alive != 0u && agents_view[b.x].id == b.y) { tied = true; }
        }
        if (!tied) { return o; }
        let cb = vec2<f32>(a.pos_x, a.pos_y);
        return capsule_vertex(vi, cb, cb, KIN_DOT_PX / view.zoom, vec3<f32>(1.0, 0.82, 0.2));
    }
    if (view.mark_organ != 0u) {
        if (a.alive == 0u) { return o; }
        var has = false;
        for (var k = 0u; k < min(a.body_len, MAX_BODY_V); k++) {
            let oc = (organs_view[slot * 32u + k / 2u] >> ((k % 2u) * 16u)) & 0xFFFFu;
            if (oc != 0u && (oc & 0x1Fu) == view.mark_organ) {
                has = true;
                break;
            }
        }
        if (!has) { return o; }
        let cm = vec2<f32>(a.pos_x, a.pos_y);
        return capsule_vertex(vi, cm, cm, KIN_DOT_PX / view.zoom, vec3<f32>(0.1, 0.95, 1.0));
    }
    let q = kin_view[slot];
    // (O parentesco já não é uma bola por agente: é o mapa de Voronoi do
    // fundo, kin_map em world_view.wgsl.)
    if (KIN_DOTS == 0u || view.signal_view != 4u || a.alive == 0u || q < 0.0) { return o; }
    // Raiz quadrada: mais resolução nos parentescos fracos (os clãs
    // distantes distinguem-se uns dos outros): 25% de 8-meros -> meio da escala.
    let t = sqrt(clamp(q, 0.0, 1.0));
    let col = select(mix(vec3<f32>(1.0, 0.85, 0.1), vec3<f32>(0.15, 1.0, 0.25), (t - 0.5) * 2.0),
                     mix(vec3<f32>(1.0, 0.12, 0.08), vec3<f32>(1.0, 0.85, 0.1), t * 2.0), t < 0.5);
    let c = vec2<f32>(a.pos_x, a.pos_y);
    return capsule_vertex(vi, c, c, KIN_DOT_PX / view.zoom, col);
}

// Ponto u (0..3) da curva de Catmull-Rom por p0..p3; antes de p0 a curva
// vem da direção do corpo (um ponto fantasma em p0 - out).
fn tail_curve(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, out: vec2<f32>, u: f32) -> vec2<f32> {
    let seg = min(floor(u), 2.0);
    let t = u - seg;
    var a = p0 - out;
    var b = p0;
    var c = p1;
    var d = p2;
    if (seg > 0.5) { a = p0; b = p1; c = p2; d = p3; }
    if (seg > 1.5) { a = p1; b = p2; c = p3; d = p3 + (p3 - p2); }
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * ((2.0 * b) + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (3.0 * b - a - 3.0 * c + d) * t3);
}

const SHOW_RNA_TAILS: bool = false;
fn rna_vertex(vi: u32, slot: u32, a: Agent, j: u32) -> AgentVsOut {
    var o: AgentVsOut;
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let hidden = view.focus_slot != 0xFFFFFFFFu && slot != view.focus_slot;
    if (a.alive == 0u || hidden) { return o; }
    // As caudas de RNA não traduzido de um corpo não se desenham (faziam
    // confusão); o RNA NU, sem corpo, continua a desenhar-se: é o agente.
    if (!SHOW_RNA_TAILS && a.body_len > 0u) { return o; }
    let trailer = j >= RNA_PER_END;
    let m = j % RNA_PER_END;
    let start = a.coding_span & 0xFFFFu;
    let after = a.coding_span >> 16u;
    let n = a.body_len;
    var base_i = 0u;
    var anchor = vec2<f32>(0.0);
    var dir = vec2<f32>(select(-1.0, 1.0, trailer), 0.0);
    if (n == 0u) {
        // RNA nu: as duas metades do genoma, em sentidos opostos.
        base_i = select(m, RNA_PER_END + m, trailer);
        if (base_i >= a.gene_len) { return o; }
    } else if (!trailer) {
        if (m >= start) { return o; }
        base_i = start - 1u - m;
        anchor = body_pos_view[slot * MAX_BODY_V];
        if (n > 1u) { dir = normalize(anchor - body_pos_view[slot * MAX_BODY_V + 1u] + vec2<f32>(1e-6, 0.0)); }
    } else {
        base_i = after + m;
        if (base_i >= a.gene_len) { return o; }
        anchor = body_pos_view[slot * MAX_BODY_V + n - 1u];
        if (n > 1u) { dir = normalize(anchor - body_pos_view[slot * MAX_BODY_V + n - 2u] + vec2<f32>(1e-6, 0.0)); }
    }
    let cr = cos(a.rot);
    let sr = sin(a.rot);
    let c0 = vec2<f32>(a.pos_x, a.pos_y);
    var pw = c0;
    var qw = c0;
    if (n == 0u) {
        // RNA nu: caminha base a base, a ondular devagar.
        var ang = atan2(dir.y, dir.x);
        var p = anchor;
        var prev = anchor;
        let seed = f32(a.id % 977u) * 0.37 + select(0.0, 2.1, trailer);
        for (var t = 0u; t <= m; t++) {
            ang += RNA_WIGGLE * sin(f32(t) * 0.7 + f32(a.age) * 0.03 + seed);
            prev = p;
            p += vec2<f32>(cos(ang), sin(ang)) * RNA_SPACING;
        }
        pw = c0 + vec2<f32>(cr * p.x - sr * p.y, sr * p.x + cr * p.y);
        qw = c0 + vec2<f32>(cr * prev.x - sr * prev.y, sr * prev.x + cr * prev.y);
    } else {
        // FITA MOLE (ver update_rna_tails): curva suave (Catmull-Rom) pela
        // raiz e pelos 3 pontos do fio guardados no mundo.
        let cnt = f32(max(select(min(start, RNA_PER_END), min(a.gene_len - min(after, a.gene_len), RNA_PER_END), trailer), 1u));
        let link = cnt * RNA_SPACING / 3.0;
        let p0 = c0 + vec2<f32>(cr * anchor.x - sr * anchor.y, sr * anchor.x + cr * anchor.y);
        let dw = vec2<f32>(cr * dir.x - sr * dir.y, sr * dir.x + cr * dir.y);
        let t0 = rna_tail_view[slot * 4u];
        let t2 = rna_tail_view[slot * 4u + 2u];
        let t3 = rna_tail_view[slot * 4u + 3u];
        var p1 = select(t0.xy, t2.zw, trailer);
        var p2 = select(t0.zw, t3.xy, trailer);
        var p3 = select(t2.xy, t3.zw, trailer);
        // Estado atrasado ou de outro agente: o fio sai a direito em vez de
        // se esticar pelo mundo fora.
        let lim = 2.0 * link + 1.0;
        if (length(p1 - p0) > lim || length(p2 - p1) > lim || length(p3 - p2) > lim) {
            p1 = p0 + dw * link;
            p2 = p1 + dw * link;
            p3 = p2 + dw * link;
        }
        qw = tail_curve(p0, p1, p2, p3, dw * link, 3.0 * f32(m) / cnt);
        pw = tail_curve(p0, p1, p2, p3, dw * link, 3.0 * f32(m + 1u) / cnt);
    }
    // (As caudas de RNA mudam de tom com a energia, como o corpo.)
    let tone = select(1.0, energy_tone(a), view.signal_view == 0u || view.signal_view == 4u);
    return capsule_vertex(vi, qw, pw, max(RNA_RADIUS, 1.0 / view.zoom), base_color(genome_base(slot, base_i)) * 0.85 * tone);
}

// Distância de p ao segmento a–b.
fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * h);
}

// Antena do centro até `tip` (com um botão na ponta): 1 dentro, 0 fora.
fn antenna(p: vec2<f32>, tip: vec2<f32>, core: f32) -> f32 {
    let stalk = seg_dist(p, vec2<f32>(0.0), tip) < 0.05;
    let knob = length(p - tip) < 0.11;
    return select(0.0, 1.0, (stalk && length(p) > core * 0.8) || knob);
}

// Dois passos com o mesmo desenho: primeiro as peças opacas (escrevem a
// altura), depois as sombras (transparentes, só sobre o que está mais baixo).
@fragment
fn fs_agent(in: AgentVsOut) -> @location(0) vec4<f32> {
    g_h = agent_height(in);
    g_c = g_h;
    g_lift = 0.0;
    let c = agent_frag(in);
    if (c.a < 0.99) { discard; }
    // VOLUME (cimo, fundo) acima do chão. O corpo INTEIRO tem o meio à mesma
    // cota (AGENT_Z, mais a ondulação in.lift): os troços e os órgãos ficam
    // enfiados uns nos outros como contas num fio, a pairar um pouco acima do
    // fundo; um órgão grande que chegasse ao chão fica achatado por baixo.
    if (view.height_pass != 0u) {
        let mid = AGENT_Z + in.lift + g_lift;
        return vec4<f32>(mid + g_h, max(mid - g_h, 0.4), 0.0, 1.0);
    }
    return c;
}

// O mesmo para o MICROSCÓPIO 3D, mas quem tapa quem é decidido pela ALTURA
// real da peça naquele ponto (o seu cimo) e não pela ordem fixa dos agentes:
// onde dois órgãos se sobrepõem fica o mais alto, e a fronteira é a linha
// onde as duas formas se cruzam, sem parede a pique. Usa-se nos dois desenhos
// (cor e volume), para a cor de cada ponto ser a da peça que lá ficou.
struct ReliefOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fs_agent_relief(in: AgentVsOut) -> ReliefOut {
    g_h = agent_height(in);
    g_c = g_h;
    g_lift = 0.0;
    let c = agent_frag(in);
    if (c.a < 0.99) { discard; }
    let mid = AGENT_Z + in.lift + g_lift;
    var o: ReliefOut;
    // (Um órgão ganha aos troços do corpo que lhe entram pela borda: sem esta
    // folga viam-se lascas do tubo a furar as bolas.)
    o.depth = clamp((mid + g_h + select(0.0, 12.0, in.mode == 1u)) / 1000.0, 0.02, 0.999);
    o.color = c;
    if (view.height_pass != 0u) { o.color = vec4<f32>(mid + g_h, max(mid - g_h, 0.4), 0.0, 1.0); }
    return o;
}

// ALTURA de uma peça SEM sprite (fios, ligações, troços de longe), para o
// relevo do microscópio 3D: meio cilindro, ou uma cúpula. As peças com
// sprite usam a altura do atlas (g_h).
fn agent_height(in: AgentVsOut) -> f32 {
    if (in.mode == 0u) {
        let r = max(in.core_phase.x, 1e-3);
        let x = clamp(seg_dist(in.local, vec2<f32>(0.0), in.tangent) / r, 0.0, 1.0);
        return r * sqrt(1.0 - x * x);
    }
    // Raio do órgão (unidades do mundo) e, em raios do órgão, o do seu
    // CORPO: o que fica fora dele mas dentro do desenho (antenas, espigões)
    // é fino e baixo.
    let core = in.core_phase.x;
    let unit = in.size * core;
    var body = 1.0 / (core * SHADOW_MARGIN);
    if (in.organ == ORGAN_PROTEASE) { body = 0.62; }
    if (in.organ == ORGAN_FOOD_SENSOR) { body = 1.24; }
    if (in.organ == ORGAN_LIGHT_SENSOR) { body = 1.36; }
    if (in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR) { body = 1.2; }
    if (in.organ == ORGAN_STORAGE) { body = 1.0; }
    var d = length(in.local) / (core * body);
    if (in.organ == ORGAN_STORAGE) {
        // Oval ao longo da cadeia (alongamento em core_phase.y).
        let t = in.tangent;
        d = length(vec2<f32>(dot(in.local, t) / max(in.core_phase.y, 1.0), dot(in.local, vec2<f32>(-t.y, t.x)))) / core;
    }
    let dome = sqrt(max(1.0 - d * d, 0.0));
    return unit * body * dome;
}

@fragment
fn fs_agent_shadow(in: AgentVsOut) -> @location(0) vec4<f32> {
    let c = agent_frag(in);
    if (c.a >= 0.99) { discard; }
    return c;
}

fn agent_frag(in: AgentVsOut) -> vec4<f32> {
    // Derivadas no ecrã das coordenadas locais (antes de qualquer ramo).
    let tdx = dpdx(in.local);
    let tdy = dpdy(in.local);
    if (in.mode == 0u) {
        // TUBO: cápsula com sombreado de cilindro (centro claro, bordas escuras).
        let r_t = in.core_phase.x;
        let dd = seg_dist(in.local, vec2<f32>(0.0), in.tangent);
        if ((in.sprite & 0x100u) != 0u) {
            // Troço de aminoácido: o sprite do seu tipo esticado ao troço. A
            // FORMA é a do sprite (cada tipo tem as suas deformações, que
            // saem um pouco da cápsula); fora dela, a sombra de contacto.
            let l = length(in.tangent);
            let e = select(vec2<f32>(1.0, 0.0), in.tangent / l, l > 1e-4);
            let nn = vec2<f32>(-e.y, e.x);
            let sc = vec2<f32>(2.0 / (l + 2.0 * r_t), 1.0 / r_t) / AMINO_MARGIN;
            let q = vec2<f32>((dot(in.local, e) - 0.5 * l) * sc.x, dot(in.local, nn) * sc.y);
            let aa = in.sprite & 0xFFu;
            let s = sprite(SPRITE_ROW_AMINO + f32(aa / SPRITE_COLS_U), f32(aa % SPRITE_COLS_U), q, vec2<f32>(dot(tdx, e), dot(tdx, nn)) * sc, vec2<f32>(dot(tdy, e), dot(tdy, nn)) * sc);
            // (Mais gordo do que a forma insuflada do sprite: as hélices e as
            // fitas são estreitas e ficavam quase planas no microscópio 3D.)
            g_h = AMINO_RELIEF * s.z * r_t * AMINO_MARGIN;
            g_c = r_t;
            if (s.y >= 0.5) { return vec4<f32>(sem_color(in.color, s.x), 1.0); }
            // Sombra com a FORMA do sprite (hélice, fita…): a sua máscara
            // desfocada, e não uma auréola de cápsula.
            let sh = SHADOW * smoothstep(0.03, 0.4, sprite_soft(SPRITE_ROW_AMINO + f32(aa / SPRITE_COLS_U), f32(aa % SPRITE_COLS_U), q));
            if (sh < 0.004) { discard; }
            return vec4<f32>(0.0, 0.0, 0.0, sh);
        }
        if (dd > r_t) { discard; }
        let x = dd / r_t;
        let shade = sqrt(max(1.0 - x * x, 0.0));
        let c = in.color * (0.35 + 0.65 * shade) + vec3<f32>(0.18) * pow(shade, 8.0);
        return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    let p = in.local;
    let d = length(p);
    let core = in.core_phase.x;
    let t = in.tangent;
    let nrm = vec2<f32>(-t.y, t.x);
    // Coordenadas ao longo (u) e através (v) da cadeia.
    let u = dot(p, t);
    let v = dot(p, nrm);
    // Coordenadas dos sprites: (ao longo, através) em raios do órgão, e as
    // suas derivadas no ecrã (calculadas aqui, antes de qualquer ramo).
    let uv_l = vec2<f32>(u, v) / core;
    let uv_x = vec2<f32>(dot(tdx, t), dot(tdx, nrm)) / core;
    let uv_y = vec2<f32>(dot(tdy, t), dot(tdy, nrm)) / core;
    let col_f = f32(in.sprite & 0xFFu);
    // ÓRGÃOS COM SPRITE: o mosaico do órgão cobre o quadrado inteiro (o
    // tamanho de cada um está em organ_extent). O depósito (esticado) e a
    // protease (corpo + espigões) têm o seu ramo mais abaixo.
    if (in.organ != NO_ORGAN && in.organ != ORGAN_STORAGE && in.organ != ORGAN_PROTEASE && in.organ != ORGAN_LINKER) {
        // (O sprite ocupa o quadrado menos a margem da sombra.)
        var q = vec2<f32>(u, v) * SHADOW_MARGIN;
        // Raio do corpo no mosaico (nos sensores, a esfera; as antenas não fazem sombra).
        var body = 0.97;
        if (in.organ == ORGAN_FOOD_SENSOR) { body = 0.62; }
        if (in.organ == ORGAN_LIGHT_SENSOR) { body = 0.85; }
        if (in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR) { body = SPRITE_STALK_BODY; }
        if (in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR) {
            // UMA antena, do lado que o sensor lê (core_phase.y = +1 esquerda, -1 direita).
            q.y *= select(-1.0, 1.0, in.core_phase.y >= 0.0);
        }
        if (in.organ == ORGAN_CLOCK) {
            // A espiral roda com a fase do relógio.
            let cs = cos(in.core_phase.y);
            let sn = sin(in.core_phase.y);
            q = vec2<f32>(q.x * cs - q.y * sn, q.x * sn + q.y * cs);
        }
        let sp = sprite(f32(in.organ), col_f, q, uv_x * core * SHADOW_MARGIN, uv_y * core * SHADOW_MARGIN);
        g_h = sp.z * in.size / SHADOW_MARGIN;
        // (Os sensores são achatados, como discos, e não bolas.)
        if (in.organ == ORGAN_FOOD_SENSOR || in.organ == ORGAN_LIGHT_SENSOR || in.organ == ORGAN_FOOD_SENSOR_DIR || in.organ == ORGAN_LIGHT_SENSOR_DIR || in.organ == ORGAN_ENERGY_SENSOR) { g_h *= SENSOR_FLAT; }
        g_c = 0.8 * in.size / SHADOW_MARGIN;
        if (sp.y < 0.5) {
            let sh = halo(length(q) / body);
            if (sh < 0.004) { discard; }
            return vec4<f32>(0.0, 0.0, 0.0, sh);
        }
        return vec4<f32>(sem_color(in.color, sp.x), 1.0);
    }
    let white = vec3<f32>(1.0);
    let rim_mix = smoothstep(core * 0.6, core * 0.8, d);

    switch in.organ {
        case NO_ORGAN: {
            if (d > 1.0) { discard; }
            let rim = smoothstep(0.75, 0.98, d);
            return vec4<f32>(mix(in.color, vec3<f32>(0.0), rim * 0.7), 1.0);
        }
        case ORGAN_MOUTH: {
            if (d > core) { discard; }
            // Abertura em cunha, virada para o lado esquerdo da cadeia.
            if (v > 0.15 * core && abs(u) < v * 0.9) { return vec4<f32>(0.05, 0.02, 0.02, 1.0); }
            return vec4<f32>(mix(in.color, white, rim_mix), 1.0);
        }
        case ORGAN_MUSCLE: {
            let e = (u / (core * 1.5)) * (u / (core * 1.5)) + (v / (core * 0.85)) * (v / (core * 0.85));
            if (e > 1.0) { discard; }
            let stripe = step(0.0, sin(u / core * 12.0));
            let red = mix(in.color, vec3<f32>(0.85, 0.25, 0.25), 0.55);
            return vec4<f32>(mix(red, red * 0.6, stripe) * mix(1.0, 0.7, smoothstep(0.7, 1.0, e)), 1.0);
        }
        case ORGAN_ENERGY_SENSOR: {
            if (d > core) { discard; }
            let ring = abs(d - core * 0.45) < core * 0.12;
            return vec4<f32>(select(mix(in.color, white, rim_mix), vec3<f32>(1.0, 0.85, 0.2), ring), 1.0);
        }
        case ORGAN_CLOCK: {
            if (d > core) { discard; }
            let hand_dir = t * cos(in.core_phase.y) + nrm * sin(in.core_phase.y);
            if (seg_dist(p, vec2<f32>(0.0), hand_dir * core * 0.8) < core * 0.12) {
                return vec4<f32>(0.05, 0.05, 0.1, 1.0);
            }
            return vec4<f32>(mix(vec3<f32>(0.85, 0.9, 1.0), white, rim_mix), 1.0);
        }
        case ORGAN_RELAY: {
            if (abs(u) + abs(v) > core * 1.25) { discard; }
            let edge = smoothstep(core * 0.85, core * 1.1, abs(u) + abs(v));
            return vec4<f32>(mix(in.color, vec3<f32>(0.6, 0.9, 1.0), edge), 1.0);
        }
        case ORGAN_PHOTOSYSTEM: {
            // Disco verde com 8 raios curtos (capta luz).
            let ang = atan2(v, u);
            let ray = d < 0.95 && d > core && abs(sin(ang * 4.0)) < 0.25;
            if (d > core && !ray) { discard; }
            let leaf = vec3<f32>(0.35, 0.95, 0.35);
            return vec4<f32>(select(mix(leaf * 0.7, leaf, 1.0 - d / core), vec3<f32>(0.85, 1.0, 0.4), ray), 1.0);
        }
        case ORGAN_CHEMO: {
            // Disco amarelo-enxofre com pintas escuras (consome o redutor).
            if (d > core) { discard; }
            let spot = step(0.6, fract(u * 3.7 / core) * fract(v * 3.1 / core) * 2.5);
            return vec4<f32>(mix(vec3<f32>(0.9, 0.78, 0.15), vec3<f32>(0.35, 0.28, 0.05), spot), 1.0);
        }
        case ORGAN_BIAS: {
            // Ponto cheio (laranja = α, verde = β) com um contorno claro.
            if (d > core) { discard; }
            return vec4<f32>(mix(in.color, vec3<f32>(1.0), smoothstep(core * 0.75, core, d)), 1.0);
        }
        case ORGAN_ANCHOR: {
            // Anel grosso (turquesa = +, azul = −; nunca vermelho, que é das proteases) com o centro escuro.
            if (d > core * 1.15 || d < core * 0.5) { discard; }
            let edge = smoothstep(core * 0.95, core * 1.15, d);
            return vec4<f32>(mix(in.color, in.color * 0.4, edge), 1.0);
        }
        case ORGAN_DORMANCY: {
            // Lua em crescente (azul-clara): disco menos um disco deslocado.
            if (d > core) { discard; }
            let bite = length(vec2<f32>(u, v) - vec2<f32>(core * 0.45, core * 0.2));
            if (bite < core * 0.8) { discard; }
            return vec4<f32>(0.72, 0.82, 1.0, 1.0);
        }
        case ORGAN_PROTEASE: {
            // OURICO: tantos espigoes quanta a forca da protease (6 a 24),
            // cada um com o seu angulo e comprimento (um hash do indice).
            // Desligada, estao recolhidos num feixe para um lado da
            // cadeia; ligada, abrem a toda a volta. Com alcance, o mais
            // comprido tem o alcance (o quadrado cresce no vertice).
            let count = floor(in.core_phase.y);
            let open = clamp((in.core_phase.y - count) / 0.99, 0.0, 1.0);
            // Raiz: um sinal fraco ja abre bastante (a protease ja morde).
            let half = mix(0.22, 3.0, sqrt(open));
            let gap = 2.0 * half / count;
            let ang = atan2(u, v);
            let hub = core * 0.62;
            // Cada espigão é um sprite (ponta para fora), desenhado POR CIMA do
            // corpo: nasce a meio dele.
            let a0 = hub * 0.45;
            let wb = core * 0.2;
            let g = max(length(uv_x), length(uv_y)) * core;
            var spike_l = -1.0;
            var tip = 0.0;
            for (var i = 0.0; i < count; i += 1.0) {
                let r1 = fract(sin(i * 12.9898 + 4.1) * 43758.5453);
                let r2 = fract(sin(i * 78.233 + 1.7) * 24634.6345);
                let at = -half + gap * (i + 0.5 + (r1 - 0.5) * 0.9);
                let len = 0.97 * (0.5 + 0.5 * r2);
                var da = abs(ang - at);
                da = min(da, 6.2831853 - da);
                let along = u * sin(at) + v * cos(at);
                let across = u * cos(at) - v * sin(at);
                let q = vec2<f32>(across / wb, (along - a0) / max(len - a0, 1e-3) * 2.0 - 1.0);
                if (da < 1.5 && abs(q.x) < 1.0 && abs(q.y) < 1.0) {
                    let s = sprite(SPRITE_ROW_SPIKE, (col_f + i) % SPRITE_COLS, q, vec2<f32>(g / wb, 0.0), vec2<f32>(0.0, 2.0 * g / max(len - a0, 1e-3)));
                    if (s.y >= 0.5 && s.x > spike_l) {
                        spike_l = s.x;
                        // (No plano do corpo, como antes: inclinados para
                        // fora do plano ficavam esfarelados no relevo.)
                        g_h = s.z * in.size * wb;
                        g_c = in.size * hub;
                        tip = len;
                    }
                }
            }
            if (spike_l < 0.0 && d <= hub) {
                let b = sprite(SPRITE_ROW_PROTEASE, col_f, vec2<f32>(u, v) / hub, uv_x * core / hub, uv_y * core / hub);
                if (b.y >= 0.5) {
                    g_h = b.z * in.size * hub;
                    g_c = in.size * hub;
                    return vec4<f32>(sem_color(in.color, b.x), 1.0);
                }
            }
            if (spike_l < 0.0) {
                let sh = halo(d / hub);
                if (sh < 0.004) { discard; }
                return vec4<f32>(0.0, 0.0, 0.0, sh);
            }
            let pale = mix(in.color, vec3<f32>(1.0, 0.8, 0.65), 0.3 + 0.5 * d / max(tip, 0.01));
            // Por cima da base, o relevo é pelo menos o da base: senão cada
            // espigão abria nela uma ranhura com a sua (pequena) espessura.
            if (d <= hub) {
                let bb = sprite(SPRITE_ROW_PROTEASE, col_f, vec2<f32>(u, v) / hub, uv_x * core / hub, uv_y * core / hub);
                if (bb.y >= 0.5) { g_h = max(g_h, bb.z * in.size * hub); }
            }
            return vec4<f32>(sem_color(pale, spike_l), 1.0);
        }
        case ORGAN_CHIRAL: {
            // Disco magenta partido ao meio: uma metade cheia, a outra
            // escura (um espelho).
            if (d > core) { discard; }
            return vec4<f32>(select(in.color, in.color * 0.2, u > 0.0), 1.0);
        }
        case ORGAN_HOLDFAST: {
            // Disco castanho com uma cruz escura (um pé assente no chão).
            if (d > core) { discard; }
            let cross = min(abs(u), abs(v)) < core * 0.2;
            return vec4<f32>(select(in.color, in.color * 0.25, cross), 1.0);
        }
        case ORGAN_AGE_BIAS: {
            // Meio disco cheio (laranja = α, verde = β), meio só contorno:
            // um bias que se vai apagando.
            if (d > core) { discard; }
            let hollow = u > 0.0 && d < core * 0.7;
            return vec4<f32>(select(in.color, in.color * 0.25, hollow), 1.0);
        }
        case ORGAN_STORAGE: {
            // Deposito: oval ao longo da cadeia (alongamento em core_phase.y,
            // pela capacidade), cheio, com aneis e um brilho de gota.
            let asp = clamp(in.core_phase.y, 1.0, 40.0);
            // SPRITE esticado ao comprimento real do depósito.
            let m = vec2<f32>(1.0 / asp, 1.0);
            let sp = sprite(f32(ORGAN_STORAGE), col_f, uv_l * m, uv_x * m, uv_y * m);
            g_h = sp.z * in.size * core;
            g_c = in.size * core;
            if (sp.y < 0.5) {
                let sh = halo(length(uv_l * m));
                if (sh < 0.004) { discard; }
                return vec4<f32>(0.0, 0.0, 0.0, sh);
            }
            return vec4<f32>(sem_color(in.color, sp.x), 1.0);
        }
        default: {
            // Outros: disco com anéis concêntricos.
            if (d > core) { discard; }
            let rings = step(0.5, fract(d / core * 3.0));
            return vec4<f32>(mix(in.color, vec3<f32>(0.15, 0.1, 0.0), rings * 0.7), 1.0);
        }
    }
}
