//! ÁRVORE DA VIDA interativa: as linhagens registadas (`lineage.rs`) num
//! canvas com zoom e arrasto, com o desenho das duas formas de cada ramo (o
//! corpo traduzido do genoma do ramo e o do seu complemento reverso, na
//! forma de repouso). É HTML + JavaScript sem dependências: um só ficheiro
//! que abre em qualquer browser, sem internet. Ferramenta do observador.

use std::fmt::Write as _;

use crate::life::amino::AA_LETTERS;
use crate::life::organs::{ORGAN_SYMBOLS, Residue, describe, translate_organs};
use crate::lineage::Lineages;
use crate::species::reverse_complement;
use crate::world::World;

/// Lado dos retratos (píxeis) e brilho do fundo.
const PORTRAIT: u32 = 192;
/// No máximo este número de ramos leva retrato (os outros ficam com o
/// desenho simplificado feito no browser).
const MAX_PORTRAITS: usize = 2000;

/// RETRATOS COM O ASPETO DA SIMULAÇÃO. Muitos ramos já se extinguiram, por
/// isso não há agente vivo para fotografar: monta-se um mundo à parte, sem
/// terreno nem corrente, semeia-se lá um agente com o genoma de cada ramo e
/// outro com o complemento, e desenha-se cada um com o mesmo código da vista
/// (tubos, órgãos, espigões). Devolve, por ramo, os PNG das formas A e B
/// (vazio se o agente não chegou a nascer).
pub fn portraits(gpu: &crate::gpu::Gpu, w: &World, l: &Lineages) -> Vec<[Vec<u8>; 2]> {
    use crate::params::SpawnRequest;
    let cfg = w.cfg;
    let clock = std::time::Instant::now();
    let mut t = World::new(gpu, cfg, 1);
    t.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    t.fumaroles.clear();
    t.seed_matter(gpu, 1);
    t.settings.fluid_enabled = false;
    // As mesmas regras do mundo real (ângulos, sinais), mas sem nada que
    // mate, mexa ou ilumine os modelos.
    let epoch = t.params.epoch;
    t.params = w.params;
    t.params.epoch = epoch;
    t.params.death_probability = 0.0;
    t.params.pairing_rate = 0.0;
    t.params.uptake_rate = 0.0;
    t.params.brownian_rot = 0.0;
    t.params.uv_strength = 0.0;
    t.params.sedimentation = 0.0;
    t.params.spawn_energy = 8.0;
    const STEP: f32 = 700.0;
    const COLS: usize = 70;
    let place = |k: usize| [1500.0 + STEP * (k % COLS) as f32, 1500.0 + STEP * (k / COLS) as f32];
    let n = l.branches.len().min(MAX_PORTRAITS);
    let mut reqs = Vec::with_capacity(2 * n);
    for (i, b) in l.branches.iter().take(n).enumerate() {
        let a = place(2 * i);
        let c = place(2 * i + 1);
        reqs.push(SpawnRequest::with_genome(a[0], a[1], &b.leader));
        reqs.push(SpawnRequest::with_genome(c[0], c[1], &reverse_complement(&b.leader)));
    }
    t.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    t.encode_steps(&gpu.queue, &mut enc, 3);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = t.read_agents_blocking(gpu);
    let t_world = clock.elapsed();
    let mut at = std::collections::HashMap::new();
    for (slot, a) in agents.iter().enumerate() {
        if a.alive != 0 {
            let key = (((a.pos_x - 1500.0) / STEP).round() as i64, ((a.pos_y - 1500.0) / STEP).round() as i64);
            at.insert(key, slot);
        }
    }
    let cap = crate::render::capture::Capture::new(gpu, &t, PORTRAIT);
    let mut out = Vec::with_capacity(l.branches.len());
    for i in 0..l.branches.len() {
        let mut pair = [Vec::new(), Vec::new()];
        if i < n {
            for (f, png) in pair.iter_mut().enumerate() {
                let k = 2 * i + f;
                if let Some(&slot) = at.get(&((k % COLS) as i64, (k / COLS) as i64)) {
                    *png = crate::report::portrait(gpu, &t, &cap, slot as u32, &agents[slot], 0.0);
                }
            }
        }
        out.push(pair);
    }
    log::info!("tree portraits: world {:.1} s, {} portraits, total {:.1} s", t_world.as_secs_f32(), 2 * n, clock.elapsed().as_secs_f32());
    out
}

const SEGMENT_LEN: f32 = 11.0;
const CHIRAL: u8 = 19;

/// Cor da classe do aminoácido (as do desenho: class_color em agents_view.wgsl).
fn class_color(aa: u8) -> [f32; 3] {
    match aa {
        0 | 7 | 9 | 10 | 17 => [0.72, 0.72, 0.62],
        4 | 18 | 19 => [0.70, 0.45, 0.95],
        15 | 16 | 11 | 13 => [0.40, 0.85, 0.45],
        1 => [0.95, 0.90, 0.30],
        8 | 14 | 6 => [0.35, 0.55, 1.00],
        2 | 3 => [1.00, 0.35, 0.30],
        5 => [0.95, 0.95, 0.95],
        _ => [1.00, 0.60, 0.20],
    }
}

/// Cor dominante de cada órgão (organ_lod_color em agents_view.wgsl).
fn organ_color(t: u8) -> [f32; 3] {
    match t {
        0 => [0.92, 0.90, 0.85],
        1 => [0.80, 0.40, 0.40],
        2 | 8 => [0.45, 1.0, 0.45],
        3 | 9 => [1.0, 0.95, 0.4],
        4 => [1.0, 0.85, 0.2],
        5 => [0.85, 0.9, 1.0],
        6 => [0.6, 0.9, 1.0],
        10 => [0.35, 0.95, 0.35],
        11 => [0.9, 0.2, 0.2],
        12 => [1.0, 0.3, 0.25],
        13 | 17 => [1.0, 0.55, 0.15],
        14 => [0.9, 0.78, 0.15],
        16 => [0.72, 0.82, 1.0],
        18 => [0.85, 0.6, 0.3],
        19 => [0.95, 0.35, 0.85],
        _ => [0.6, 0.6, 0.7],
    }
}

fn hex(c: [f32; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", (c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

/// Uma forma (corpo) em JSON: pontos na forma de repouso, normalizados a uma
/// caixa de lado 1 centrada em 0 ([x, y, raio, cor, é órgão]), a proteína em
/// letras e a lista dos órgãos por extenso.
fn form_json(genome: &[u8], w: &World, code: &[u32]) -> String {
    let body: Vec<Residue> = translate_organs(genome, w.params.require_start != 0, code);
    let mut pts: Vec<([f32; 2], f32, String, bool)> = Vec::new();
    let (mut p, mut ang, mut chir) = ([0.0f32, 0.0f32], 0.0f32, 1.0f32);
    for r in &body {
        let row = w.amino.get(r.aa as usize);
        let organ = r.organ.and_then(|(t, _, _)| w.organ_table.get(t as usize).map(|o| (t, o)));
        // Como em body.wgsl / fold.wgsl: a posição é guardada ANTES de virar.
        let rad = 0.9 + 3.0 * (row.map_or(100.0, |a| a.volume) / 130.0).powf(1.4);
        let col = match organ {
            Some((t, _)) => organ_color(t),
            None => class_color(r.aa),
        };
        pts.push((p, if organ.is_some() { rad * 2.2 } else { rad }, hex(col), organ.is_some()));
        let mut bend = row.map_or(0.0, |a| a.angulo_repouso);
        if let Some((t, o)) = organ {
            if let Some(a) = o.angulo {
                bend = a;
            }
            if t == CHIRAL {
                chir = -chir;
            }
        }
        ang += bend * chir * w.params.rest_angle_mult;
        let len = (SEGMENT_LEN * row.map_or(1.0, |a| a.comprimento / SEGMENT_LEN).max(0.1) * organ.map_or(1.0, |(_, o)| o.comprimento_mult)).max(1.0);
        p = [p[0] + ang.cos() * len, p[1] + ang.sin() * len];
    }
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for (q, r, _, _) in &pts {
        for i in 0..2 {
            lo[i] = lo[i].min(q[i] - r);
            hi[i] = hi[i].max(q[i] + r);
        }
    }
    let side = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
    let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    let mut s = String::from("{\"p\":[");
    for (i, (q, r, c, o)) in pts.iter().enumerate() {
        // Cima no ecrã = +y no mundo.
        write!(s, "{}[{:.3},{:.3},{:.3},\"{c}\",{}]", if i > 0 { "," } else { "" }, (q[0] - mid[0]) / side, -(q[1] - mid[1]) / side, r / side, *o as u8).unwrap();
    }
    let seq: String = body.iter().map(|r| r.organ.map_or(AA_LETTERS[r.aa as usize], |(t, _, _)| ORGAN_SYMBOLS[t as usize])).collect();
    let organs: String = body.iter().filter_map(|r| r.organ.map(|(t, _, _)| ORGAN_SYMBOLS[t as usize])).collect();
    let list: Vec<String> = body
        .iter()
        .enumerate()
        .filter_map(|(k, r)| r.organ.map(|(t, p, g)| js_str(&format!("{} position {k}: {}", ORGAN_SYMBOLS[t as usize], describe(t, p, g, &w.organ_table)))))
        .collect();
    // Lista de aminoácidos com as cores do desenho (órgãos a branco, a negrito).
    let mut ph = String::new();
    for r in &body {
        match r.organ {
            Some((t, _, _)) => write!(ph, "<b>{}</b>", ORGAN_SYMBOLS[t as usize]).unwrap(),
            None => write!(ph, "<span style=\"color:{}\">{}</span>", hex(class_color(r.aa)), AA_LETTERS[r.aa as usize]).unwrap(),
        }
    }
    write!(s, "],\"seq\":{},\"ph\":{},\"org\":{},\"list\":[{}],\"n\":{}}}", js_str(&seq), js_str(&ph), js_str(&organs), list.join(","), body.len()).unwrap();
    s
}

/// O retrato `f` do ramo `i` como endereço de dados (ou cadeia vazia).
fn pic(pics: Option<&[[Vec<u8>; 2]]>, i: usize, f: usize) -> String {
    match pics.and_then(|p| p.get(i)).map(|p| &p[f]) {
        Some(png) if !png.is_empty() => format!("\"data:image/png;base64,{}\"", crate::mcp::base64(png)),
        _ => "\"\"".into(),
    }
}

/// O visualizador (um bloco de HTML para pôr numa página). `height` em vh.
pub fn viewer(l: &Lineages, w: &World, pics: Option<&[[Vec<u8>; 2]]>) -> String {
    if l.censuses < 2 || l.branches.is_empty() {
        return format!(
            "<p>There are no lineages recorded in this scene yet ({} census(es)). The record is made during the run, every {} epochs, and is saved with the scene.</p>",
            l.censuses, l.every
        );
    }
    let code = crate::life::table::code_to_gpu(&w.organ_code);
    let mut data = String::from("[");
    for (i, b) in l.branches.iter().enumerate() {
        let counts: Vec<String> = b.counts.iter().map(|(e, n)| format!("[{e},{n}]")).collect();
        write!(
            data,
            "{}{{\"id\":{},\"name\":{},\"par\":{},\"born\":{},\"last\":{},\"alive\":{},\"peak\":{},\"bases\":{},\"c\":[{}],\"a\":{},\"b\":{},\"ia\":{},\"ib\":{}}}",
            if i > 0 { "," } else { "" },
            b.id,
            // Nome em latim da linhagem do líder (as duas formas partilham-no).
            js_str(&crate::names::lineage_name_in(&b.leader, w.params.require_start != 0, &code)),
            b.parent.map_or("null".to_string(), |p| p.to_string()),
            b.born,
            b.last,
            l.alive(b),
            b.peak,
            b.leader.len(),
            counts.join(","),
            form_json(&b.leader, w, &code),
            form_json(&reverse_complement(&b.leader), w, &code),
            pic(pics, i, 0),
            pic(pics, i, 1)
        )
        .unwrap();
    }
    data.push(']');
    let alive = l.branches.iter().filter(|b| l.alive(b)).count();
    format!(
        "<p>{} branches recorded in {} censuses (every {} epochs), {} alive. The horizontal axis is TIME: each node sits at the epoch in which the branch appeared and the bar in front of it lasts for as long as it existed. A curve links it to the branch it came from. Green = alive (the bar ends in an arrow); gray = extinct (the bar ends in a crossbar, at the last census in which it appeared); thicker = more agents at the peak. Dashed = the new branch appeared after the one it came from had disappeared. <b>Wheel</b> = zoom, <b>drag</b> = move, <b>click</b> = view the branch. Up close each node shows the drawing of the two forms.</p>\
<div class=\"tv-bar\"><label>hide branches with a peak below <input id=\"tv-min\" type=\"range\" min=\"0\" max=\"100\" value=\"0\"> <span id=\"tv-minv\">0</span></label> <label><input id=\"tv-alive\" type=\"checkbox\"> only the living and their ancestors</label> <button id=\"tv-fit\">see all</button></div>\
<div class=\"tv-wrap\"><canvas id=\"tv\"></canvas><div id=\"tv-info\"><i>click a branch</i></div></div>\
<script>const TV_DATA={data};const TV_T0={};const TV_T1={};const TV_EVERY={};\n{JS}</script>",
        l.branches.len(),
        l.censuses,
        l.every,
        alive,
        l.first_epoch,
        l.last_epoch,
        l.every.max(1)
    )
}

/// Página só com a árvore (gera-se num instante: não lê nada da GPU).
pub fn page(l: &Lineages, w: &World, title: &str, pics: Option<&[[Vec<u8>; 2]]>) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>{t}</title><style>body{{background:#12151a;color:#dde3ea;font:14px/1.45 system-ui,sans-serif;margin:16px}}h1{{font-size:20px}}{CSS}</style><h1>{t}</h1>{}</html>",
        viewer(l, w, pics),
        t = title.replace('<', "&lt;")
    )
}

pub const CSS: &str = ".tv-wrap{display:flex;gap:12px;align-items:stretch}\
#tv{flex:1;min-width:0;height:78vh;background:#0e1014;border:1px solid #2c333b;border-radius:8px;cursor:grab;touch-action:none}\
#tv-info{width:330px;flex:none;background:#1a1f25;border:1px solid #2c333b;border-radius:8px;padding:10px;font-size:12px;overflow:auto;max-height:78vh}\
#tv-info h4{margin:8px 0 2px;font-size:13px}#tv-info canvas{background:#0e1014;border-radius:6px}#tv-info .seq{font-family:Consolas,monospace;font-size:15px;line-height:1.5;word-break:break-all;color:#8fa3b8;background:#0e1014;border-radius:6px;padding:6px 8px;margin-top:4px}#tv-info .seq b{color:#fff}#tv-info img{border-radius:6px;display:block}\
#tv-info li{margin:2px 0;color:#aab4c0}#tv-info ul{padding-left:16px;margin:4px 0}\
.tv-bar{font-size:12px;color:#aab4c0;margin:6px 0;display:flex;gap:18px;align-items:center;flex-wrap:wrap}.tv-bar button{background:#2a313a;color:#dde3ea;border:1px solid #3a434e;border-radius:5px;padding:2px 10px;cursor:pointer}";

const JS: &str = r##"(function(){
const cv=document.getElementById('tv'),ctx=cv.getContext('2d'),info=document.getElementById('tv-info');
const byId=new Map(TV_DATA.map(b=>[b.id,b]));
const kids=new Map();TV_DATA.forEach(b=>{const k=b.par===null?-1:b.par;if(!kids.has(k))kids.set(k,[]);kids.get(k).push(b);});
let minPeak=0,onlyAlive=false,nodes=[],sel=null;
// Os ramos que saem do mesmo ramo ficam do MAIS NOVO para o mais velho por
// baixo dele: assim a descida para um ramo antigo passa a esquerda de todos
// os mais novos e nenhuma linha atravessa outra.
kids.forEach(a=>a.sort((p,q)=>q.born-p.born||q.id-p.id));
// NÓS E LINHAS NO TEMPO. x = epoch em que o ramo apareceu (um censo = STEP
// unidades); cada ramo tem a sua linha, por baixo do ramo de onde saiu. O nó
// é um cartão (retratos das duas formas + texto) e a barra à frente dele
// dura até ao último censo em que apareceu. Vista: ecrã = o + mundo * z.
const CW=262,CH=74,STEP=350,ROW=88;
let z=1,ox=0,oy=0,W=100,H=100;
const TX=t=>(t-TV_T0)/TV_EVERY*STEP;
let pend=false; function later(){ if(!pend){ pend=true; requestAnimationFrame(()=>{pend=false;draw();}); } }
function img(n,k){ const key='_'+k; if(n[key]===undefined){ if(!n[k]) n[key]=null; else { const im=new Image(); im.onload=later; im.src=n[k]; n[key]=im; } } const im=n[key]; return im&&im.complete&&im.naturalWidth?im:null; }
function layout(){
  let ok=null;
  if(onlyAlive){ ok=new Set(); TV_DATA.forEach(b=>{ if(b.alive){ let x=b; while(x&&!ok.has(x.id)){ ok.add(x.id); x=x.par===null?null:byId.get(x.par);} } }); }
  const vis=b=>b.peak>=minPeak&&(!ok||ok.has(b.id));
  nodes=[];
  // um ramo escondido passa os filhos ao antepassado visível
  const st=(kids.get(-1)||[]).map(b=>[b,null]).reverse();
  while(st.length){ const [b,up]=st.pop(); let me=up; if(vis(b)){ b.up=up; b.x=Math.max(TX(b.born),up?up.x+STEP:0); b.y=nodes.length*ROW; b.end=Math.max(b.x+CW,b.x+TX(b.last)-TX(b.born)); b.reach=b.end; nodes.push(b); me=b; } const k=kids.get(b.id); if(k) for(let i=k.length-1;i>=0;i--) st.push([k[i],me]); }
}
function fit(){ let x1=1,y1=1; for(const n of nodes){ x1=Math.max(x1,n.reach); y1=Math.max(y1,n.y); }
  z=Math.min((W-50)/x1,(H-70)/(y1+CH),1.2); ox=20; oy=44+CH/2*z; draw(); }
function resize(){ const r=cv.getBoundingClientRect(); W=r.width;H=r.height; const d=window.devicePixelRatio||1; cv.width=W*d;cv.height=H*d; }
function body(c,f,cx,cy,size){
  const p=f.p; if(!p.length) return;
  c.lineCap='round';
  for(let i=0;i+1<p.length;i++){ c.strokeStyle=p[i][3]; c.lineWidth=Math.max(1,2*Math.min(p[i][2],0.06)*size); c.beginPath(); c.moveTo(cx+p[i][0]*size,cy+p[i][1]*size); c.lineTo(cx+p[i+1][0]*size,cy+p[i+1][1]*size); c.stroke(); }
  for(const q of p){ if(q[4]){ c.fillStyle=q[3]; c.beginPath(); c.arc(cx+q[0]*size,cy+q[1]*size,Math.max(1.5,q[2]*size),0,6.2832); c.fill(); c.strokeStyle='#0e1014'; c.lineWidth=1; c.stroke(); } }
  const e=p[p.length-1]; c.fillStyle=e[3]; c.beginPath(); c.arc(cx+e[0]*size,cy+e[1]*size,Math.max(1,e[2]*size),0,6.2832); c.fill();
}
function fmtT(t){ return t<2e6? Math.round(t/1e3)+'k' : (t/1e6).toFixed(2)+'M'; }
function rr(x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }
function draw(){
  const d=window.devicePixelRatio||1;
  ctx.setTransform(d,0,0,d,0,0); ctx.clearRect(0,0,W,H);
  // eixo do tempo (no ecrã): riscas nos epochs redondos
  const perPx=TV_EVERY/(STEP*z), span=W*perPx; let stp=Math.pow(10,Math.floor(Math.log10(Math.max(1,span/6)))); if(span/stp>12) stp*=5; else if(span/stp>6) stp*=2;
  ctx.font='11px system-ui'; ctx.textAlign='center';
  for(let t=Math.ceil((TV_T0-ox*perPx)/stp)*stp;;t+=stp){ const x=ox+TX(t)*z; if(x>W) break; ctx.strokeStyle='#1c2128'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,18); ctx.lineTo(x,H); ctx.stroke(); ctx.fillStyle='#8b96a3'; ctx.fillText(stp>=1e4?fmtT(t):String(t),x,13); }
  ctx.textAlign='left';
  ctx.save(); ctx.beginPath(); ctx.rect(0,20,W,H-20); ctx.clip();
  ctx.setTransform(d*z,0,0,d*z,d*ox,d*oy);
  const vx0=-ox/z,vx1=(W-ox)/z,vy0=-oy/z,vy1=(H-oy)/z;
  // de longe os cartões passam a pontos
  const cards=CW*z>=64, text=z>=0.42;
  const wd=n=>Math.max(1.2/z,1.5+1.3*Math.log10(Math.max(1,n.peak)));
  ctx.lineCap='round';
  // ligações: saem da barra do ramo de origem, um pouco antes do nó, e descem em curva
  for(const n of nodes){ const p=n.up; if(!p) continue; if(n.y<vy0-ROW||p.y>vy1+ROW||n.x<vx0||n.x-60>vx1) continue;
    // sai da barra do ramo de origem; se este ja tinha desaparecido dos censos
    // quando o novo apareceu, sai do fim da barra, a tracejado
    const late=n.x-60>p.end+1, x0=late?p.end:n.x-60; ctx.strokeStyle=n.alive?'#4f9e62':'#56606b'; ctx.lineWidth=wd(n);
    if(late) ctx.setLineDash([6/z,5/z]);
    ctx.beginPath(); ctx.moveTo(x0,p.y); ctx.bezierCurveTo(x0+4,n.y,x0+10,n.y,x0+50,n.y); ctx.lineTo(n.x,n.y); ctx.stroke(); ctx.setLineDash([]); }
  for(const n of nodes){ if(n.y<vy0-CH||n.y>vy1+CH||n.x>vx1||n.reach<vx0) continue;
    const col=n===sel?'#ffd866':(n.alive?'#6fcf7f':'#6b7480');
    // barra da vida do ramo (e o prolongamento fino até ao último ramo que sai dele)
    ctx.strokeStyle=n.alive?'#4f9e62':'#56606b'; ctx.lineWidth=wd(n); ctx.beginPath(); ctx.moveTo(n.x,n.y); ctx.lineTo(n.end,n.y); ctx.stroke();
    // ponta da barra: seta = continua vivo; travessa = ultimo censo em que apareceu (extinto)
    if(n.end>n.x+CW+2||!cards){ const e=n.end, k=Math.max(7,5/z); if(n.alive){ ctx.fillStyle='#6fcf7f'; ctx.beginPath(); ctx.moveTo(e+k*1.6,n.y); ctx.lineTo(e,n.y-k); ctx.lineTo(e,n.y+k); ctx.fill(); } else { ctx.strokeStyle='#8a94a0'; ctx.lineWidth=Math.max(2,1.5/z); ctx.beginPath(); ctx.moveTo(e,n.y-k); ctx.lineTo(e,n.y+k); ctx.stroke(); } }
    if(!cards){ ctx.fillStyle=col; ctx.beginPath(); ctx.arc(n.x,n.y,Math.max(3/z,6+5*Math.log10(Math.max(1,n.peak))),0,6.2832); ctx.fill(); continue; }
    const y=n.y-CH/2;
    rr(n.x,y,CW,CH,10); ctx.fillStyle='#07090b'; ctx.fill(); ctx.strokeStyle=col; ctx.lineWidth=n===sel?3:1.6; ctx.stroke();
    const ia=img(n,'ia'), ib=img(n,'ib');
    if(ia) ctx.drawImage(ia,n.x+5,y+4,66,66); else body(ctx,n.a,n.x+38,n.y,52);
    if(ib) ctx.drawImage(ib,n.x+73,y+4,66,66); else body(ctx,n.b,n.x+106,n.y,52);
    if(text){ const tx=n.x+146; ctx.fillStyle=n.alive?'#e6edf3':'#9aa4af'; ctx.font='italic bold 12px system-ui'; ctx.fillText(n.name,tx,y+19,CW-152);
      ctx.font='11px system-ui'; ctx.fillStyle='#9fb0c0'; ctx.fillText('B'+n.id+' · '+n.bases+' bases · peak '+n.peak,tx,y+36,CW-152);
      ctx.fillText(fmtT(n.born)+' → '+(n.alive?'alive':fmtT(n.last)),tx,y+50,CW-152);
      ctx.fillStyle='#c8b06a'; ctx.fillText(n.a.org+' | '+n.b.org,tx,y+65,CW-152); }
  }
  ctx.restore();
}
function pick(mx,my){ const x=(mx-ox)/z,y=(my-oy)/z; const cards=CW*z>=64; let best=null,bd=1e18;
  for(const n of nodes){ if(cards){ if(x>=n.x&&x<=n.x+CW&&Math.abs(y-n.y)<=CH/2) return n; } else { const dd=(x-n.x)**2+(y-n.y)**2; if(dd<bd){bd=dd;best=n;} } }
  return (!cards&&bd<(14/z)**2)?best:null; }
function show(b){
  sel=b; if(!b){ info.innerHTML='<i>click a branch</i>'; draw(); return; }
  const nm=x=>'<i>'+x.name+'</i> (B'+x.id+')';
  const par=b.par===null?'root (no recognizable relative)':(byId.has(b.par)?nm(byId.get(b.par)):'B'+b.par);
  const kn=(kids.get(b.id)||[]).map(nm).join(', ')||'none';
  let h='<h3 style="margin:0"><i>'+b.name+'</i></h3><div>B'+b.id+(b.alive?' · alive':' · extinct')+'</div><div>'+b.bases+' bases · peak '+b.peak+' agents</div><div>appeared at epoch '+b.born+', last census '+b.last+'</div><div>comes from: '+par+'</div><div>branches that come from it: '+kn+'</div>';
  h+='<h4>population at the censuses</h4><canvas id="tv-sp" width="300" height="60"></canvas>';
  for(const [nm,f,im] of [['form A',b.a,b.ia],['form B (the complement, the children)',b.b,b.ib]]){
    h+='<h4>'+nm+' · '+f.n+' residues</h4>'+(im?'<img width="300" height="300" src="'+im+'">':'<canvas class="tv-b" width="300" height="200"></canvas>')+'<div class="seq" title="list of amino acids, from N to C; organs in white">'+f.ph+'</div><ul>'+(f.list.length?f.list.map(x=>'<li>'+x.replace(/</g,'&lt;')+'</li>').join(''):'<li>no organs</li>')+'</ul>';
  }
  info.innerHTML=h;
  let ci=0; const cs=info.querySelectorAll('canvas.tv-b'); [[b.a,b.ia],[b.b,b.ib]].forEach(([f,im])=>{ if(!im){ const c=cs[ci++].getContext('2d'); body(c,f,150,100,180); } });
  const sp=document.getElementById('tv-sp').getContext('2d'); const mx=Math.max(1,b.peak); sp.strokeStyle='#6fcf7f'; sp.lineWidth=1.5; sp.beginPath();
  b.c.forEach((q,i)=>{ const x=4+292*(q[0]-TV_T0)/Math.max(1,TV_T1-TV_T0), y=56-50*q[1]/mx; if(i) sp.lineTo(x,y); else sp.moveTo(x,y); }); sp.stroke();
  sp.fillStyle='#6fcf7f'; b.c.forEach(q=>{ const x=4+292*(q[0]-TV_T0)/Math.max(1,TV_T1-TV_T0), y=56-50*q[1]/mx; sp.beginPath(); sp.arc(x,y,2.2,0,6.3); sp.fill(); });
  draw();
}
let drag=null,moved=false;
cv.addEventListener('pointerdown',e=>{ drag=[e.clientX,e.clientY]; moved=false; cv.setPointerCapture(e.pointerId); cv.style.cursor='grabbing'; });
cv.addEventListener('pointermove',e=>{ if(!drag) return; const dx=e.clientX-drag[0],dy=e.clientY-drag[1]; if(Math.abs(dx)+Math.abs(dy)>3) moved=true; ox+=dx; oy+=dy; drag=[e.clientX,e.clientY]; draw(); });
cv.addEventListener('pointerup',e=>{ cv.style.cursor='grab'; if(drag&&!moved){ const r=cv.getBoundingClientRect(); show(pick(e.clientX-r.left,e.clientY-r.top)); } drag=null; });
cv.addEventListener('wheel',e=>{ e.preventDefault(); const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=Math.pow(1.0018,-e.deltaY);
  const nz=Math.min(Math.max(z*f,0.02),4); ox=mx-(mx-ox)*nz/z; oy=my-(my-oy)*nz/z; z=nz; draw(); },{passive:false});
const mn=document.getElementById('tv-min'),mv=document.getElementById('tv-minv');
const peaks=TV_DATA.map(b=>b.peak).sort((a,b)=>a-b);
mn.addEventListener('input',()=>{ minPeak=mn.value==0?0:peaks[Math.min(peaks.length-1,Math.floor(peaks.length*mn.value/101))]; mv.textContent=minPeak; layout(); fit(); });
document.getElementById('tv-alive').addEventListener('change',e=>{ onlyAlive=e.target.checked; layout(); fit(); });
document.getElementById('tv-fit').addEventListener('click',fit);
window.addEventListener('resize',()=>{ resize(); draw(); });
layout(); resize(); fit();
// #z=1&sel=3 no endereço: abre já aproximado nesse ramo (para testar).
const hs=new URLSearchParams(location.hash.slice(1));
if(hs.get('z')){ z=+hs.get('z'); const n=nodes[Math.min(nodes.length-1,+(hs.get('sel')||0))]; ox=W/2-(n.x+CW/2)*z; oy=H/2-n.y*z; show(n); }
})();"##;
