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
        .filter_map(|(k, r)| r.organ.map(|(t, p, g)| js_str(&format!("{} posição {k}: {}", ORGAN_SYMBOLS[t as usize], describe(t, p, g, &w.organ_table)))))
        .collect();
    write!(s, "],\"seq\":{},\"org\":{},\"list\":[{}],\"n\":{}}}", js_str(&seq), js_str(&organs), list.join(","), body.len()).unwrap();
    s
}

/// O visualizador (um bloco de HTML para pôr numa página). `height` em vh.
pub fn viewer(l: &Lineages, w: &World) -> String {
    if l.censuses < 2 || l.branches.is_empty() {
        return format!(
            "<p>Ainda não há linhagens registadas nesta cena ({} censo(s)). O registo faz-se durante a corrida, de {} em {} epochs, e fica guardado com a cena.</p>",
            l.censuses, l.every, l.every
        );
    }
    let code = crate::life::table::code_to_gpu(&w.organ_code);
    let mut data = String::from("[");
    for (i, b) in l.branches.iter().enumerate() {
        let counts: Vec<String> = b.counts.iter().map(|(e, n)| format!("[{e},{n}]")).collect();
        write!(
            data,
            "{}{{\"id\":{},\"par\":{},\"born\":{},\"last\":{},\"alive\":{},\"peak\":{},\"bases\":{},\"c\":[{}],\"a\":{},\"b\":{}}}",
            if i > 0 { "," } else { "" },
            b.id,
            b.parent.map_or("null".to_string(), |p| p.to_string()),
            b.born,
            b.last,
            l.alive(b),
            b.peak,
            b.leader.len(),
            counts.join(","),
            form_json(&b.leader, w, &code),
            form_json(&reverse_complement(&b.leader), w, &code)
        )
        .unwrap();
    }
    data.push(']');
    let alive = l.branches.iter().filter(|b| l.alive(b)).count();
    format!(
        "<p>{} ramos registados em {} censos (de {} em {} epochs), {} vivos. Cada nó é um ramo, ligado àquele de onde saiu; verde = vivo, cinzento = extinto, linha mais grossa = mais agentes no pico. <b>Roda</b> = zoom, <b>arrastar</b> = mover, <b>clique</b> = ver o ramo. De perto cada nó mostra o desenho das duas formas.</p>\
<div class=\"tv-bar\"><label>esconder ramos com pico abaixo de <input id=\"tv-min\" type=\"range\" min=\"0\" max=\"100\" value=\"0\"> <span id=\"tv-minv\">0</span></label> <label><input id=\"tv-alive\" type=\"checkbox\"> só os vivos e os seus antepassados</label> <button id=\"tv-fit\">ver tudo</button></div>\
<div class=\"tv-wrap\"><canvas id=\"tv\"></canvas><div id=\"tv-info\"><i>clica num ramo</i></div></div>\
<script>const TV_DATA={data};const TV_T0={};const TV_T1={};\n{JS}</script>",
        l.branches.len(),
        l.censuses,
        l.every,
        l.every,
        alive,
        l.first_epoch,
        l.last_epoch
    )
}

/// Página só com a árvore (gera-se num instante: não lê nada da GPU).
pub fn page(l: &Lineages, w: &World, title: &str) -> String {
    format!(
        "<!doctype html><html lang=\"pt\"><meta charset=\"utf-8\"><title>{t}</title><style>body{{background:#12151a;color:#dde3ea;font:14px/1.45 system-ui,sans-serif;margin:16px}}h1{{font-size:20px}}{CSS}</style><h1>{t}</h1>{}</html>",
        viewer(l, w),
        t = title.replace('<', "&lt;")
    )
}

pub const CSS: &str = ".tv-wrap{display:flex;gap:12px;align-items:stretch}\
#tv{flex:1;min-width:0;height:78vh;background:#0e1014;border:1px solid #2c333b;border-radius:8px;cursor:grab;touch-action:none}\
#tv-info{width:330px;flex:none;background:#1a1f25;border:1px solid #2c333b;border-radius:8px;padding:10px;font-size:12px;overflow:auto;max-height:78vh}\
#tv-info h4{margin:8px 0 2px;font-size:13px}#tv-info canvas{background:#0e1014;border-radius:6px}#tv-info .seq{font-family:Consolas,monospace;word-break:break-all;color:#8fa3b8}\
#tv-info li{margin:2px 0;color:#aab4c0}#tv-info ul{padding-left:16px;margin:4px 0}\
.tv-bar{font-size:12px;color:#aab4c0;margin:6px 0;display:flex;gap:18px;align-items:center;flex-wrap:wrap}.tv-bar button{background:#2a313a;color:#dde3ea;border:1px solid #3a434e;border-radius:5px;padding:2px 10px;cursor:pointer}";

const JS: &str = r##"(function(){
const cv=document.getElementById('tv'),ctx=cv.getContext('2d'),info=document.getElementById('tv-info');
const byId=new Map(TV_DATA.map(b=>[b.id,b]));
const kids=new Map();TV_DATA.forEach(b=>{const k=b.par===null?-1:b.par;if(!kids.has(k))kids.set(k,[]);kids.get(k).push(b);});
let minPeak=0,onlyAlive=false,nodes=[],sel=null;
// NÓS E LINHAS: cada ramo é um cartão (desenho das duas formas + texto),
// ligado ao ramo de onde saiu por uma curva. As raízes saem de um nó
// "origem". Vista: ecrã = o + mundo * z (zoom igual nos dois eixos).
const CW=236,CH=74,COL=330,ROW=92;
const ROOT={id:-1,root:true,alive:true,peak:1,x:0,y:0,kids:[]};
let z=1,ox=0,oy=0,W=100,H=100;
function layout(){
  let ok=null;
  if(onlyAlive){ ok=new Set(); TV_DATA.forEach(b=>{ if(b.alive){ let x=b; while(x&&!ok.has(x.id)){ ok.add(x.id); x=x.par===null?null:byId.get(x.par);} } }); }
  const vis=b=>b.peak>=minPeak&&(!ok||ok.has(b.id));
  nodes=[ROOT]; ROOT.kids=[];
  // filhos visíveis: um ramo escondido passa os filhos ao antepassado visível
  const st=(kids.get(-1)||[]).map(b=>[b,ROOT]).reverse();
  while(st.length){ const [b,up]=st.pop(); let me=up; if(vis(b)){ b.kids=[]; b.up=up; b.depth=up.root?1:up.depth+1; up.kids.push(b); nodes.push(b); me=b; } const k=kids.get(b.id); if(k) for(let i=k.length-1;i>=0;i--) st.push([k[i],me]); }
  ROOT.depth=0;
  // y: folhas em fila, cada pai a meio dos filhos (pós-ordem sem recursão)
  let next=0; const post=[]; const s2=[ROOT];
  while(s2.length){ const n=s2.pop(); post.push(n); for(const k of n.kids) s2.push(k); }
  for(let i=post.length-1;i>=0;i--){ const n=post[i]; n.x=n.depth*COL; if(!n.kids.length){ n.y=next*ROW; next++; n.live=n.alive; } else { n.y=(n.kids[0].y+n.kids[n.kids.length-1].y)/2; n.live=n.alive||n.kids.some(k=>k.live); } }
}
function fit(){ let x1=0,y0=1e9,y1=-1e9; for(const n of nodes){ x1=Math.max(x1,n.x+CW); y0=Math.min(y0,n.y-CH/2); y1=Math.max(y1,n.y+CH/2); }
  z=Math.min((W-40)/(x1+60),(H-40)/Math.max(1,y1-y0),1.2); ox=20+30*z; oy=H/2-(y0+y1)/2*z; draw(); }
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
  ctx.setTransform(d*z,0,0,d*z,d*ox,d*oy);
  const vx0=-ox/z,vx1=(W-ox)/z,vy0=-oy/z,vy1=(H-oy)/z;
  // de longe os cartões passam a pontos (maiores para os ramos com mais gente)
  const cards=CW*z>=70, text=z>=0.42;
  const outX=n=>n.root?26:(cards?n.x+CW:n.x+8);
  // linhas
  ctx.lineCap='round';
  for(const n of nodes){ if(n.root) continue; const p=n.up; if(Math.max(n.y,p.y)<vy0-50||Math.min(n.y,p.y)>vy1+50||p.x>vx1||n.x<vx0-COL) continue;
    const x0=outX(p),x1=n.x-(cards?0:8),m=(x0+x1)/2;
    ctx.strokeStyle=n.live?'#4f9e62':'#48515c'; ctx.lineWidth=Math.max(1.2/z,1.5+1.3*Math.log10(Math.max(1,n.peak)));
    ctx.beginPath(); ctx.moveTo(x0,p.y); ctx.bezierCurveTo(m,p.y,m,n.y,x1,n.y); ctx.stroke(); }
  // nós
  for(const n of nodes){ if(n.y<vy0-CH||n.y>vy1+CH||n.x>vx1||n.x+CW<vx0) continue;
    if(n.root){ ctx.fillStyle='#c9a227'; ctx.beginPath(); ctx.arc(8,n.y,18,0,6.2832); ctx.fill(); if(z>=0.3){ ctx.fillStyle='#12151a'; ctx.font='bold 9px system-ui'; ctx.textAlign='center'; ctx.fillText('origem',8,n.y+3); ctx.textAlign='left'; } continue; }
    const col=n===sel?'#ffd866':(n.alive?'#6fcf7f':'#6b7480');
    if(!cards){ ctx.fillStyle=col; ctx.beginPath(); ctx.arc(n.x,n.y,Math.max(3/z,6+5*Math.log10(Math.max(1,n.peak))),0,6.2832); ctx.fill(); continue; }
    const y=n.y-CH/2;
    rr(n.x,y,CW,CH,10); ctx.fillStyle=n.alive?'#182219':'#1a1e24'; ctx.fill(); ctx.strokeStyle=col; ctx.lineWidth=n===sel?3:1.6; ctx.stroke();
    body(ctx,n.a,n.x+32,n.y,52); body(ctx,n.b,n.x+90,n.y,52);
    if(text){ const tx=n.x+124; ctx.fillStyle=n.alive?'#e6edf3':'#9aa4af'; ctx.font='bold 14px system-ui'; ctx.fillText('R'+n.id,tx,y+20);
      ctx.font='11px system-ui'; ctx.fillStyle='#9fb0c0'; ctx.fillText(n.bases+' bases · pico '+n.peak,tx,y+36,CW-130);
      ctx.fillText(fmtT(n.born)+' → '+(n.alive?'vivo':fmtT(n.last)),tx,y+50,CW-130);
      ctx.fillStyle='#c8b06a'; ctx.fillText(n.a.org+' | '+n.b.org,tx,y+65,CW-130); }
  }
}
function pick(mx,my){ const x=(mx-ox)/z,y=(my-oy)/z; const cards=CW*z>=70; let best=null,bd=1e18;
  for(const n of nodes){ if(n.root) continue; if(cards){ if(x>=n.x&&x<=n.x+CW&&Math.abs(y-n.y)<=CH/2) return n; } else { const dd=(x-n.x)**2+(y-n.y)**2; if(dd<bd){bd=dd;best=n;} } }
  return (!cards&&bd<(14/z)**2)?best:null; }
function show(b){
  sel=b; if(!b||b.root){ sel=null; info.innerHTML='<i>clica num ramo</i>'; draw(); return; }
  const par=b.par===null?'raiz (sem parente reconhecível)':'R'+b.par;
  const kn=(kids.get(b.id)||[]).map(k=>'R'+k.id).join(', ')||'nenhum';
  let h='<h3 style="margin:0">R'+b.id+(b.alive?' · vivo':' · extinto')+'</h3><div>'+b.bases+' bases · pico '+b.peak+' agentes</div><div>apareceu ao epoch '+b.born+', último censo '+b.last+'</div><div>sai de: '+par+'</div><div>ramos que saem dele: '+kn+'</div>';
  h+='<h4>população nos censos</h4><canvas id="tv-sp" width="300" height="60"></canvas>';
  for(const [nm,f] of [['forma A',b.a],['forma B (o complemento, os filhos)',b.b]]){
    h+='<h4>'+nm+' · '+f.n+' resíduos</h4><canvas class="tv-b" width="300" height="200"></canvas><div class="seq">'+f.seq+'</div><ul>'+(f.list.length?f.list.map(x=>'<li>'+x.replace(/</g,'&lt;')+'</li>').join(''):'<li>sem órgãos</li>')+'</ul>';
  }
  info.innerHTML=h;
  const cs=info.querySelectorAll('canvas.tv-b'); [b.a,b.b].forEach((f,i)=>{ const c=cs[i].getContext('2d'); body(c,f,150,100,180); });
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
if(hs.get('z')){ z=+hs.get('z'); const n=nodes[Math.min(nodes.length-1,1+(+(hs.get('sel')||0)))]; ox=W/2-(n.x+CW/2)*z; oy=H/2-n.y*z; show(n); }
})();"##;
