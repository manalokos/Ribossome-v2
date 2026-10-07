//! Ligações entre agentes por âncoras: cada agente tem uma âncora +
//! permanente numa ponta e uma − na outra (filamentos possíveis), na piscina
//! do laboratório. Conta as ligações (e as de nascimento, com PAIR > 0) e
//! confirma que são simétricas (A->B com (i, j) <=> B->A com (j, i)).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{BOND_STRIDE, MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 20_000, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 3);
    world.configure_lab();
    world.seed_lab(&gpu, 3, 6.0);
    world.params.pairing_rate = std::env::var("PAIR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.spawn_energy = 200.0;
    world.params.pairing_cost = 0.0;
    // Genoma = X + complementar invertido de X: o filho (lido da cadeia
    // complementar) tem o mesmo corpo que o pai.
    // Y (UAU) + S (UCU) = âncora variante 0 (+, permanente); Y + P (CCU) = variante 1 (−).
    // SOLO=1: só a âncora + (nenhuma −): antes de as âncoras se agarrarem a
    // qualquer resíduo, estes corpos nunca se ligavam ao tocar.
    let text = if std::env::var("SOLO").is_ok() {
        format!("AUG UAU UCU {} UAA", "GGU ".repeat(8))
    } else {
        format!("AUG UAU UCU {} UAU CCU UAA", "GGU ".repeat(6))
    };
    let x: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let mut g = x.clone();
    g.extend(x.iter().rev().map(|b| b ^ 1));
    let s = cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    // Em montinhos para se tocarem.
    let reqs: Vec<SpawnRequest> = (0..2000)
        .map(|i| {
            let (cx, cy) = (0.2 + 0.6 * ((i / 20) % 10) as f32 / 10.0, 0.2 + 0.6 * (i / 200) as f32 / 10.0);
            SpawnRequest::with_genome(s * (cx + 0.01 * rng.f32()), s * (cy + 0.01 * rng.f32()), &g)
        })
        .collect();
    world.request_seeds(&reqs);
    // SHARE: partilha de matéria pela ligação. Nas duas últimas voltas a
    // captura do meio é desligada (FREEZE=1), para as cópias só poderem
    // mudar por passagem entre parceiros.
    let share: f32 = std::env::var("SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let freeze = std::env::var("FREEZE").is_ok();
    world.params.bond_matter_share = share;
    let pair = world.params.pairing_rate;
    let mut before: Vec<(u32, u32)> = Vec::new();
    for round in 0..4 {
        if freeze && round >= 2 {
            world.params.pairing_rate = 0.0;
        } else {
            world.params.pairing_rate = pair;
        }
        let mut done = 0;
        while done < 500 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += MAX_STEPS_PER_FRAME;
        }
        let agents = world.read_agents_blocking(&gpu);
        let raw: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.bonds_buf)).to_vec();
        let bond = |slot: usize, i: usize| -> [u32; 4] {
            let o = (slot * BOND_STRIDE as usize + i) * 4;
            [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]
        };
        let (mut total, mut asym, mut with, mut birth) = (0, 0, 0, 0);
        for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            let mut has = false;
            for i in 0..4 {
                let b = bond(slot, i);
                if b[0] == u32::MAX || agents[b[0] as usize].alive == 0 || agents[b[0] as usize].id != b[1] {
                    continue;
                }
                has = true;
                total += 1;
                birth += (b[2] >> 16 != 0) as u32;
                let swapped = ((b[2] >> 8) & 0xFF) | ((b[2] & 0xFF) << 8) | (b[2] & 0xFFFF_0000);
                let back = (0..4).any(|j| {
                    let c = bond(b[0] as usize, j);
                    c[0] == slot as u32 && c[1] == a.id && c[2] == swapped
                });
                if !back {
                    asym += 1;
                }
            }
            with += has as u32;
        }
        let alive = agents.iter().filter(|a| a.alive != 0).count();
        // Cópias: total de complementos capturados e quantos agentes (o
        // mesmo id no mesmo slot) mudaram desde a volta anterior.
        let now: Vec<(u32, u32)> = agents.iter().map(|a| (if a.alive != 0 { a.id } else { 0 }, a.pair_count)).collect();
        let changed = now.iter().zip(&before).filter(|(n, b)| n.0 != 0 && n.0 == b.0 && n.1 != b.1).count();
        let copied: u64 = now.iter().filter(|n| n.0 != 0).map(|n| n.1 as u64).sum();
        before = now;
        let l = world.ledger_blocking(&gpu);
        println!(
            "    matéria por canal {:?} (em cópias {:?}); complementos {copied}, agentes com a cópia mudada {changed}",
            (0..4).map(|c| l.channel(c)).collect::<Vec<_>>(),
            l.held
        );
        println!(
            "{} passos: {alive} agentes, {with} com ligações, {total} pontas de ligação ({birth} de nascimento), {asym} sem par",
            (round + 1) * 500
        );
    }
}
