// LIVRO-RAZÃO: soma exata da matéria livre por canal e estado.
// Uma soma parcial por workgroup (atómicos partilhados) e depois um único
// atomicAdd global por contador. O buffer `ledger` é limpo antes, no CPU.

var<workgroup> wg_ledger: array<atomic<u32>, 8>;

@compute @workgroup_size(16, 16)
fn ledger_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if (lid < 8u) { atomicStore(&wg_ledger[lid], 0u); }
    workgroupBarrier();
    if (gid.x < GRID_SIZE && gid.y < GRID_SIZE) {
        let idx = gid.y * GRID_SIZE + gid.x;
        for (var ch = 0u; ch < 4u; ch++) {
            let v = atomicLoad(&chem_grid[idx * 4u + ch]);
            atomicAdd(&wg_ledger[ch], v & CHEM_STATE_MASK);
            atomicAdd(&wg_ledger[4u + ch], v >> 16u);
        }
    }
    workgroupBarrier();
    if (lid < 8u) {
        let s = atomicLoad(&wg_ledger[lid]);
        if (s > 0u) { atomicAdd(&ledger[lid], s); }
    }
}
