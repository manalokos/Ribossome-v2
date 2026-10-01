// Terreno (gamma): quanta inteiros por célula do ambiente.
// >= GAMMA_SOLID_THRESHOLD = rocha sólida; 1–2 = entulho.

const GAMMA_SOLID_THRESHOLD: u32 = 3u;

fn gamma_count(cell: u32) -> u32 {
    return atomicLoad(&gamma_grid[cell]);
}
