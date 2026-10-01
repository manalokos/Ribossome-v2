// Hash inteiro sem estado (o mesmo do v3). Semente = slot ^ epoch ^ seed.
fn hash(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u);
    return x;
}

fn hash_f32(v: u32) -> f32 { return f32(hash(v)) / 4294967295.0; }
