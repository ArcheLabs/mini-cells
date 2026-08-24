fn main() {
    let (_, vector, _) = minicells_sim::run_generation_zero();
    println!("{}", serde_json::to_string_pretty(&vector).unwrap());
}
