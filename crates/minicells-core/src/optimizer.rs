use crate::{
    fixed::clamp_parameter,
    model::{PackedModel, PARAMETER_COUNT},
};
use blake2b_simd::Params;

fn seed(parent_hash: &[u8; 32], generation: u64) -> u64 {
    let mut s = Params::new().hash_length(32).to_state();
    s.update(b"mini-cells:spsa:v1");
    s.update(parent_hash);
    s.update(&generation.to_le_bytes());
    u64::from_le_bytes(s.finalize().as_bytes()[..8].try_into().unwrap())
}

pub fn delta_at(parent_hash: &[u8; 32], generation: u64, index: usize) -> i16 {
    let mut z = seed(parent_hash, generation)
        .wrapping_add(0x9e3779b97f4a7c15u64.wrapping_mul(index as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    if (z ^ (z >> 31)) & 1 == 0 {
        -1
    } else {
        1
    }
}

pub fn candidate(
    base: &PackedModel,
    parent_hash: &[u8; 32],
    generation: u64,
    side: i8,
    perturbation_q: i16,
) -> PackedModel {
    let mut out = base.clone();
    candidate_into(
        base,
        &mut out,
        parent_hash,
        generation,
        side,
        perturbation_q,
    );
    out
}

pub fn candidate_into(
    base: &PackedModel,
    out: &mut PackedModel,
    parent_hash: &[u8; 32],
    generation: u64,
    side: i8,
    perturbation_q: i16,
) {
    for i in 0..PARAMETER_COUNT {
        out.parameters[i] = clamp_parameter(
            base.parameters[i] as i32
                + side as i32 * perturbation_q as i32 * delta_at(parent_hash, generation, i) as i32,
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateDecision {
    Keep,
    Plus,
    Minus,
}

pub fn choose_update(base_loss: i64, plus_loss: i64, minus_loss: i64) -> UpdateDecision {
    if plus_loss < base_loss && plus_loss < minus_loss {
        UpdateDecision::Plus
    } else if minus_loss < base_loss && minus_loss < plus_loss {
        UpdateDecision::Minus
    } else {
        UpdateDecision::Keep
    }
}

pub fn apply_update(
    model: &mut PackedModel,
    parent_hash: &[u8; 32],
    generation: u64,
    base_loss: i64,
    loss_plus: i64,
    loss_minus: i64,
    step_q: i16,
) -> UpdateDecision {
    let decision = choose_update(base_loss, loss_plus, loss_minus);
    let direction = match decision {
        UpdateDecision::Plus => 1,
        UpdateDecision::Minus => -1,
        UpdateDecision::Keep => return UpdateDecision::Keep,
    };
    for i in 0..PARAMETER_COUNT {
        model.parameters[i] = clamp_parameter(
            model.parameters[i] as i32
                + direction * step_q as i32 * delta_at(parent_hash, generation, i) as i32,
        );
    }
    decision
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidates_are_symmetric() {
        let m = PackedModel::default();
        let h = [1; 32];
        let p = candidate(&m, &h, 0, 1, 4);
        let n = candidate(&m, &h, 0, -1, 4);
        for i in 0..PARAMETER_COUNT {
            assert_eq!(p.parameters[i], -n.parameters[i]);
        }
    }
    #[test]
    fn better_side_drives_update() {
        let mut m = PackedModel::default();
        let h = [2; 32];
        let p = candidate(&m, &h, 0, 1, 1);
        assert_eq!(
            apply_update(&mut m, &h, 0, 100, 1, 2, 1),
            UpdateDecision::Plus
        );
        assert_eq!(m, p);
    }
    #[test]
    fn tie_does_not_change() {
        let mut m = PackedModel::default();
        assert_eq!(
            apply_update(&mut m, &[0; 32], 0, 100, 1, 1, 1),
            UpdateDecision::Keep
        );
        assert_eq!(m, PackedModel::default());
    }

    #[test]
    fn guarded_decisions_require_unique_improvement_over_base() {
        assert_eq!(choose_update(100, 80, 110), UpdateDecision::Plus);
        assert_eq!(choose_update(100, 120, 90), UpdateDecision::Minus);
        assert_eq!(choose_update(100, 110, 120), UpdateDecision::Keep);
        assert_eq!(choose_update(100, 100, 120), UpdateDecision::Keep);
        assert_eq!(choose_update(100, 80, 80), UpdateDecision::Keep);
        assert_eq!(choose_update(100, 100, 100), UpdateDecision::Keep);
    }
}
