pub const Q_SCALE: i32 = 256;
pub const PARAM_MIN: i16 = -2048;
pub const PARAM_MAX: i16 = 2048;
pub const STATE_MIN: i16 = -256;
pub const STATE_MAX: i16 = 256;

pub fn round_q16_16_to_q8_8(value: i64) -> i32 {
    if value >= 0 {
        ((value.saturating_add(128)) / 256).min(i32::MAX as i64) as i32
    } else {
        -((value.saturating_neg().saturating_add(128)) / 256).min(i32::MAX as i64) as i32
    }
}

pub fn clamp_parameter(value: i32) -> i16 {
    value.clamp(PARAM_MIN as i32, PARAM_MAX as i32) as i16
}

pub fn clamp_state(value: i32) -> i16 {
    value.clamp(STATE_MIN as i32, STATE_MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_explicit_and_symmetric() {
        assert_eq!(round_q16_16_to_q8_8(127), 0);
        assert_eq!(round_q16_16_to_q8_8(128), 1);
        assert_eq!(round_q16_16_to_q8_8(-127), 0);
        assert_eq!(round_q16_16_to_q8_8(-128), -1);
        assert_eq!(round_q16_16_to_q8_8(256), 1);
    }

    #[test]
    fn saturation_uses_v1_bounds() {
        assert_eq!(clamp_parameter(9999), 2048);
        assert_eq!(clamp_parameter(-9999), -2048);
        assert_eq!(clamp_state(300), 256);
        assert_eq!(clamp_state(-300), -256);
    }
}
