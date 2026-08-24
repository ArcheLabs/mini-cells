pub const META: &[u8] = b"mc:v1:meta";
pub const MODEL: &[u8] = b"mc:v1:model";
pub const PENDING_PLUS: &[u8] = b"mc:v1:pending:plus";
pub const PENDING_MINUS: &[u8] = b"mc:v1:pending:minus";
pub fn history_key(slot: u8) -> [u8; 16] {
    let mut key = *b"mc:v1:history:00";
    key[14] = b'0' + (slot % 64) / 10;
    key[15] = b'0' + (slot % 64) % 10;
    key
}
pub fn inference_key(slot: u8) -> [u8; 14] {
    let mut key = *b"mc:v1:infer:00";
    key[12] = b'0' + (slot % 16) / 10;
    key[13] = b'0' + (slot % 16) % 10;
    key
}
pub fn inference_slot(request_id: u64) -> u8 {
    let mut x = request_id;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    (x & 15) as u8
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_are_bounded() {
        assert_eq!(&history_key(63), b"mc:v1:history:63");
        assert_eq!(&inference_key(15), b"mc:v1:infer:15");
    }
}
