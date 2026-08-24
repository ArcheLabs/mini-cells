pub const PAD_ID: u8 = 0;
pub const VOCAB_SIZE: usize = 44;
pub const SYMBOLS: &[u8; 43] = b"abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-";

pub fn encode_byte(value: u8) -> Option<u8> {
    SYMBOLS
        .iter()
        .position(|candidate| *candidate == value)
        .map(|index| index as u8 + 1)
}

pub fn decode_id(id: u8) -> Option<u8> {
    if id == PAD_ID {
        None
    } else {
        SYMBOLS.get(id as usize - 1).copied()
    }
}

pub fn encode_text(text: &[u8], output: &mut [u8; 64]) -> Result<usize, ()> {
    if text.len() > 32 {
        return Err(());
    }
    output.fill(PAD_ID);
    for (index, value) in text.iter().enumerate() {
        output[index] = encode_byte(*value).ok_or(())?;
    }
    Ok(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vocabulary_is_frozen() {
        assert_eq!(VOCAB_SIZE, 44);
        assert_eq!(encode_byte(b'a'), Some(1));
        assert_eq!(encode_byte(b'-'), Some(43));
        assert_eq!(decode_id(43), Some(b'-'));
        assert_eq!(encode_byte(b'A'), None);
    }
}
