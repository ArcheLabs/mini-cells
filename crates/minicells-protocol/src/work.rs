use crate::{
    codec::{Reader, Writer},
    Error,
};
pub const WORK_MAGIC: [u8; 4] = *b"MCW1";
pub const WORK_VERSION: u8 = 1;
pub const USE_CURRENT_GENERATION: u64 = u64::MAX;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Op {
    Infer = 1,
    TrainPlus = 2,
    TrainMinus = 3,
    StatusProbe = 4,
}
impl TryFrom<u8> for Op {
    type Error = Error;
    fn try_from(v: u8) -> Result<Self, Error> {
        match v {
            1 => Ok(Self::Infer),
            2 => Ok(Self::TrainPlus),
            3 => Ok(Self::TrainMinus),
            4 => Ok(Self::StatusProbe),
            _ => Err(Error::Invalid),
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkBody {
    Infer {
        expected_generation: u64,
        text_len: u8,
        text: [u8; 32],
    },
    Train {
        generation: u64,
        parent_model_hash: [u8; 32],
    },
    StatusProbe,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkPayload {
    pub op: Op,
    pub flags: u16,
    pub request_id: u64,
    pub body: WorkBody,
}
impl WorkPayload {
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(out);
        w.bytes(&WORK_MAGIC)?;
        w.u8(WORK_VERSION)?;
        w.u8(self.op as u8)?;
        w.u16(self.flags)?;
        w.u64(self.request_id)?;
        match &self.body {
            WorkBody::Infer {
                expected_generation,
                text_len,
                text,
            } => {
                if *text_len as usize > 32 {
                    return Err(Error::Invalid);
                }
                w.u64(*expected_generation)?;
                w.u8(*text_len)?;
                w.bytes(&text[..*text_len as usize])?
            }
            WorkBody::Train {
                generation,
                parent_model_hash,
            } => {
                w.u64(*generation)?;
                w.bytes(parent_model_hash)?
            }
            WorkBody::StatusProbe => {}
        }
        Ok(w.len())
    }
    pub fn decode(input: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(input);
        if r.bytes::<4>()? != WORK_MAGIC || r.u8() != Ok(WORK_VERSION) {
            return Err(Error::Invalid);
        }
        let op = Op::try_from(r.u8()?)?;
        let flags = r.u16()?;
        let request_id = r.u64()?;
        let body = match op {
            Op::Infer => {
                let expected_generation = r.u64()?;
                let text_len = r.u8()?;
                if text_len > 32 {
                    return Err(Error::Invalid);
                }
                let mut text = [0; 32];
                text[..text_len as usize].copy_from_slice(r.slice(text_len as usize)?);
                WorkBody::Infer {
                    expected_generation,
                    text_len,
                    text,
                }
            }
            Op::TrainPlus | Op::TrainMinus => WorkBody::Train {
                generation: r.u64()?,
                parent_model_hash: r.bytes()?,
            },
            Op::StatusProbe => WorkBody::StatusProbe,
        };
        if !r.done() {
            return Err(Error::Invalid);
        }
        Ok(Self {
            op,
            flags,
            request_id,
            body,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn codec_round_trip() {
        let mut text = [0; 32];
        text[..5].copy_from_slice(b"hello");
        let x = WorkPayload {
            op: Op::Infer,
            flags: 0,
            request_id: 9,
            body: WorkBody::Infer {
                expected_generation: u64::MAX,
                text_len: 5,
                text,
            },
        };
        let mut b = [0; 96];
        let n = x.encode_into(&mut b).unwrap();
        assert_eq!(WorkPayload::decode(&b[..n]), Ok(x));
    }
}
