use crate::{
    codec::{Reader, Writer},
    work::Op,
    Error,
};
pub const RESULT_MAGIC: [u8; 4] = *b"MCR1";
pub const RESULT_VERSION: u8 = 1;
pub const STATUS_OK: u16 = 0;
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultBody {
    Training {
        side: i8,
        loss: i64,
        correct_tokens: u32,
        total_tokens: u32,
        eval_digest: [u8; 32],
    },
    Inference {
        input_len: u8,
        output_len: u8,
        input: [u8; 32],
        output: [u8; 32],
        matching_tokens: u8,
    },
    Status,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefineResult {
    pub op: Op,
    pub status: u16,
    pub request_id: u64,
    pub generation: u64,
    pub model_hash: [u8; 32],
    pub body: ResultBody,
}
impl RefineResult {
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(out);
        w.bytes(&RESULT_MAGIC)?;
        w.u8(RESULT_VERSION)?;
        w.u8(self.op as u8)?;
        w.u16(self.status)?;
        w.u64(self.request_id)?;
        w.u64(self.generation)?;
        w.bytes(&self.model_hash)?;
        match &self.body {
            ResultBody::Training {
                side,
                loss,
                correct_tokens,
                total_tokens,
                eval_digest,
            } => {
                w.i8(*side)?;
                w.bytes(&[0; 7])?;
                w.i64(*loss)?;
                w.u32(*correct_tokens)?;
                w.u32(*total_tokens)?;
                w.bytes(eval_digest)?
            }
            ResultBody::Inference {
                input_len,
                output_len,
                input,
                output,
                matching_tokens,
            } => {
                w.u8(*input_len)?;
                w.u8(*output_len)?;
                w.u16(0)?;
                w.bytes(input)?;
                w.bytes(output)?;
                w.u8(*matching_tokens)?
            }
            ResultBody::Status => {}
        }
        Ok(w.len())
    }
    pub fn decode(input: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(input);
        if r.bytes::<4>()? != RESULT_MAGIC || r.u8() != Ok(RESULT_VERSION) {
            return Err(Error::Invalid);
        }
        let op = Op::try_from(r.u8()?)?;
        let status = r.u16()?;
        let request_id = r.u64()?;
        let generation = r.u64()?;
        let model_hash = r.bytes()?;
        let body = match op {
            Op::TrainPlus | Op::TrainMinus => {
                let side = r.i8()?;
                if r.bytes::<7>()? != [0; 7] || ![-1, 1].contains(&side) {
                    return Err(Error::Invalid);
                }
                ResultBody::Training {
                    side,
                    loss: r.i64()?,
                    correct_tokens: r.u32()?,
                    total_tokens: r.u32()?,
                    eval_digest: r.bytes()?,
                }
            }
            Op::Infer => ResultBody::Inference {
                input_len: r.u8()?,
                output_len: r.u8()?,
                input: {
                    let _ = r.u16()?;
                    r.bytes()?
                },
                output: r.bytes()?,
                matching_tokens: r.u8()?,
            },
            Op::StatusProbe => ResultBody::Status,
        };
        if !r.done() {
            return Err(Error::Invalid);
        }
        Ok(Self {
            op,
            status,
            request_id,
            generation,
            model_hash,
            body,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn training_codec() {
        let x = RefineResult {
            op: Op::TrainPlus,
            status: 0,
            request_id: 1,
            generation: 2,
            model_hash: [3; 32],
            body: ResultBody::Training {
                side: 1,
                loss: 4,
                correct_tokens: 5,
                total_tokens: 6,
                eval_digest: [7; 32],
            },
        };
        let mut b = [0; 160];
        let n = x.encode_into(&mut b).unwrap();
        assert_eq!(RefineResult::decode(&b[..n]), Ok(x));
    }
}
