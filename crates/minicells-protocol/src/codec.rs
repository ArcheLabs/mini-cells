use crate::Error;

pub struct Writer<'a> {
    output: &'a mut [u8],
    offset: usize,
}
impl<'a> Writer<'a> {
    pub fn new(output: &'a mut [u8]) -> Self {
        Self { output, offset: 0 }
    }
    pub fn len(&self) -> usize {
        self.offset
    }
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }
    pub fn bytes(&mut self, value: &[u8]) -> Result<(), Error> {
        let end = self
            .offset
            .checked_add(value.len())
            .ok_or(Error::Overflow)?;
        let target = self
            .output
            .get_mut(self.offset..end)
            .ok_or(Error::Truncated)?;
        target.copy_from_slice(value);
        self.offset = end;
        Ok(())
    }
    pub fn u8(&mut self, v: u8) -> Result<(), Error> {
        self.bytes(&[v])
    }
    pub fn i8(&mut self, v: i8) -> Result<(), Error> {
        self.u8(v as u8)
    }
    pub fn u16(&mut self, v: u16) -> Result<(), Error> {
        self.bytes(&v.to_le_bytes())
    }
    pub fn i16(&mut self, v: i16) -> Result<(), Error> {
        self.bytes(&v.to_le_bytes())
    }
    pub fn u32(&mut self, v: u32) -> Result<(), Error> {
        self.bytes(&v.to_le_bytes())
    }
    pub fn u64(&mut self, v: u64) -> Result<(), Error> {
        self.bytes(&v.to_le_bytes())
    }
    pub fn i64(&mut self, v: i64) -> Result<(), Error> {
        self.bytes(&v.to_le_bytes())
    }
}

pub struct Reader<'a> {
    input: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self { input, offset: 0 }
    }
    pub fn done(&self) -> bool {
        self.offset == self.input.len()
    }
    pub fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let end = self.offset.checked_add(N).ok_or(Error::Overflow)?;
        let source = self.input.get(self.offset..end).ok_or(Error::Truncated)?;
        let mut out = [0; N];
        out.copy_from_slice(source);
        self.offset = end;
        Ok(out)
    }
    pub fn slice(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.offset.checked_add(n).ok_or(Error::Overflow)?;
        let source = self.input.get(self.offset..end).ok_or(Error::Truncated)?;
        self.offset = end;
        Ok(source)
    }
    pub fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.bytes::<1>()?[0])
    }
    pub fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.u8()? as i8)
    }
    pub fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }
    pub fn i16(&mut self) -> Result<i16, Error> {
        Ok(i16::from_le_bytes(self.bytes()?))
    }
    pub fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    pub fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    pub fn i64(&mut self) -> Result<i64, Error> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }
}
