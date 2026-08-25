//! Deterministic JSONL -> Cells dataset compiler.
//!
//! The compiler is intentionally small and dependency-light.  Its output is
//! canonical JSON in `dataset.bin` (the extension is kept for the stable lab
//! interface), plus a human-readable manifest and rejected-record log.

use blake2b_simd::Params;
use minicells_core::{
    batch::{EchoBatch, MAX_BATCH},
    vocab::encode_byte,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufRead, BufReader, Write},
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

pub const DATASET_FORMAT: &str = "minicells.dataset.v1";
pub const MAX_SEGMENT_BYTES: usize = 32;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Sample {
    pub hash: String,
    pub text: String,
    pub bytes: Vec<u8>,
    pub split: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Dataset {
    pub format: String,
    pub samples: Vec<Sample>,
    pub sample_count: usize,
    pub dataset_root: String,
    pub merkle_root: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Rejection {
    pub line: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct BatchSelection {
    pub dataset_root: String,
    pub parent_model_hash: String,
    pub generation: u64,
    pub indices: Vec<usize>,
    pub sample_hashes: Vec<String>,
    pub digest: String,
    pub witness: String,
    pub proofs: Vec<MerkleProof>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct MerkleProof {
    pub leaf: String,
    pub leaf_index: usize,
    pub siblings: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("dataset has no train samples")]
    EmptyTrain,
    #[error("dataset contains fewer than four train samples")]
    TooSmall,
    #[error("invalid sample: {0}")]
    Invalid(String),
}

fn hash(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut state = Params::new().hash_length(32).to_state();
    state.update(domain);
    for part in parts {
        state.update(&(part.len() as u64).to_le_bytes());
        state.update(part);
    }
    let mut out = [0; 32];
    out.copy_from_slice(state.finalize().as_bytes());
    out
}

fn sample_hash(bytes: &[u8]) -> [u8; 32] {
    hash(b"mini-cells:dataset-sample:v1", &[bytes])
}

/// NFKC + Unicode lowercase + whitespace collapse followed by the fixed Cells
/// ASCII punctuation mapping.  Keeping this function pure makes normalization
/// easy to audit and guarantees identical results across operating systems.
pub fn normalize_text(input: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut pending_space = false;
    for c in input.nfkc().flat_map(char::to_lowercase) {
        let mapped = match c {
            '\u{2018}' | '\u{2019}' | '\u{201b}' | '\u{2032}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201f}' => '\'',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
            c if c.is_whitespace() => {
                pending_space = true;
                continue;
            }
            c => c,
        };
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        pending_space = false;
        if mapped.is_ascii() && minicells_core::vocab::SYMBOLS.contains(&(mapped as u8)) {
            out.push(mapped);
        } else {
            return Err(format!("unsupported character U+{:04X}", mapped as u32));
        }
    }
    Ok(out.trim().to_owned())
}

pub fn segment_text(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let remaining = bytes.len() - start;
        let mut end = (start + remaining.min(MAX_SEGMENT_BYTES)).min(bytes.len());
        if end < bytes.len() {
            if let Some(relative) = bytes[start..end].iter().rposition(|b| *b == b' ') {
                if relative > 0 {
                    end = start + relative;
                }
            }
        }
        let part = text[start..end].trim();
        if !part.is_empty() {
            out.push(part.to_owned());
        }
        start = end;
        while start < bytes.len() && bytes[start] == b' ' {
            start += 1;
        }
    }
    out
}

fn extract_text(value: serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Object(map) => ["text", "content", "prompt", "input"]
            .iter()
            .find_map(|key| {
                map.get(*key)
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned)
            })
            .ok_or_else(|| "JSON object has no text/content/prompt/input string".into()),
        _ => Err("JSON line must be a string or object with a text field".into()),
    }
}

pub fn compile_jsonl(path: impl AsRef<Path>) -> Result<(Dataset, Vec<Rejection>), DatasetError> {
    let file = fs::File::open(path)?;
    let mut accepted = Vec::<(String, Vec<u8>, [u8; 32])>::new();
    let mut rejected = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line_no = line_index + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let result = serde_json::from_str::<serde_json::Value>(&line)
            .map_err(|e| e.to_string())
            .and_then(extract_text)
            .and_then(|text| normalize_text(&text));
        match result {
            Ok(normalized) => {
                for segment in segment_text(&normalized) {
                    let bytes = segment.as_bytes().to_vec();
                    accepted.push((segment, bytes.clone(), sample_hash(&bytes)));
                }
            }
            Err(reason) => rejected.push(Rejection {
                line: line_no,
                reason,
            }),
        }
    }
    accepted.sort_by_key(|(_, _, hash)| *hash);
    accepted.dedup_by(|a, b| a.2 == b.2);
    if accepted.is_empty() {
        return Err(DatasetError::EmptyTrain);
    }
    let mut samples = Vec::with_capacity(accepted.len());
    for (text, bytes, h) in accepted {
        let split = if (h[0] as usize * 256 + h[1] as usize) % 10 < 8 {
            "train"
        } else {
            "validation"
        };
        samples.push(Sample {
            hash: format!("0x{}", hex::encode(h)),
            text,
            bytes,
            split: split.into(),
        });
    }
    let leaves: Vec<[u8; 32]> = samples
        .iter()
        .map(|s| {
            hex::decode(s.hash.trim_start_matches("0x"))
                .unwrap()
                .try_into()
                .unwrap()
        })
        .collect();
    let merkle = merkle_root(&leaves);
    let root = hash(
        b"mini-cells:dataset-root:v1",
        &[&(samples.len() as u64).to_le_bytes(), &merkle],
    );
    Ok((
        Dataset {
            format: DATASET_FORMAT.into(),
            sample_count: samples.len(),
            samples,
            dataset_root: format!("0x{}", hex::encode(root)),
            merkle_root: format!("0x{}", hex::encode(merkle)),
        },
        rejected,
    ))
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return hash(b"mini-cells:dataset-empty:v1", &[]);
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            next.push(hash(
                b"mini-cells:dataset-node:v1",
                &[&pair[0][..], &pair.get(1).unwrap_or(&pair[0])[..]],
            ))
        }
        level = next;
    }
    level[0]
}

pub fn verify_merkle_proof(
    leaf: [u8; 32],
    leaf_index: usize,
    siblings: &[[u8; 32]],
    expected: [u8; 32],
) -> bool {
    let mut current = leaf;
    let mut index = leaf_index;
    for sibling in siblings {
        current = if index % 2 == 0 {
            hash(b"mini-cells:dataset-node:v1", &[&current, sibling])
        } else {
            hash(b"mini-cells:dataset-node:v1", &[sibling, &current])
        };
        index /= 2;
    }
    current == expected
}

fn merkle_proof(leaves: &[[u8; 32]], leaf_index: usize) -> Vec<[u8; 32]> {
    let mut level = leaves.to_vec();
    let mut index = leaf_index;
    let mut siblings = Vec::new();
    while level.len() > 1 {
        let sibling = if index % 2 == 0 {
            level.get(index + 1).unwrap_or(&level[index])
        } else {
            &level[index - 1]
        };
        siblings.push(*sibling);
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            next.push(hash(
                b"mini-cells:dataset-node:v1",
                &[&pair[0][..], &pair.get(1).unwrap_or(&pair[0])[..]],
            ));
        }
        level = next;
        index /= 2;
    }
    siblings
}

impl Dataset {
    pub fn save(
        &self,
        output: impl AsRef<Path>,
        rejected: &[Rejection],
    ) -> Result<(), DatasetError> {
        let output = output.as_ref();
        fs::create_dir_all(output)?;
        fs::write(output.join("dataset.bin"), serde_json::to_vec(self)?)?;
        let manifest = serde_json::json!({"format": self.format, "sample_count": self.sample_count,
            "dataset_root": self.dataset_root, "merkle_root": self.merkle_root,
            "train_count": self.samples.iter().filter(|s| s.split == "train").count(),
            "validation_count": self.samples.iter().filter(|s| s.split == "validation").count()});
        fs::write(
            output.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let mut file = fs::File::create(output.join("rejected.jsonl"))?;
        for item in rejected {
            writeln!(file, "{}", serde_json::to_string(item)?)?;
        }
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, DatasetError> {
        Ok(serde_json::from_slice(&fs::read(
            path.as_ref().join("dataset.bin"),
        )?)?)
    }

    pub fn batch(
        &self,
        parent_model_hash: [u8; 32],
        generation: u64,
        batch_size: usize,
    ) -> Result<(EchoBatch, BatchSelection), DatasetError> {
        if batch_size == 0 || batch_size > MAX_BATCH {
            return Err(DatasetError::Invalid("batch size must be 1..=4".into()));
        }
        let train: Vec<(usize, &Sample)> = self
            .samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.split == "train")
            .collect();
        if train.is_empty() {
            return Err(DatasetError::EmptyTrain);
        }
        if train.len() < batch_size {
            return Err(DatasetError::TooSmall);
        }
        let root = hex::decode(self.dataset_root.trim_start_matches("0x"))
            .map_err(|_| DatasetError::Invalid("bad dataset root".into()))?;
        let seed_hash = hash(
            b"mini-cells:batch-seed:v1",
            &[&root, &parent_model_hash, &generation.to_le_bytes()],
        );
        let mut state = u64::from_le_bytes(seed_hash[..8].try_into().unwrap());
        let mut chosen = Vec::with_capacity(batch_size);
        while chosen.len() < batch_size {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            let index = (z ^ (z >> 31)) as usize % train.len();
            if !chosen.contains(&index) {
                chosen.push(index);
            }
        }
        chosen.sort_unstable();
        let mut batch = EchoBatch::default();
        batch.size = batch_size as u8;
        let mut digest_parts: Vec<Vec<u8>> = Vec::new();
        let mut sample_hashes = Vec::new();
        let mut indices = Vec::new();
        let all_leaves: Vec<[u8; 32]> = self
            .samples
            .iter()
            .map(|sample| {
                hex::decode(sample.hash.trim_start_matches("0x"))
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect();
        let mut proofs = Vec::new();
        for (slot, train_index) in chosen.iter().enumerate() {
            let (original_index, sample) = train[*train_index];
            indices.push(original_index);
            sample_hashes.push(sample.hash.clone());
            let leaf = all_leaves[original_index];
            proofs.push(MerkleProof {
                leaf: format!("0x{}", hex::encode(leaf)),
                leaf_index: original_index,
                siblings: merkle_proof(&all_leaves, original_index)
                    .iter()
                    .map(|x| format!("0x{}", hex::encode(x)))
                    .collect(),
            });
            batch.lengths[slot] = sample.bytes.len() as u8;
            for (cursor, byte) in sample.bytes.iter().enumerate() {
                batch.ids[slot][cursor] = encode_byte(*byte).ok_or_else(|| {
                    DatasetError::Invalid("compiled byte outside vocabulary".into())
                })?;
            }
            digest_parts.push(sample.bytes.clone());
        }
        let generation_bytes = generation.to_le_bytes();
        let mut digest_input: Vec<&[u8]> = vec![&root, &parent_model_hash, &generation_bytes];
        for bytes in &digest_parts {
            digest_input.push(bytes);
        }
        let digest = hash(b"mini-cells:batch-digest:v1", &digest_input);
        let witness = hash(b"mini-cells:batch-witness:v1", &digest_input);
        let selection = BatchSelection {
            dataset_root: self.dataset_root.clone(),
            parent_model_hash: format!("0x{}", hex::encode(parent_model_hash)),
            generation,
            indices,
            sample_hashes,
            digest: format!("0x{}", hex::encode(digest)),
            witness: format!("0x{}", hex::encode(witness)),
            proofs,
        };
        Ok((batch, selection))
    }
}

pub fn inspect(path: impl AsRef<Path>) -> Result<serde_json::Value, DatasetError> {
    let d = Dataset::load(path)?;
    Ok(
        serde_json::json!({"format": d.format, "sample_count": d.sample_count, "dataset_root": d.dataset_root,
        "merkle_root": d.merkle_root, "train_count": d.samples.iter().filter(|s| s.split == "train").count(),
        "validation_count": d.samples.iter().filter(|s| s.split == "validation").count()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_and_segmentation_are_stable() {
        assert_eq!(
            normalize_text("  HeLLo  “WORLD”—x  ").unwrap(),
            "hello 'world'-x"
        );
        assert!(normalize_text("café").is_err());
        let text = "a very long deterministic sentence that must split";
        assert!(segment_text(text).iter().all(|s| s.len() <= 32));
        assert_eq!(segment_text(text), segment_text(text));
    }
    #[test]
    fn roots_and_batches_are_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("data.jsonl");
        fs::write(&input, "{\"text\":\"hello world\"}\n{\"text\":\"mini cells\"}\n{\"text\":\"jam local\"}\n{\"text\":\"echo neural\"}\n{\"text\":\"hello world\"}\n").unwrap();
        let (a, _) = compile_jsonl(&input).unwrap();
        let (b, _) = compile_jsonl(&input).unwrap();
        assert_eq!(a, b);
        let (_, x) = a.batch([7; 32], 0, 4).unwrap();
        let (_, y) = a.batch([7; 32], 0, 4).unwrap();
        assert_eq!(x, y);
        let root = hex::decode(a.merkle_root.trim_start_matches("0x"))
            .unwrap()
            .try_into()
            .unwrap();
        for proof in &x.proofs {
            let leaf = hex::decode(proof.leaf.trim_start_matches("0x"))
                .unwrap()
                .try_into()
                .unwrap();
            let siblings: Vec<[u8; 32]> = proof
                .siblings
                .iter()
                .map(|s| {
                    hex::decode(s.trim_start_matches("0x"))
                        .unwrap()
                        .try_into()
                        .unwrap()
                })
                .collect();
            assert!(verify_merkle_proof(leaf, proof.leaf_index, &siblings, root));
        }
    }
}
