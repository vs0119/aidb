use serde::{Deserialize, Serialize};

/// Simple Bloom filter implementation optimized for immutable SSTable membership checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

impl BloomFilter {
    pub fn new(num_bits: usize, num_hashes: u32) -> Self {
        let num_bits = num_bits.max(64);
        let slots = (num_bits + 63) / 64;
        Self {
            bits: vec![0; slots],
            num_bits: slots * 64,
            num_hashes: num_hashes.max(1),
        }
    }

    pub fn from_keys<I, K>(keys: I, bits_per_key: usize, num_hashes: u32) -> Self
    where
        I: IntoIterator<Item = K>,
        K: AsRef<[u8]>,
    {
        let collected: Vec<Vec<u8>> = keys.into_iter().map(|k| k.as_ref().to_vec()).collect();
        let num_items = collected.len().max(1);
        let num_bits = num_items * bits_per_key.max(1);
        let mut filter = Self::new(num_bits, num_hashes);
        for key in &collected {
            filter.insert(key);
        }
        filter
    }

    pub fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let hash = hash64_with_seed(key, hash_seed(i));
            let bit = (hash as usize) % self.num_bits;
            let word = bit / 64;
            let offset = bit % 64;
            self.bits[word] |= 1u64 << offset;
        }
    }

    pub fn might_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let hash = hash64_with_seed(key, hash_seed(i));
            let bit = (hash as usize) % self.num_bits;
            let word = bit / 64;
            let offset = bit % 64;
            if (self.bits[word] & (1u64 << offset)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn bit_vec(&self) -> &[u64] {
        &self.bits
    }

    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }
}

fn hash_seed(round: u32) -> u64 {
    0x9e3779b97f4a7c15u64.wrapping_mul(round as u64 + 1)
}

fn hash64_with_seed(data: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= hash >> 32;
        hash = hash.wrapping_mul(0x9e3779b185ebca87);
    }
    hash
}
