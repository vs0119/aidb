use std::collections::HashMap;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    compression::{CompressedData, CompressionAlgorithm, CompressionManager},
    PageId, Result, StorageEngineError,
};

/// Maximum amount of columns we allow to be stored inside a single columnar page.
const MAX_COLUMN_COUNT: usize = 256;

/// Magic number used to identify columnar pages on disk.
pub const COLUMNAR_PAGE_MAGIC: u32 = 0xA1D0C01u32;

/// Layout metadata for a columnar page.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnarPageHeader {
    /// Identifier of the page in the global page catalog.
    pub page_id: PageId,
    /// Number of logical rows represented by this page.
    pub row_count: u32,
    /// Number of materialized columns stored inside the page.
    pub column_count: u16,
    /// Bit flags encoding optional features (dictionary usage, compression, ...).
    pub flags: u16,
    /// Magic value used to validate the page on read.
    pub magic: u32,
}

impl ColumnarPageHeader {
    fn new(page_id: PageId, row_count: u32, column_count: u16, flags: u16) -> Self {
        Self {
            page_id,
            row_count,
            column_count,
            flags,
            magic: COLUMNAR_PAGE_MAGIC,
        }
    }
}

/// Encoding technique used for a column chunk.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum ColumnEncoding {
    Plain = 0,
    Dictionary = 1,
    RunLength = 2,
    BitPacked = 3,
}

/// Binary representation of a column inside a columnar page.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnChunkMetadata {
    /// Column identifier relative to the owning collection schema.
    pub column_id: u16,
    /// Encoding used for the column chunk.
    pub encoding: ColumnEncoding,
    /// Optional compression algorithm applied to the stored data.
    pub compression: Option<CompressionAlgorithm>,
    /// Number of rows represented by this chunk.
    pub row_count: u32,
    /// Number of null values in the column.
    pub null_count: u32,
    /// Length of the encoded data in bytes.
    pub data_length: u32,
    /// Length of the null bitmap in bytes.
    pub null_bitmap_length: u32,
    /// Size of the column prior to compression.
    pub uncompressed_length: u32,
}

/// Raw bytes plus optional compression used to store columnar values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColumnChunkStorage {
    Plain(Vec<u8>),
    Compressed(CompressedData),
}

impl ColumnChunkStorage {
    fn data_length(&self) -> usize {
        match self {
            ColumnChunkStorage::Plain(bytes) => bytes.len(),
            ColumnChunkStorage::Compressed(data) => data.data.len(),
        }
    }
}

/// Encoded column and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnChunk {
    pub metadata: ColumnChunkMetadata,
    pub storage: ColumnChunkStorage,
    pub null_bitmap: Option<Vec<u8>>,
}

impl ColumnChunk {
    /// Materializes the raw bytes of the column chunk. If the chunk is stored
    /// using a compression algorithm the provided [`CompressionManager`] is
    /// used to perform the decompression lazily.
    pub fn materialize(&self, manager: Option<&CompressionManager>) -> Result<Vec<u8>> {
        match &self.storage {
            ColumnChunkStorage::Plain(bytes) => Ok(bytes.clone()),
            ColumnChunkStorage::Compressed(data) => {
                let manager = manager.ok_or_else(|| {
                    StorageEngineError::CompressionError(
                        "compression manager required for compressed column".to_string(),
                    )
                })?;
                manager.decompress(data)
            }
        }
    }

    /// Returns the stored byte length after compression.
    pub fn stored_len(&self) -> usize {
        self.storage.data_length()
    }
}

/// Builder friendly representation of a columnar page.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnarPage {
    pub header: ColumnarPageHeader,
    pub columns: Vec<ColumnChunk>,
}

impl ColumnarPage {
    /// Creates an empty columnar page for the specified number of rows.
    pub fn new(page_id: PageId, row_count: u32) -> Self {
        Self {
            header: ColumnarPageHeader::new(page_id, row_count, 0, 0),
            columns: Vec::new(),
        }
    }

    /// Adds a new column chunk to the page and optionally compresses the data
    /// using the provided compression manager.
    pub fn add_column(
        &mut self,
        column_id: u16,
        encoding: ColumnEncoding,
        mut data: Vec<u8>,
        null_bitmap: Option<Vec<u8>>,
        compression: Option<&CompressionManager>,
    ) -> Result<()> {
        if self.columns.len() >= MAX_COLUMN_COUNT {
            return Err(StorageEngineError::PageCorruption(
                "columnar page exceeds maximum column count".to_string(),
            ));
        }

        let uncompressed_length = data.len() as u32;
        let storage = if let Some(manager) = compression {
            if data.is_empty() {
                ColumnChunkStorage::Plain(data)
            } else {
                let compressed = manager.compress(&data)?;
                data.clear();
                ColumnChunkStorage::Compressed(compressed)
            }
        } else {
            ColumnChunkStorage::Plain(data)
        };

        let compression_algorithm = match &storage {
            ColumnChunkStorage::Plain(_) => None,
            ColumnChunkStorage::Compressed(comp) => Some(comp.algorithm),
        };

        let data_length = storage.data_length() as u32;
        let null_bitmap_length = null_bitmap.as_ref().map(|b| b.len()).unwrap_or(0) as u32;
        let null_count = null_bitmap
            .as_ref()
            .map(|bitmap| bitmap.iter().map(|byte| byte.count_ones()).sum::<u32>())
            .unwrap_or(0);

        let chunk = ColumnChunk {
            metadata: ColumnChunkMetadata {
                column_id,
                encoding,
                compression: compression_algorithm,
                row_count: self.header.row_count,
                null_count,
                data_length,
                null_bitmap_length,
                uncompressed_length,
            },
            storage,
            null_bitmap,
        };

        self.columns.push(chunk);
        self.header.column_count += 1;
        Ok(())
    }

    /// Returns the total byte footprint for the page (metadata + data).
    pub fn total_size(&self) -> usize {
        let header_size = std::mem::size_of::<ColumnarPageHeader>();
        let metadata_size = self.columns.len() * std::mem::size_of::<ColumnChunkMetadata>();
        let bitmap_size: usize = self
            .columns
            .iter()
            .map(|c| c.null_bitmap.as_ref().map(|b| b.len()).unwrap_or(0))
            .sum();
        let data_size: usize = self.columns.iter().map(|c| c.stored_len()).sum();

        header_size + metadata_size + bitmap_size + data_size
    }

    /// Fetches the column chunk by identifier.
    pub fn column(&self, column_id: u16) -> Option<&ColumnChunk> {
        self.columns
            .iter()
            .find(|c| c.metadata.column_id == column_id)
    }
}

/// A bitmap representing nullability for dictionary encoded columns.
fn build_null_bitmap(len: usize, null_indices: &[usize]) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }

    let mut bitmap = vec![0u8; (len + 7) / 8];
    for &index in null_indices {
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        bitmap[byte_idx] |= 1 << bit_idx;
    }
    bitmap
}

/// Dictionary encoder used to compact string heavy columns.
#[derive(Debug, Default, Clone)]
pub struct DictionaryEncoder {
    dictionary: Vec<String>,
    index: HashMap<String, u32>,
}

impl DictionaryEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encodes the provided iterator of optional string slices.
    pub fn encode<I, S>(&mut self, values: I) -> DictionaryEncodedColumn
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let mut indices = Vec::new();
        let mut nulls = Vec::new();

        for (position, value) in values.into_iter().enumerate() {
            match value {
                Some(val) => {
                    let entry = if let Some(existing) = self.index.get(val.as_ref()) {
                        *existing
                    } else {
                        let next = self.dictionary.len() as u32;
                        self.dictionary.push(val.as_ref().to_string());
                        self.index.insert(val.as_ref().to_string(), next);
                        next
                    };
                    indices.push(entry);
                }
                None => {
                    nulls.push(position);
                    indices.push(0); // Placeholder, null bitmap disambiguates.
                }
            }
        }

        let bit_packed = BitPackedColumn::from_values(&indices);
        let null_bitmap = build_null_bitmap(indices.len(), &nulls);

        DictionaryEncodedColumn {
            dictionary: self.dictionary.clone(),
            indices: bit_packed,
            null_bitmap,
        }
    }
}

/// Materialized dictionary encoded data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryEncodedColumn {
    pub dictionary: Vec<String>,
    pub indices: BitPackedColumn,
    pub null_bitmap: Vec<u8>,
}

/// Decodes dictionary encoded columns back into optional strings.
#[derive(Debug, Default, Clone)]
pub struct DictionaryDecoder;

impl DictionaryDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decode(&self, column: &DictionaryEncodedColumn) -> Vec<Option<String>> {
        let indices = column.indices.decode();
        let mut result = Vec::with_capacity(indices.len());

        for (idx, value) in indices.into_iter().enumerate() {
            if is_null(&column.null_bitmap, idx) {
                result.push(None);
            } else {
                let decoded = column
                    .dictionary
                    .get(value as usize)
                    .cloned()
                    .unwrap_or_default();
                result.push(Some(decoded));
            }
        }

        result
    }
}

fn is_null(bitmap: &[u8], index: usize) -> bool {
    if bitmap.is_empty() {
        return false;
    }
    let byte_idx = index / 8;
    if byte_idx >= bitmap.len() {
        return false;
    }
    let bit_idx = index % 8;
    (bitmap[byte_idx] & (1 << bit_idx)) != 0
}

/// Simple run length encoder for generic values.
#[derive(Debug, Default, Clone)]
pub struct RLECompressor;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct RleRun<T> {
    value: T,
    run_length: u32,
}

impl RLECompressor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn encode<T>(&self, values: &[T]) -> Result<Vec<u8>>
    where
        T: PartialEq + Clone + Serialize,
    {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut runs: Vec<RleRun<T>> = Vec::new();
        let mut current_value = values[0].clone();
        let mut run_length = 1u32;

        for value in &values[1..] {
            if *value == current_value {
                run_length += 1;
            } else {
                runs.push(RleRun {
                    value: current_value,
                    run_length,
                });
                current_value = value.clone();
                run_length = 1;
            }
        }

        runs.push(RleRun {
            value: current_value,
            run_length,
        });

        bincode::serialize(&runs).map_err(|err| {
            StorageEngineError::CompressionError(format!("RLE encoding failed: {err}"))
        })
    }

    pub fn decode<T>(&self, data: &[u8]) -> Result<Vec<T>>
    where
        T: DeserializeOwned + Clone,
    {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let runs: Vec<RleRun<T>> = bincode::deserialize(data).map_err(|err| {
            StorageEngineError::CompressionError(format!("RLE decoding failed: {err}"))
        })?;

        let mut values = Vec::new();
        for run in runs {
            for _ in 0..run.run_length {
                values.push(run.value.clone());
            }
        }
        Ok(values)
    }
}

/// Compact representation of low-cardinality integer data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitPackedColumn {
    bit_width: u8,
    length: usize,
    data: Vec<u8>,
}

impl BitPackedColumn {
    pub fn from_values(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                bit_width: 0,
                length: 0,
                data: Vec::new(),
            };
        }

        let max_value = values.iter().copied().max().unwrap_or(0);
        let bit_width = if max_value == 0 {
            1
        } else {
            (32 - max_value.leading_zeros()) as u8
        };

        let mut data = Vec::with_capacity(((values.len() * bit_width as usize) + 7) / 8);
        let mut current_byte = 0u8;
        let mut bits_filled = 0u8;

        for &value in values {
            let mut remaining_bits = bit_width;
            let mut value = value;
            while remaining_bits > 0 {
                let available = 8 - bits_filled;
                let bits_to_write = remaining_bits.min(available);
                let mask = if bits_to_write == 32 {
                    u32::MAX
                } else {
                    (1u32 << bits_to_write) - 1
                };
                let chunk = (value & mask) as u8;
                current_byte |= chunk << bits_filled;
                bits_filled += bits_to_write;
                value >>= bits_to_write;
                remaining_bits -= bits_to_write;

                if bits_filled == 8 {
                    data.push(current_byte);
                    current_byte = 0;
                    bits_filled = 0;
                }
            }
        }

        if bits_filled > 0 {
            data.push(current_byte);
        }

        Self {
            bit_width,
            length: values.len(),
            data,
        }
    }

    pub fn decode(&self) -> Vec<u32> {
        if self.length == 0 {
            return Vec::new();
        }

        if self.bit_width == 0 {
            return vec![0; self.length];
        }

        let mut values = Vec::with_capacity(self.length);
        let mut bit_index = 0usize;
        for _ in 0..self.length {
            let mut collected = 0u32;
            let mut bits_collected = 0u8;
            while bits_collected < self.bit_width {
                let byte_index = bit_index / 8;
                let bit_offset = (bit_index % 8) as u8;
                let available = 8 - bit_offset;
                let needed = self.bit_width - bits_collected;
                let take = available.min(needed);
                let mask = ((1u16 << take) - 1) as u8;
                let source = self.data.get(byte_index).copied().unwrap_or(0);
                let chunk = (source >> bit_offset) & mask;
                collected |= (chunk as u32) << bits_collected;
                bits_collected += take;
                bit_index += take as usize;
            }
            values.push(collected);
        }
        values
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Storage mode used by a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    Row,
    Columnar,
}

/// Lightweight statistics used for selecting the optimal storage layout.
#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    pub row_count: usize,
    pub average_row_width: usize,
    pub column_cardinalities: Vec<usize>,
}

impl CollectionStatistics {
    pub fn new(
        row_count: usize,
        average_row_width: usize,
        column_cardinalities: Vec<usize>,
    ) -> Self {
        Self {
            row_count,
            average_row_width,
            column_cardinalities,
        }
    }
}

/// Heuristic based selector that chooses between row oriented and columnar
/// layout depending on the workload characteristics.
#[derive(Debug, Clone)]
pub struct StorageModeSelector {
    /// Minimum amount of rows required before columnar storage becomes
    /// attractive.
    pub columnar_row_threshold: usize,
    /// Maximum average row width where row storage is still competitive.
    pub row_width_threshold: usize,
    /// Cardinality ratio below which bit packing/dictionary encoding performs well.
    pub low_cardinality_ratio: f64,
}

impl Default for StorageModeSelector {
    fn default() -> Self {
        Self {
            columnar_row_threshold: 10_000,
            row_width_threshold: 256,
            low_cardinality_ratio: 0.1,
        }
    }
}

impl StorageModeSelector {
    pub fn new(
        columnar_row_threshold: usize,
        row_width_threshold: usize,
        low_cardinality_ratio: f64,
    ) -> Self {
        Self {
            columnar_row_threshold,
            row_width_threshold,
            low_cardinality_ratio,
        }
    }

    /// Selects the optimal storage mode. Columnar storage is preferred when we
    /// have enough rows to amortize the encoding costs and at least one column
    /// with sufficiently low cardinality that benefits from compression.
    pub fn select_mode(&self, stats: &CollectionStatistics) -> StorageMode {
        if stats.row_count < self.columnar_row_threshold {
            return StorageMode::Row;
        }

        if stats.average_row_width <= self.row_width_threshold {
            return StorageMode::Row;
        }

        if stats.column_cardinalities.is_empty() {
            return StorageMode::Row;
        }

        let average_cardinality = stats.column_cardinalities.iter().copied().sum::<usize>() as f64
            / stats.column_cardinalities.len() as f64;
        let cardinality_ratio = average_cardinality / stats.row_count.max(1) as f64;

        if cardinality_ratio <= self.low_cardinality_ratio {
            StorageMode::Columnar
        } else {
            StorageMode::Row
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_encoding_roundtrip() {
        let mut encoder = DictionaryEncoder::new();
        let data = vec![
            Some("alpha"),
            Some("beta"),
            Some("alpha"),
            None,
            Some("gamma"),
        ];
        let encoded = encoder.encode(data.clone());
        assert_eq!(encoded.dictionary.len(), 3);
        let decoder = DictionaryDecoder::new();
        let decoded = decoder.decode(&encoded);
        assert_eq!(
            decoded,
            data.into_iter()
                .map(|v| v.map(|s| s.to_string()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rle_encoding_roundtrip() {
        let compressor = RLECompressor::new();
        let data = vec![1u32, 1, 1, 2, 2, 3, 3, 3, 3];
        let encoded = compressor.encode(&data).expect("rle encoding");
        let decoded: Vec<u32> = compressor.decode(&encoded).expect("rle decoding");
        assert_eq!(decoded, data);
    }

    #[test]
    fn bitpacking_roundtrip() {
        let values = vec![0u32, 1, 2, 3, 1, 0, 3, 2];
        let packed = BitPackedColumn::from_values(&values);
        assert!(packed.bit_width() <= 2);
        let decoded = packed.decode();
        assert_eq!(decoded, values);
    }

    #[test]
    fn storage_mode_selection() {
        let selector = StorageModeSelector::default();
        let stats = CollectionStatistics::new(50_000, 1024, vec![10, 5, 20]);
        assert_eq!(selector.select_mode(&stats), StorageMode::Columnar);

        let small_stats = CollectionStatistics::new(100, 128, vec![100, 200]);
        assert_eq!(selector.select_mode(&small_stats), StorageMode::Row);
    }

    #[test]
    fn columnar_page_metadata() {
        let mut page = ColumnarPage::new(1, 1000);
        let compression_manager = CompressionManager::new();
        page.add_column(
            0,
            ColumnEncoding::Plain,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            None,
            Some(&compression_manager),
        )
        .expect("add column");
        assert_eq!(page.header.column_count, 1);
        assert!(page.total_size() > std::mem::size_of::<ColumnarPageHeader>());
        let column = page.column(0).expect("column exists");
        assert_eq!(column.metadata.column_id, 0);
        assert_eq!(column.metadata.row_count, 1000);
    }
}
