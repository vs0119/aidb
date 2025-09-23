use std::hash::{Hash, Hasher};

use crate::{
    coerce_static_value, column_index_in_table, compare_values, ColumnType, Predicate,
    SqlDatabaseError, Table, Value,
};

#[derive(Debug, Clone)]
pub struct PartitionMetadata {
    pub name: String,
    pub bounds: PartitionBounds,
}

#[derive(Debug, Clone)]
pub enum PartitionBounds {
    Range { upper: Option<Value> },
    List { values: Vec<Value> },
    Hash { modulus: usize, remainder: usize },
    Default,
}

#[derive(Debug, Clone)]
pub enum PartitioningDefinition {
    Range { partitions: Vec<PartitionMetadata> },
    Hash { partitions: usize },
    List { partitions: Vec<PartitionMetadata> },
}

#[derive(Debug, thiserror::Error)]
pub enum PartitionError {
    #[error("partitioning is not defined for table")]
    MissingPartitioning,
    #[error("partition '{0}' already exists")]
    DuplicatePartition(String),
    #[error("partition '{0}' does not exist")]
    UnknownPartition(String),
    #[error("value does not map to any partition")]
    ValueOutOfRange,
    #[error("invalid partition definition: {0}")]
    InvalidDefinition(String),
}

pub trait PartitioningScheme {
    fn column_index(&self) -> usize;
    fn column_name(&self) -> &str;
    fn partitions(&self) -> &[PartitionMetadata];
    fn matches_column(&self, column: &str) -> bool {
        self.column_name().eq_ignore_ascii_case(column)
    }
    fn locate_partition(&self, value: &Value) -> Option<&PartitionMetadata>;
    fn prune_partitions(
        &self,
        predicate: &Predicate,
        column_type: ColumnType,
    ) -> Result<Option<Vec<String>>, PartitionError>;
    fn add_partition(&mut self, metadata: PartitionMetadata) -> Result<(), PartitionError>;
    fn drop_partition(&mut self, name: &str) -> Result<PartitionMetadata, PartitionError>;
}

#[derive(Debug, Clone)]
pub enum PartitionSchemeDefinition {
    Range(RangePartitioning),
    Hash(HashPartitioning),
    List(ListPartitioning),
}

impl PartitioningScheme for PartitionSchemeDefinition {
    fn column_index(&self) -> usize {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.column_index(),
            PartitionSchemeDefinition::Hash(scheme) => scheme.column_index(),
            PartitionSchemeDefinition::List(scheme) => scheme.column_index(),
        }
    }

    fn column_name(&self) -> &str {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.column_name(),
            PartitionSchemeDefinition::Hash(scheme) => scheme.column_name(),
            PartitionSchemeDefinition::List(scheme) => scheme.column_name(),
        }
    }

    fn partitions(&self) -> &[PartitionMetadata] {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.partitions(),
            PartitionSchemeDefinition::Hash(scheme) => scheme.partitions(),
            PartitionSchemeDefinition::List(scheme) => scheme.partitions(),
        }
    }

    fn locate_partition(&self, value: &Value) -> Option<&PartitionMetadata> {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.locate_partition(value),
            PartitionSchemeDefinition::Hash(scheme) => scheme.locate_partition(value),
            PartitionSchemeDefinition::List(scheme) => scheme.locate_partition(value),
        }
    }

    fn prune_partitions(
        &self,
        predicate: &Predicate,
        column_type: ColumnType,
    ) -> Result<Option<Vec<String>>, PartitionError> {
        match self {
            PartitionSchemeDefinition::Range(scheme) => {
                scheme.prune_partitions(predicate, column_type)
            }
            PartitionSchemeDefinition::Hash(scheme) => {
                scheme.prune_partitions(predicate, column_type)
            }
            PartitionSchemeDefinition::List(scheme) => {
                scheme.prune_partitions(predicate, column_type)
            }
        }
    }

    fn add_partition(&mut self, metadata: PartitionMetadata) -> Result<(), PartitionError> {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.add_partition(metadata),
            PartitionSchemeDefinition::Hash(scheme) => scheme.add_partition(metadata),
            PartitionSchemeDefinition::List(scheme) => scheme.add_partition(metadata),
        }
    }

    fn drop_partition(&mut self, name: &str) -> Result<PartitionMetadata, PartitionError> {
        match self {
            PartitionSchemeDefinition::Range(scheme) => scheme.drop_partition(name),
            PartitionSchemeDefinition::Hash(scheme) => scheme.drop_partition(name),
            PartitionSchemeDefinition::List(scheme) => scheme.drop_partition(name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangePartitioning {
    column_index: usize,
    column_name: String,
    partitions: Vec<PartitionMetadata>,
}

impl RangePartitioning {
    pub(crate) fn new(
        column_index: usize,
        column_name: String,
        partitions: Vec<PartitionMetadata>,
    ) -> Result<Self, PartitionError> {
        if partitions.is_empty() {
            return Err(PartitionError::InvalidDefinition(
                "at least one range partition is required".into(),
            ));
        }
        for metadata in &partitions {
            match metadata.bounds {
                PartitionBounds::Range { .. } => {}
                _ => {
                    return Err(PartitionError::InvalidDefinition(
                        "range partition must use RANGE bounds".into(),
                    ))
                }
            }
        }
        let mut result = Self {
            column_index,
            column_name,
            partitions,
        };
        result.sort_partitions();
        Ok(result)
    }

    fn sort_partitions(&mut self) {
        self.partitions
            .sort_by(|left, right| match (&left.bounds, &right.bounds) {
                (
                    PartitionBounds::Range { upper: Some(l) },
                    PartitionBounds::Range { upper: Some(r) },
                ) => compare_values(l, r).unwrap_or(std::cmp::Ordering::Equal),
                (
                    PartitionBounds::Range { upper: Some(_) },
                    PartitionBounds::Range { upper: None },
                ) => std::cmp::Ordering::Less,
                (
                    PartitionBounds::Range { upper: None },
                    PartitionBounds::Range { upper: Some(_) },
                ) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            });
    }

    fn partition_index_for_value(&self, value: &Value) -> Option<usize> {
        for (idx, partition) in self.partitions.iter().enumerate() {
            match &partition.bounds {
                PartitionBounds::Range { upper: Some(bound) } => {
                    if let Some(ordering) = compare_values(value, bound) {
                        if ordering == std::cmp::Ordering::Less {
                            return Some(idx);
                        }
                    }
                }
                PartitionBounds::Range { upper: None } => return Some(idx),
                _ => {}
            }
        }
        None
    }

    fn overlapping_partition_names(&self, start: &Value, end: Option<&Value>) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(end_value) = end {
            if let (Some(mut start_idx), Some(mut end_idx)) = (
                self.partition_index_for_value(start),
                self.partition_index_for_value(end_value),
            ) {
                if start_idx > end_idx {
                    std::mem::swap(&mut start_idx, &mut end_idx);
                }
                for idx in start_idx..=end_idx {
                    if let Some(partition) = self.partitions.get(idx) {
                        names.push(partition.name.clone());
                    }
                }
            }
        } else if let Some(start_idx) = self.partition_index_for_value(start) {
            for idx in start_idx..self.partitions.len() {
                if let Some(partition) = self.partitions.get(idx) {
                    names.push(partition.name.clone());
                }
            }
        }
        names
    }
}

impl PartitioningScheme for RangePartitioning {
    fn column_index(&self) -> usize {
        self.column_index
    }

    fn column_name(&self) -> &str {
        &self.column_name
    }

    fn partitions(&self) -> &[PartitionMetadata] {
        &self.partitions
    }

    fn locate_partition(&self, value: &Value) -> Option<&PartitionMetadata> {
        self.partition_index_for_value(value)
            .and_then(|idx| self.partitions.get(idx))
    }

    fn prune_partitions(
        &self,
        predicate: &Predicate,
        column_type: ColumnType,
    ) -> Result<Option<Vec<String>>, PartitionError> {
        match predicate {
            Predicate::Equals { value, .. } => {
                let coerced = coerce_static_value(value, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                if let Some(partition) = self.locate_partition(&coerced) {
                    return Ok(Some(vec![partition.name.clone()]));
                }
                Ok(Some(Vec::new()))
            }
            Predicate::Between { start, end, .. } => {
                let start_value = coerce_static_value(start, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                let end_value = coerce_static_value(end, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                let mut names = Vec::new();
                let lower_names = self.overlapping_partition_names(&start_value, Some(&end_value));
                names.extend(lower_names);
                names.sort();
                names.dedup();
                Ok(Some(names))
            }
            Predicate::GreaterOrEqual { value, .. } => {
                let start_value = coerce_static_value(value, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                let mut names = self.overlapping_partition_names(&start_value, None);
                names.sort();
                names.dedup();
                Ok(Some(names))
            }
            _ => Ok(None),
        }
    }

    fn add_partition(&mut self, metadata: PartitionMetadata) -> Result<(), PartitionError> {
        if self
            .partitions
            .iter()
            .any(|partition| partition.name.eq_ignore_ascii_case(&metadata.name))
        {
            return Err(PartitionError::DuplicatePartition(metadata.name));
        }
        if !matches!(metadata.bounds, PartitionBounds::Range { .. }) {
            return Err(PartitionError::InvalidDefinition(
                "range partition must use RANGE bounds".into(),
            ));
        }
        self.partitions.push(metadata);
        self.sort_partitions();
        Ok(())
    }

    fn drop_partition(&mut self, name: &str) -> Result<PartitionMetadata, PartitionError> {
        if self.partitions.len() == 1 {
            return Err(PartitionError::InvalidDefinition(
                "cannot drop the last range partition".into(),
            ));
        }
        if let Some(index) = self
            .partitions
            .iter()
            .position(|partition| partition.name.eq_ignore_ascii_case(name))
        {
            Ok(self.partitions.remove(index))
        } else {
            Err(PartitionError::UnknownPartition(name.into()))
        }
    }
}

#[derive(Debug, Clone)]
pub struct HashPartitioning {
    column_index: usize,
    column_name: String,
    partitions: Vec<PartitionMetadata>,
    modulus: usize,
}

impl HashPartitioning {
    pub(crate) fn new(
        column_index: usize,
        column_name: String,
        modulus: usize,
    ) -> Result<Self, PartitionError> {
        if modulus == 0 {
            return Err(PartitionError::InvalidDefinition(
                "hash partition count must be greater than zero".into(),
            ));
        }
        let partitions = (0..modulus)
            .map(|remainder| PartitionMetadata {
                name: format!("p{remainder}"),
                bounds: PartitionBounds::Hash { modulus, remainder },
            })
            .collect();
        Ok(Self {
            column_index,
            column_name,
            partitions,
            modulus,
        })
    }

    fn hash_value(&self, value: &Value) -> Result<usize, PartitionError> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match value {
            Value::Null => {
                return Err(PartitionError::ValueOutOfRange);
            }
            Value::Integer(v) => v.hash(&mut hasher),
            Value::Float(v) => v.to_bits().hash(&mut hasher),
            Value::Text(v) => v.hash(&mut hasher),
            Value::Boolean(v) => v.hash(&mut hasher),
            Value::Timestamp(v) => v.timestamp_nanos_opt().hash(&mut hasher),
            Value::Json(v) | Value::Jsonb(v) => {
                let fingerprint = crate::canonical_json(v);
                fingerprint.hash(&mut hasher);
            }
            Value::Xml(v) => v.hash(&mut hasher),
            Value::Geometry(g) => {
                crate::geometry_to_string(g).hash(&mut hasher);
            }
        }
        Ok((hasher.finish() as usize) % self.modulus)
    }
}

impl PartitioningScheme for HashPartitioning {
    fn column_index(&self) -> usize {
        self.column_index
    }

    fn column_name(&self) -> &str {
        &self.column_name
    }

    fn partitions(&self) -> &[PartitionMetadata] {
        &self.partitions
    }

    fn locate_partition(&self, value: &Value) -> Option<&PartitionMetadata> {
        let Ok(remainder) = self.hash_value(value) else {
            return None;
        };
        self.partitions
            .iter()
            .find(|partition| match partition.bounds {
                PartitionBounds::Hash { remainder: r, .. } => r == remainder,
                _ => false,
            })
    }

    fn prune_partitions(
        &self,
        predicate: &Predicate,
        column_type: ColumnType,
    ) -> Result<Option<Vec<String>>, PartitionError> {
        match predicate {
            Predicate::Equals { value, .. } => {
                let coerced = coerce_static_value(value, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                let remainder = self.hash_value(&coerced)?;
                let name = self
                    .partitions
                    .iter()
                    .find(|partition| match partition.bounds {
                        PartitionBounds::Hash { remainder: r, .. } => r == remainder,
                        _ => false,
                    })
                    .map(|meta| meta.name.clone())
                    .unwrap_or_else(|| format!("p{remainder}"));
                Ok(Some(vec![name]))
            }
            _ => Ok(None),
        }
    }

    fn add_partition(&mut self, metadata: PartitionMetadata) -> Result<(), PartitionError> {
        match metadata.bounds {
            PartitionBounds::Hash { modulus, remainder } => {
                if modulus != self.modulus {
                    return Err(PartitionError::InvalidDefinition(
                        "hash modulus mismatch".into(),
                    ));
                }
                if remainder >= modulus {
                    return Err(PartitionError::InvalidDefinition(
                        "hash remainder out of range".into(),
                    ));
                }
                if self
                    .partitions
                    .iter()
                    .any(|partition| partition.name.eq_ignore_ascii_case(&metadata.name))
                {
                    return Err(PartitionError::DuplicatePartition(metadata.name));
                }
                self.partitions.push(metadata);
                Ok(())
            }
            _ => Err(PartitionError::InvalidDefinition(
                "hash partition must use HASH bounds".into(),
            )),
        }
    }

    fn drop_partition(&mut self, name: &str) -> Result<PartitionMetadata, PartitionError> {
        Err(PartitionError::InvalidDefinition(format!(
            "cannot drop hash partition '{name}'"
        )))
    }
}

#[derive(Debug, Clone)]
pub struct ListPartitioning {
    column_index: usize,
    column_name: String,
    partitions: Vec<PartitionMetadata>,
}

impl ListPartitioning {
    pub(crate) fn new(
        column_index: usize,
        column_name: String,
        partitions: Vec<PartitionMetadata>,
    ) -> Result<Self, PartitionError> {
        if partitions.is_empty() {
            return Err(PartitionError::InvalidDefinition(
                "at least one list partition is required".into(),
            ));
        }
        let mut encountered = Vec::new();
        for metadata in &partitions {
            match &metadata.bounds {
                PartitionBounds::List { values } => {
                    for value in values {
                        if encountered.iter().any(|existing| existing == value) {
                            return Err(PartitionError::InvalidDefinition(
                                "duplicate value across list partitions".into(),
                            ));
                        }
                        encountered.push(value.clone());
                    }
                }
                PartitionBounds::Default => {}
                _ => {
                    return Err(PartitionError::InvalidDefinition(
                        "list partition must use LIST bounds".into(),
                    ))
                }
            }
        }
        Ok(Self {
            column_index,
            column_name,
            partitions,
        })
    }
}

impl PartitioningScheme for ListPartitioning {
    fn column_index(&self) -> usize {
        self.column_index
    }

    fn column_name(&self) -> &str {
        &self.column_name
    }

    fn partitions(&self) -> &[PartitionMetadata] {
        &self.partitions
    }

    fn locate_partition(&self, value: &Value) -> Option<&PartitionMetadata> {
        for partition in &self.partitions {
            match &partition.bounds {
                PartitionBounds::List { values } => {
                    if values.iter().any(|candidate| candidate == value) {
                        return Some(partition);
                    }
                }
                PartitionBounds::Default => {
                    return Some(partition);
                }
                _ => {}
            }
        }
        None
    }

    fn prune_partitions(
        &self,
        predicate: &Predicate,
        column_type: ColumnType,
    ) -> Result<Option<Vec<String>>, PartitionError> {
        match predicate {
            Predicate::Equals { value, .. } => {
                let coerced = coerce_static_value(value, column_type)
                    .map_err(|err| PartitionError::InvalidDefinition(err.to_string()))?;
                if let Some(partition) = self.locate_partition(&coerced) {
                    return Ok(Some(vec![partition.name.clone()]));
                }
                Ok(Some(Vec::new()))
            }
            _ => Ok(None),
        }
    }

    fn add_partition(&mut self, metadata: PartitionMetadata) -> Result<(), PartitionError> {
        if self
            .partitions
            .iter()
            .any(|partition| partition.name.eq_ignore_ascii_case(&metadata.name))
        {
            return Err(PartitionError::DuplicatePartition(metadata.name));
        }
        if !matches!(
            metadata.bounds,
            PartitionBounds::List { .. } | PartitionBounds::Default
        ) {
            return Err(PartitionError::InvalidDefinition(
                "list partition must use LIST bounds".into(),
            ));
        }
        self.partitions.push(metadata);
        Ok(())
    }

    fn drop_partition(&mut self, name: &str) -> Result<PartitionMetadata, PartitionError> {
        if self.partitions.len() == 1 {
            return Err(PartitionError::InvalidDefinition(
                "cannot drop the last list partition".into(),
            ));
        }
        if let Some(index) = self
            .partitions
            .iter()
            .position(|partition| partition.name.eq_ignore_ascii_case(name))
        {
            Ok(self.partitions.remove(index))
        } else {
            Err(PartitionError::UnknownPartition(name.into()))
        }
    }
}

pub(crate) struct PartitionManager<'a> {
    table: &'a mut Table,
}

impl<'a> PartitionManager<'a> {
    pub(crate) fn new(table: &'a mut Table) -> Self {
        Self { table }
    }

    pub(crate) fn partitions(&self) -> Option<&[PartitionMetadata]> {
        self.table
            .partitioning
            .as_ref()
            .map(|scheme| scheme.partitions())
    }

    pub(crate) fn rebuild(&mut self) -> Result<(), PartitionError> {
        self.table.partitions.clear();
        let Some(scheme) = self.table.partitioning.as_ref() else {
            return Ok(());
        };
        for metadata in scheme.partitions() {
            self.table
                .partitions
                .entry(metadata.name.clone())
                .or_default();
        }
        for (row_index, row) in self.table.rows.iter().enumerate() {
            let value = row.get(scheme.column_index()).ok_or_else(|| {
                PartitionError::InvalidDefinition("partition column index out of bounds".into())
            })?;
            let partition = scheme
                .locate_partition(value)
                .ok_or(PartitionError::ValueOutOfRange)?;
            self.table
                .partitions
                .entry(partition.name.clone())
                .or_default()
                .push(row_index);
        }
        Ok(())
    }

    pub(crate) fn add_partition(
        &mut self,
        metadata: PartitionMetadata,
    ) -> Result<(), PartitionError> {
        let scheme = self
            .table
            .partitioning
            .as_mut()
            .ok_or(PartitionError::MissingPartitioning)?;
        scheme.add_partition(metadata.clone())?;
        self.table.partitions.entry(metadata.name).or_default();
        Ok(())
    }

    pub(crate) fn drop_partition(&mut self, name: &str) -> Result<Vec<usize>, PartitionError> {
        let scheme = self
            .table
            .partitioning
            .as_mut()
            .ok_or(PartitionError::MissingPartitioning)?;
        let removed = scheme.drop_partition(name)?;
        Ok(self
            .table
            .partitions
            .remove(&removed.name)
            .unwrap_or_default())
    }
}

pub(crate) struct InsertRouter<'a> {
    table: &'a mut Table,
}

impl<'a> InsertRouter<'a> {
    pub(crate) fn new(table: &'a mut Table) -> Self {
        Self { table }
    }

    pub(crate) fn route(&mut self, row_index: usize) -> Result<(), PartitionError> {
        let Some(scheme) = self.table.partitioning.as_ref() else {
            return Ok(());
        };
        let value = self
            .table
            .rows
            .get(row_index)
            .and_then(|row| row.get(scheme.column_index()))
            .ok_or_else(|| {
                PartitionError::InvalidDefinition("partition column index out of bounds".into())
            })?;
        let partition = scheme
            .locate_partition(value)
            .ok_or(PartitionError::ValueOutOfRange)?;
        self.table
            .partitions
            .entry(partition.name.clone())
            .or_default()
            .push(row_index);
        Ok(())
    }
}

pub(crate) struct PartitionPruner<'a> {
    table: &'a Table,
    scheme: &'a dyn PartitioningScheme,
}

impl<'a> PartitionPruner<'a> {
    pub(crate) fn new(table: &'a Table, scheme: &'a dyn PartitioningScheme) -> Self {
        Self { table, scheme }
    }

    pub(crate) fn prune(
        &self,
        predicate: &Predicate,
    ) -> Result<Option<Vec<usize>>, PartitionError> {
        if !self.scheme.matches_column(predicate.column_name()) {
            return Ok(None);
        }
        let column_index = self.scheme.column_index();
        let column_type = self.table.columns[column_index].ty;
        let Some(partition_names) = self.scheme.prune_partitions(predicate, column_type)? else {
            return Ok(None);
        };
        let mut rows = Vec::new();
        for name in partition_names {
            if let Some(part_rows) = self.table.partitions.get(&name) {
                rows.extend(part_rows.iter().copied());
            }
        }
        rows.sort_unstable();
        rows.dedup();
        Ok(Some(rows))
    }
}

pub(crate) fn build_partitioning_scheme(
    table: &Table,
    column: &str,
    definition: PartitioningDefinition,
) -> Result<PartitionSchemeDefinition, SqlDatabaseError> {
    let column_index = column_index_in_table(table, column)?;
    let column_name = table.columns[column_index].name.clone();
    let column_type = table.columns[column_index].ty;
    let scheme = match definition {
        PartitioningDefinition::Range { partitions } => {
            let mut coerced = Vec::new();
            for mut metadata in partitions {
                if let PartitionBounds::Range {
                    upper: Some(ref mut value),
                } = metadata.bounds
                {
                    let coerced_value = coerce_static_value(value, column_type)
                        .map_err(|err| SqlDatabaseError::Partition(err.to_string()))?;
                    *value = coerced_value;
                }
                coerced.push(metadata);
            }
            let scheme = RangePartitioning::new(column_index, column_name, coerced)
                .map_err(|err| SqlDatabaseError::Partition(err.to_string()))?;
            PartitionSchemeDefinition::Range(scheme)
        }
        PartitioningDefinition::Hash { partitions } => {
            let scheme = HashPartitioning::new(column_index, column_name, partitions)
                .map_err(|err| SqlDatabaseError::Partition(err.to_string()))?;
            PartitionSchemeDefinition::Hash(scheme)
        }
        PartitioningDefinition::List { partitions } => {
            let mut coerced = Vec::new();
            for mut metadata in partitions {
                if let PartitionBounds::List { ref mut values } = metadata.bounds {
                    for value in values.iter_mut() {
                        let coerced_value = coerce_static_value(value, column_type)
                            .map_err(|err| SqlDatabaseError::Partition(err.to_string()))?;
                        *value = coerced_value;
                    }
                }
                coerced.push(metadata);
            }
            let scheme = ListPartitioning::new(column_index, column_name, coerced)
                .map_err(|err| SqlDatabaseError::Partition(err.to_string()))?;
            PartitionSchemeDefinition::List(scheme)
        }
    };
    Ok(scheme)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColumnType;

    #[test]
    fn range_partition_locate() {
        let scheme = RangePartitioning::new(
            0,
            "ts".into(),
            vec![
                PartitionMetadata {
                    name: "p0".into(),
                    bounds: PartitionBounds::Range {
                        upper: Some(Value::Integer(10)),
                    },
                },
                PartitionMetadata {
                    name: "p1".into(),
                    bounds: PartitionBounds::Range {
                        upper: Some(Value::Integer(20)),
                    },
                },
                PartitionMetadata {
                    name: "pmax".into(),
                    bounds: PartitionBounds::Range { upper: None },
                },
            ],
        )
        .unwrap();
        assert_eq!(
            scheme.locate_partition(&Value::Integer(5)).unwrap().name,
            "p0"
        );
        assert_eq!(
            scheme.locate_partition(&Value::Integer(15)).unwrap().name,
            "p1"
        );
        assert_eq!(
            scheme.locate_partition(&Value::Integer(25)).unwrap().name,
            "pmax"
        );
    }

    #[test]
    fn hash_partition_prune() {
        let scheme = HashPartitioning::new(0, "id".into(), 4).unwrap();
        let predicate = Predicate::Equals {
            column: "id".into(),
            value: Value::Integer(10),
        };
        let partitions = scheme
            .prune_partitions(&predicate, ColumnType::Integer)
            .unwrap()
            .unwrap();
        assert_eq!(partitions.len(), 1);
    }
}
