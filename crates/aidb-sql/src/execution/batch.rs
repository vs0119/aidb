use crate::{ColumnType, Value};

#[derive(Clone)]
pub(crate) struct ColumnVector<'a> {
    values: Vec<&'a Value>,
}

impl<'a> ColumnVector<'a> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, value: &'a Value) {
        self.values.push(value);
    }

    pub(crate) fn value(&self, index: usize) -> &'a Value {
        self.values[index]
    }

    pub(crate) fn values(&self) -> &[&'a Value] {
        &self.values
    }

    pub(crate) fn retain_by_mask(&mut self, mask: &[bool]) {
        debug_assert_eq!(mask.len(), self.values.len());
        let mut write = 0usize;
        for (read, keep) in mask.iter().copied().enumerate() {
            if keep {
                if write != read {
                    self.values.swap(write, read);
                }
                write += 1;
            }
        }
        self.values.truncate(write);
    }
}

pub(crate) struct ColumnarBatch<'a> {
    column_types: Vec<ColumnType>,
    columns: Vec<ColumnVector<'a>>,
    row_indices: Vec<usize>,
}

impl<'a> ColumnarBatch<'a> {
    pub(crate) fn with_capacity(column_types: &[ColumnType], capacity: usize) -> Self {
        let columns = column_types
            .iter()
            .copied()
            .map(|_| ColumnVector::new(capacity))
            .collect();
        Self {
            column_types: column_types.to_vec(),
            columns,
            row_indices: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push_row(&mut self, row_index: usize, row: &'a [Value]) {
        debug_assert_eq!(self.columns.len(), row.len());
        self.row_indices.push(row_index);
        for (column, value) in self.columns.iter_mut().zip(row.iter()) {
            column.push(value);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.row_indices.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.row_indices.is_empty()
    }

    pub(crate) fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub(crate) fn column(&self, index: usize) -> &ColumnVector<'a> {
        &self.columns[index]
    }

    pub(crate) fn columns(&self) -> &[ColumnVector<'a>] {
        &self.columns
    }

    pub(crate) fn retain_by_mask(&mut self, mask: &[bool]) {
        debug_assert_eq!(mask.len(), self.len());
        let mut write = 0usize;
        for (read, keep) in mask.iter().copied().enumerate() {
            if keep {
                if write != read {
                    self.row_indices.swap(write, read);
                }
                write += 1;
            }
        }
        self.row_indices.truncate(write);
        for column in &mut self.columns {
            column.retain_by_mask(mask);
        }
    }

    pub(crate) fn project(&self, indices: &[usize]) -> ColumnarBatch<'a> {
        let mut projected_types = Vec::with_capacity(indices.len());
        let mut projected_columns = Vec::with_capacity(indices.len());
        for &idx in indices {
            projected_types.push(self.column_types[idx]);
            projected_columns.push(self.columns[idx].clone());
        }
        ColumnarBatch {
            column_types: projected_types,
            columns: projected_columns,
            row_indices: self.row_indices.clone(),
        }
    }
}
