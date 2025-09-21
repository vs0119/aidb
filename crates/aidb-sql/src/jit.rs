use std::collections::{HashMap, HashSet};
use std::mem;
use std::ptr;
use std::slice;
use std::sync::Arc;

use cranelift_codegen::binemit::{NullStackMapSink, NullTrapSink};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{B1, F64, I8};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage};

use super::{
    coerce_static_value, ColumnType, PlanCacheKey, Predicate, SelectColumns, SelectItem,
    SqlDatabaseError, Table, Value,
};

type JitFilterFn = unsafe extern "C" fn(
    values_ptr: *const u8,
    nulls_ptr: *const u8,
    row_indices_ptr: *const usize,
    len: usize,
    output_ptr: *mut usize,
) -> usize;

type JitProjectionFn = unsafe extern "C" fn(
    ctx_ptr: *const ProjectionContext,
    row_indices_ptr: *const usize,
    len: usize,
);

#[derive(Clone)]
enum FilterKernelKind {
    NoPredicate,
    IntEquals { value: i64 },
    IntBetween { low: i64, high: i64 },
    FloatEquals { value: f64 },
    FloatBetween { low: f64, high: f64 },
    BoolEquals { value: bool },
    IsNull,
}

pub(crate) struct JitManager {
    filter_cache: HashMap<PlanCacheKey, Arc<FilterKernelEntry>>,
    projection_cache: HashMap<PlanCacheKey, Arc<ProjectionKernelEntry>>,
    unsupported_filters: HashSet<PlanCacheKey>,
    unsupported_projections: HashSet<PlanCacheKey>,
}

impl JitManager {
    pub(crate) fn new() -> Self {
        Self {
            filter_cache: HashMap::new(),
            projection_cache: HashMap::new(),
            unsupported_filters: HashSet::new(),
            unsupported_projections: HashSet::new(),
        }
    }

    pub(crate) fn invalidate_keys(&mut self, keys: &[PlanCacheKey]) {
        for key in keys {
            self.filter_cache.remove(key);
            self.projection_cache.remove(key);
            self.unsupported_filters.remove(key);
            self.unsupported_projections.remove(key);
        }
    }

    pub(crate) fn prepare_filter(
        &mut self,
        key: PlanCacheKey,
        table: &Table,
        predicate: Option<&Predicate>,
    ) -> Result<Option<Arc<FilterKernelEntry>>, SqlDatabaseError> {
        if let Some(entry) = self.filter_cache.get(&key) {
            return Ok(Some(entry.clone()));
        }
        if self.unsupported_filters.contains(&key) {
            return Ok(None);
        }

        let (kind, column_index) = match predicate {
            Some(pred) => match analyze_predicate(table, pred)? {
                Some(spec) => spec,
                None => {
                    self.unsupported_filters.insert(key);
                    return Ok(None);
                }
            },
            None => (FilterKernelKind::NoPredicate, None),
        };

        let entry = FilterKernelEntry::compile(kind.clone(), column_index)
            .map_err(|_| SqlDatabaseError::Unsupported)?;
        let entry = Arc::new(entry);
        self.filter_cache.insert(key, entry.clone());
        Ok(Some(entry))
    }

    pub(crate) fn prepare_projection(
        &mut self,
        key: PlanCacheKey,
        table: &Table,
        columns: &SelectColumns,
    ) -> Result<Option<Arc<ProjectionKernelEntry>>, SqlDatabaseError> {
        if let Some(entry) = self.projection_cache.get(&key) {
            return Ok(Some(entry.clone()));
        }
        if self.unsupported_projections.contains(&key) {
            return Ok(None);
        }

        let (indices, names, use_all) = match analyze_projection(table, columns) {
            Some(spec) => spec,
            None => {
                self.unsupported_projections.insert(key);
                return Ok(None);
            }
        };

        let entry = ProjectionKernelEntry::compile(indices, names, use_all)
            .map_err(|_| SqlDatabaseError::Unsupported)?;
        let entry = Arc::new(entry);
        self.projection_cache.insert(key, entry.clone());
        Ok(Some(entry))
    }
}

#[derive(Default)]
struct LoopProfile {
    invocations: usize,
    total_rows: usize,
    hot: bool,
}

pub(crate) struct LoopProfiler {
    stats: HashMap<PlanCacheKey, LoopProfile>,
    hot_invocations: usize,
    hot_rows: usize,
}

impl LoopProfiler {
    pub(crate) fn new() -> Self {
        Self {
            stats: HashMap::new(),
            hot_invocations: 5,
            hot_rows: 4_096,
        }
    }

    pub(crate) fn observe(&mut self, key: &PlanCacheKey, rows: usize) -> bool {
        let entry = self.stats.entry(key.clone()).or_default();
        entry.invocations += 1;
        entry.total_rows += rows;
        if entry.hot {
            true
        } else if entry.invocations >= self.hot_invocations && entry.total_rows >= self.hot_rows {
            entry.hot = true;
            true
        } else {
            false
        }
    }

    pub(crate) fn invalidate_keys(&mut self, keys: &[PlanCacheKey]) {
        for key in keys {
            self.stats.remove(key);
        }
    }
}

struct FilterKernelEntry {
    module: JITModule,
    func: JitFilterFn,
    kind: FilterKernelKind,
    column_index: Option<usize>,
}

impl FilterKernelEntry {
    fn compile(
        kind: FilterKernelKind,
        column_index: Option<usize>,
    ) -> Result<Self, cranelift_module::ModuleError> {
        let mut builder = JITBuilder::new(default_libcall_names());
        let module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let ptr_ty = module.target_config().pointer_type();

        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // values
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // nulls
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // row indices
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // len
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // output
        ctx.func.signature.returns.push(AbiParam::new(ptr_ty)); // count

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            build_filter_body(&mut builder, &module, ptr_ty, &kind);
        }

        let func_id =
            module.declare_function("aidb_filter_kernel", Linkage::Export, &ctx.func.signature)?;
        module.define_function(
            func_id,
            &mut ctx,
            &mut NullTrapSink {},
            &mut NullStackMapSink {},
        )?;
        module.clear_context(&mut ctx);
        module.finalize_definitions()?;
        let code = module.get_finalized_function(func_id);
        let func = unsafe { mem::transmute::<_, JitFilterFn>(code) };

        Ok(Self {
            module,
            func,
            kind,
            column_index,
        })
    }

    fn execute(&self, table: &Table, candidates: &[usize]) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut output = vec![0usize; candidates.len()];
        let count = match (&self.kind, self.column_index) {
            (FilterKernelKind::NoPredicate, _) => unsafe {
                (self.func)(
                    ptr::null(),
                    ptr::null(),
                    candidates.as_ptr(),
                    candidates.len(),
                    output.as_mut_ptr(),
                )
            },
            (FilterKernelKind::IsNull, Some(index)) => {
                let nulls: Vec<u8> = candidates
                    .iter()
                    .map(|&row_index| matches!(table.rows[row_index][index], Value::Null) as u8)
                    .collect();
                unsafe {
                    (self.func)(
                        ptr::null(),
                        nulls.as_ptr(),
                        candidates.as_ptr(),
                        candidates.len(),
                        output.as_mut_ptr(),
                    )
                }
            }
            (
                FilterKernelKind::IntEquals { .. } | FilterKernelKind::IntBetween { .. },
                Some(index),
            ) => {
                let mut values = Vec::with_capacity(candidates.len());
                let mut nulls = Vec::with_capacity(candidates.len());
                for &row_index in candidates {
                    match &table.rows[row_index][index] {
                        Value::Integer(v) => {
                            values.push(*v);
                            nulls.push(0);
                        }
                        Value::Null => {
                            values.push(0);
                            nulls.push(1);
                        }
                        _ => {
                            values.push(0);
                            nulls.push(1);
                        }
                    }
                }
                unsafe {
                    (self.func)(
                        values.as_ptr() as *const u8,
                        nulls.as_ptr(),
                        candidates.as_ptr(),
                        candidates.len(),
                        output.as_mut_ptr(),
                    )
                }
            }
            (
                FilterKernelKind::FloatEquals { .. } | FilterKernelKind::FloatBetween { .. },
                Some(index),
            ) => {
                let mut values = Vec::with_capacity(candidates.len());
                let mut nulls = Vec::with_capacity(candidates.len());
                for &row_index in candidates {
                    match &table.rows[row_index][index] {
                        Value::Float(v) => {
                            values.push(*v);
                            nulls.push(0);
                        }
                        Value::Null => {
                            values.push(0.0);
                            nulls.push(1);
                        }
                        _ => {
                            values.push(0.0);
                            nulls.push(1);
                        }
                    }
                }
                unsafe {
                    (self.func)(
                        values.as_ptr() as *const u8,
                        nulls.as_ptr(),
                        candidates.as_ptr(),
                        candidates.len(),
                        output.as_mut_ptr(),
                    )
                }
            }
            (FilterKernelKind::BoolEquals { .. }, Some(index)) => {
                let mut values = Vec::with_capacity(candidates.len());
                let mut nulls = Vec::with_capacity(candidates.len());
                for &row_index in candidates {
                    match &table.rows[row_index][index] {
                        Value::Boolean(v) => {
                            values.push(*v as u8);
                            nulls.push(0);
                        }
                        Value::Null => {
                            values.push(0);
                            nulls.push(1);
                        }
                        _ => {
                            values.push(0);
                            nulls.push(1);
                        }
                    }
                }
                unsafe {
                    (self.func)(
                        values.as_ptr(),
                        nulls.as_ptr(),
                        candidates.as_ptr(),
                        candidates.len(),
                        output.as_mut_ptr(),
                    )
                }
            }
            _ => 0,
        };

        let count = count.min(output.len());
        output.truncate(count);
        output
    }
}

fn build_filter_body(
    builder: &mut FunctionBuilder,
    module: &JITModule,
    ptr_ty: cranelift_codegen::ir::Type,
    kind: &FilterKernelKind,
) {
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let params = builder.block_params(entry_block);
    let values_ptr = params[0];
    let nulls_ptr = params[1];
    let rows_ptr = params[2];
    let len = params[3];
    let output_ptr = params[4];

    let index_var = Variable::new(0);
    let count_var = Variable::new(1);
    builder.declare_var(index_var, ptr_ty);
    builder.declare_var(count_var, ptr_ty);
    let zero = builder.ins().iconst(ptr_ty, 0);
    builder.def_var(index_var, zero);
    builder.def_var(count_var, zero);

    let loop_block = builder.create_block();
    let exit_block = builder.create_block();
    builder.ins().jump(loop_block, &[]);

    builder.switch_to_block(loop_block);
    let idx = builder.use_var(index_var);
    let cond = builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
    let body_block = builder.create_block();
    builder.ins().brif(cond, body_block, &[], exit_block, &[]);
    builder.seal_block(loop_block);

    builder.switch_to_block(body_block);
    let ptr_bytes = module.target_config().pointer_bytes() as i64;
    let idx_value = builder.use_var(index_var);

    let mut pass_condition = builder.ins().bconst(B1, true);
    match kind {
        FilterKernelKind::NoPredicate => {}
        FilterKernelKind::IsNull => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            pass_condition = builder.ins().icmp_imm(IntCC::NotEqual, null_flag, 0);
        }
        FilterKernelKind::IntEquals { value } => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            let not_null = builder.ins().icmp_imm(IntCC::Equal, null_flag, 0);
            let offset = builder.ins().imul_imm(idx_value, 8);
            let value_addr = builder.ins().iadd(values_ptr, offset);
            let loaded = builder.ins().load(ptr_ty, MemFlags::new(), value_addr, 0);
            let cmp = builder.ins().icmp_imm(IntCC::Equal, loaded, *value);
            pass_condition = builder.ins().band(not_null, cmp);
        }
        FilterKernelKind::IntBetween { low, high } => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            let not_null = builder.ins().icmp_imm(IntCC::Equal, null_flag, 0);
            let offset = builder.ins().imul_imm(idx_value, 8);
            let value_addr = builder.ins().iadd(values_ptr, offset);
            let loaded = builder.ins().load(ptr_ty, MemFlags::new(), value_addr, 0);
            let ge_low = builder
                .ins()
                .icmp_imm(IntCC::SignedGreaterThanOrEqual, loaded, *low);
            let le_high = builder
                .ins()
                .icmp_imm(IntCC::SignedLessThanOrEqual, loaded, *high);
            let within = builder.ins().band(ge_low, le_high);
            pass_condition = builder.ins().band(not_null, within);
        }
        FilterKernelKind::FloatEquals { value } => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            let not_null = builder.ins().icmp_imm(IntCC::Equal, null_flag, 0);
            let offset = builder.ins().imul_imm(idx_value, 8);
            let value_addr = builder.ins().iadd(values_ptr, offset);
            let loaded = builder.ins().load(F64, MemFlags::new(), value_addr, 0);
            let cmp = builder
                .ins()
                .fcmp(FloatCC::Equal, loaded, builder.ins().f64const(*value));
            pass_condition = builder.ins().band(not_null, cmp);
        }
        FilterKernelKind::FloatBetween { low, high } => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            let not_null = builder.ins().icmp_imm(IntCC::Equal, null_flag, 0);
            let offset = builder.ins().imul_imm(idx_value, 8);
            let value_addr = builder.ins().iadd(values_ptr, offset);
            let loaded = builder.ins().load(F64, MemFlags::new(), value_addr, 0);
            let low_const = builder.ins().f64const(*low);
            let high_const = builder.ins().f64const(*high);
            let ge_low = builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, loaded, low_const);
            let le_high = builder
                .ins()
                .fcmp(FloatCC::LessThanOrEqual, loaded, high_const);
            let within = builder.ins().band(ge_low, le_high);
            pass_condition = builder.ins().band(not_null, within);
        }
        FilterKernelKind::BoolEquals { value } => {
            let null_addr = builder.ins().iadd(nulls_ptr, idx_value);
            let null_flag = builder.ins().load(I8, MemFlags::new(), null_addr, 0);
            let not_null = builder.ins().icmp_imm(IntCC::Equal, null_flag, 0);
            let value_addr = builder.ins().iadd(values_ptr, idx_value);
            let loaded = builder.ins().load(I8, MemFlags::new(), value_addr, 0);
            let cmp = builder.ins().icmp_imm(IntCC::Equal, loaded, *value as i64);
            pass_condition = builder.ins().band(not_null, cmp);
        }
    }

    let store_block = builder.create_block();
    let skip_block = builder.create_block();
    let cont_block = builder.create_block();
    builder
        .ins()
        .brif(pass_condition, store_block, &[], skip_block, &[]);
    builder.seal_block(store_block);
    builder.seal_block(skip_block);
    builder.seal_block(body_block);

    builder.switch_to_block(store_block);
    let count = builder.use_var(count_var);
    let row_offset = if ptr_bytes == 1 {
        builder.use_var(index_var)
    } else {
        builder
            .ins()
            .imul_imm(builder.use_var(index_var), ptr_bytes as i64)
    };
    let row_addr = builder.ins().iadd(rows_ptr, row_offset);
    let row_value = builder.ins().load(ptr_ty, MemFlags::new(), row_addr, 0);
    let dest_offset = if ptr_bytes == 1 {
        count
    } else {
        builder.ins().imul_imm(count, ptr_bytes as i64)
    };
    let dest_addr = builder.ins().iadd(output_ptr, dest_offset);
    builder
        .ins()
        .store(MemFlags::new(), row_value, dest_addr, 0);
    let next_count = builder.ins().iadd_imm(count, 1);
    builder.def_var(count_var, next_count);
    builder.ins().jump(cont_block, &[]);

    builder.switch_to_block(skip_block);
    builder.ins().jump(cont_block, &[]);

    builder.switch_to_block(cont_block);
    let next_idx = builder.ins().iadd_imm(builder.use_var(index_var), 1);
    builder.def_var(index_var, next_idx);
    builder.ins().jump(loop_block, &[]);
    builder.seal_block(cont_block);

    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    let count = builder.use_var(count_var);
    builder.ins().return_(&[count]);
}

struct ProjectionKernelEntry {
    module: JITModule,
    func: JitProjectionFn,
    column_indices: Vec<usize>,
    column_names: Vec<String>,
    use_all: bool,
}

impl ProjectionKernelEntry {
    fn compile(
        column_indices: Vec<usize>,
        column_names: Vec<String>,
        use_all: bool,
    ) -> Result<Self, cranelift_module::ModuleError> {
        let mut builder = JITBuilder::new(default_libcall_names());
        builder.symbol(
            "aidb_jit_emit_projection",
            aidb_jit_emit_projection as *const u8,
        );
        let module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let ptr_ty = module.target_config().pointer_type();

        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // ctx
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // rows
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // len

        let mut emit_sig = module.make_signature();
        emit_sig.params.push(AbiParam::new(ptr_ty));
        emit_sig.params.push(AbiParam::new(ptr_ty));
        let emit_id =
            module.declare_function("aidb_jit_emit_projection", Linkage::Import, &emit_sig)?;

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            build_projection_body(&mut builder, &module, ptr_ty, emit_id);
        }

        let func_id = module.declare_function(
            "aidb_projection_kernel",
            Linkage::Export,
            &ctx.func.signature,
        )?;
        module.define_function(
            func_id,
            &mut ctx,
            &mut NullTrapSink {},
            &mut NullStackMapSink {},
        )?;
        module.clear_context(&mut ctx);
        module.finalize_definitions()?;
        let code = module.get_finalized_function(func_id);
        let func = unsafe { mem::transmute::<_, JitProjectionFn>(code) };

        Ok(Self {
            module,
            func,
            column_indices,
            column_names,
            use_all,
        })
    }

    fn execute(&self, table: &Table, row_indices: &[usize]) -> Vec<Vec<Value>> {
        if row_indices.is_empty() {
            return Vec::new();
        }

        let mut rows = Vec::with_capacity(row_indices.len());
        let mut context = ProjectionContext {
            table_ptr: table as *const Table,
            column_indices_ptr: self.column_indices.as_ptr(),
            column_count: self.column_indices.len(),
            output_ptr: &mut rows as *mut Vec<Vec<Value>>,
            use_all: if self.use_all { 1 } else { 0 },
        };
        unsafe {
            (self.func)(
                &context as *const ProjectionContext,
                row_indices.as_ptr(),
                row_indices.len(),
            );
        }
        rows
    }

    fn column_names(&self) -> &[String] {
        &self.column_names
    }
}

fn build_projection_body(
    builder: &mut FunctionBuilder,
    module: &JITModule,
    ptr_ty: cranelift_codegen::ir::Type,
    emit_id: FuncId,
) {
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let params = builder.block_params(entry_block);
    let ctx_ptr = params[0];
    let rows_ptr = params[1];
    let len = params[2];

    let index_var = Variable::new(0);
    builder.declare_var(index_var, ptr_ty);
    let zero = builder.ins().iconst(ptr_ty, 0);
    builder.def_var(index_var, zero);

    let loop_block = builder.create_block();
    let exit_block = builder.create_block();
    builder.ins().jump(loop_block, &[]);

    builder.switch_to_block(loop_block);
    let idx = builder.use_var(index_var);
    let cond = builder.ins().icmp(IntCC::UnsignedLessThan, idx, len);
    let body_block = builder.create_block();
    builder.ins().brif(cond, body_block, &[], exit_block, &[]);
    builder.seal_block(loop_block);

    builder.switch_to_block(body_block);
    let ptr_bytes = module.target_config().pointer_bytes() as i64;
    let idx_val = builder.use_var(index_var);
    let offset = if ptr_bytes == 1 {
        idx_val
    } else {
        builder.ins().imul_imm(idx_val, ptr_bytes as i64)
    };
    let row_addr = builder.ins().iadd(rows_ptr, offset);
    let row_value = builder.ins().load(ptr_ty, MemFlags::new(), row_addr, 0);

    let emit_func = module.declare_func_in_func(emit_id, builder.func);
    builder.ins().call(emit_func, &[ctx_ptr, row_value]);

    let next_block = builder.create_block();
    builder.ins().jump(next_block, &[]);
    builder.seal_block(body_block);

    builder.switch_to_block(next_block);
    let next_idx = builder.ins().iadd_imm(builder.use_var(index_var), 1);
    builder.def_var(index_var, next_idx);
    builder.ins().jump(loop_block, &[]);
    builder.seal_block(next_block);

    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    builder.ins().return_(&[]);
}

#[repr(C)]
struct ProjectionContext {
    table_ptr: *const Table,
    column_indices_ptr: *const usize,
    column_count: usize,
    output_ptr: *mut Vec<Vec<Value>>,
    use_all: u8,
}

unsafe extern "C" fn aidb_jit_emit_projection(ctx_ptr: *const ProjectionContext, row_index: usize) {
    let ctx = &*ctx_ptr;
    let table = &*ctx.table_ptr;
    let output = &mut *ctx.output_ptr;
    if ctx.use_all != 0 {
        output.push(table.rows[row_index].clone());
    } else {
        let indices = slice::from_raw_parts(ctx.column_indices_ptr, ctx.column_count);
        let mut row = Vec::with_capacity(indices.len());
        for &col in indices {
            row.push(table.rows[row_index][col].clone());
        }
        output.push(row);
    }
}

fn analyze_predicate(
    table: &Table,
    predicate: &Predicate,
) -> Result<Option<(FilterKernelKind, Option<usize>)>, SqlDatabaseError> {
    match predicate {
        Predicate::Equals { column, value } => {
            let index = match column_index(table, column) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            let column_type = table.columns[index].ty;
            let coerced = coerce_static_value(value, column_type)?;
            let kind = match (column_type, coerced) {
                (ColumnType::Integer, Value::Integer(v)) => {
                    FilterKernelKind::IntEquals { value: v }
                }
                (ColumnType::Float, Value::Float(v)) => FilterKernelKind::FloatEquals { value: v },
                (ColumnType::Boolean, Value::Boolean(v)) => {
                    FilterKernelKind::BoolEquals { value: v }
                }
                _ => return Ok(None),
            };
            Ok(Some((kind, Some(index))))
        }
        Predicate::Between { column, start, end } => {
            let index = match column_index(table, column) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            let column_type = table.columns[index].ty;
            let start_val = coerce_static_value(start, column_type)?;
            let end_val = coerce_static_value(end, column_type)?;
            let kind = match (column_type, start_val, end_val) {
                (ColumnType::Integer, Value::Integer(a), Value::Integer(b)) => {
                    let (low, high) = if a <= b { (a, b) } else { (b, a) };
                    FilterKernelKind::IntBetween { low, high }
                }
                (ColumnType::Float, Value::Float(a), Value::Float(b)) => {
                    let (low, high) = if a <= b { (a, b) } else { (b, a) };
                    FilterKernelKind::FloatBetween { low, high }
                }
                _ => return Ok(None),
            };
            Ok(Some((kind, Some(index))))
        }
        Predicate::IsNull { column } => {
            let index = match column_index(table, column) {
                Some(idx) => idx,
                None => return Ok(None),
            };
            Ok(Some((FilterKernelKind::IsNull, Some(index))))
        }
        _ => Ok(None),
    }
}

fn analyze_projection(
    table: &Table,
    columns: &SelectColumns,
) -> Option<(Vec<usize>, Vec<String>, bool)> {
    match columns {
        SelectColumns::All => {
            let indices: Vec<_> = (0..table.columns.len()).collect();
            let names = table.columns.iter().map(|c| c.name.clone()).collect();
            Some((indices, names, true))
        }
        SelectColumns::Some(items) => {
            let mut indices = Vec::with_capacity(items.len());
            let mut names = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    SelectItem::Column(name) => {
                        let idx = column_index(table, name)?;
                        indices.push(idx);
                        names.push(table.columns[idx].name.clone());
                    }
                    SelectItem::WindowFunction(_) => return None,
                }
            }
            Some((indices, names, false))
        }
    }
}

fn column_index(table: &Table, name: &str) -> Option<usize> {
    table
        .columns
        .iter()
        .position(|c| c.name.eq_ignore_ascii_case(name))
}
