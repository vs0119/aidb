use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use tokio::sync::{OwnedRwLockReadGuard, RwLock};

use crate::page::{Page, PAGE_HEADER_SIZE, PAGE_SIZE};
use crate::{BufferPool, PageId, PageManager, Result, RowId, StorageEngineError, VectorRow};

const NODE_MAGIC: u32 = 0xB1_700517;
const PAGE_TYPE_BTREE_INTERNAL: u8 = 0x21;
const PAGE_TYPE_BTREE_LEAF: u8 = 0x22;
const NULL_PAGE: PageId = 0;

#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize)]
#[repr(C)]
struct NodeHeader {
    magic: u32,
    node_type: u8,
    level: u8,
    reserved: u16,
    key_count: u16,
    payload_size: u32,
    parent: PageId,
    next_leaf: PageId,
    prev_leaf: PageId,
}

impl Default for NodeHeader {
    fn default() -> Self {
        Self {
            magic: NODE_MAGIC,
            node_type: 1,
            level: 0,
            reserved: 0,
            key_count: 0,
            payload_size: 0,
            parent: NULL_PAGE,
            next_leaf: NULL_PAGE,
            prev_leaf: NULL_PAGE,
        }
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct NodePayload<K>
where
    K: BTreeKey + for<'de> serde::Deserialize<'de>,
{
    keys: Vec<K>,
    children: Vec<PageId>,
    values: Vec<Vec<RowId>>,
}

#[derive(Debug, Clone)]
struct BTreeNode<K: BTreeKey> {
    page_id: PageId,
    header: NodeHeader,
    keys: Vec<K>,
    children: Vec<PageId>,
    values: Vec<Vec<RowId>>,
}

impl<K: BTreeKey> BTreeNode<K> {
    fn new_leaf(page_id: PageId, prev_leaf: Option<PageId>, next_leaf: Option<PageId>) -> Self {
        let mut header = NodeHeader::default();
        header.node_type = 1;
        header.level = 0;
        header.prev_leaf = prev_leaf.unwrap_or(NULL_PAGE);
        header.next_leaf = next_leaf.unwrap_or(NULL_PAGE);
        Self {
            page_id,
            header,
            keys: Vec::new(),
            children: Vec::new(),
            values: Vec::new(),
        }
    }

    fn new_internal(page_id: PageId, level: u8, parent: Option<PageId>) -> Self {
        let mut header = NodeHeader::default();
        header.node_type = 0;
        header.level = level;
        header.parent = parent.unwrap_or(NULL_PAGE);
        header.prev_leaf = NULL_PAGE;
        header.next_leaf = NULL_PAGE;
        Self {
            page_id,
            header,
            keys: Vec::new(),
            children: Vec::new(),
            values: Vec::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.header.node_type == 1
    }

    fn parent(&self) -> Option<PageId> {
        if self.header.parent == NULL_PAGE {
            None
        } else {
            Some(self.header.parent)
        }
    }

    fn set_parent(&mut self, parent: Option<PageId>) {
        self.header.parent = parent.unwrap_or(NULL_PAGE);
    }

    fn level(&self) -> u8 {
        self.header.level
    }

    fn set_level(&mut self, level: u8) {
        self.header.level = level;
    }

    fn next_leaf(&self) -> Option<PageId> {
        if self.header.next_leaf == NULL_PAGE {
            None
        } else {
            Some(self.header.next_leaf)
        }
    }

    fn set_next_leaf(&mut self, page: Option<PageId>) {
        self.header.next_leaf = page.unwrap_or(NULL_PAGE);
    }

    fn prev_leaf(&self) -> Option<PageId> {
        if self.header.prev_leaf == NULL_PAGE {
            None
        } else {
            Some(self.header.prev_leaf)
        }
    }

    fn set_prev_leaf(&mut self, page: Option<PageId>) {
        self.header.prev_leaf = page.unwrap_or(NULL_PAGE);
    }

    fn sync_header(&mut self) {
        self.header.key_count = self.keys.len() as u16;
    }

    fn payload(&self) -> NodePayload<K> {
        NodePayload {
            keys: self.keys.clone(),
            children: self.children.clone(),
            values: self.values.clone(),
        }
    }
}

pub trait BTreeKey: Ord + Clone + Serialize + serde::de::DeserializeOwned + Send + Sync + 'static {}

impl<T> BTreeKey for T where T: Ord + Clone + Serialize + serde::de::DeserializeOwned + Send + Sync + 'static {}

#[derive(Debug, Clone)]
struct TreeMetadata {
    root: PageId,
    height: u16,
}

impl TreeMetadata {
    fn new(root: PageId) -> Self {
        Self { root, height: 1 }
    }
}

pub struct BPlusTree<K: BTreeKey> {
    buffer_pool: Arc<BufferPool>,
    page_manager: Arc<PageManager>,
    fanout: usize,
    metadata: Arc<RwLock<TreeMetadata>>,
    structure_lock: Arc<RwLock<()>>,
    _marker: PhantomData<K>,
}

impl<K: BTreeKey> BPlusTree<K> {
    pub async fn create(
        buffer_pool: Arc<BufferPool>,
        page_manager: Arc<PageManager>,
        fanout: usize,
    ) -> Result<Self> {
        let root_page = page_manager.allocate_page().await?;
        let tree = Self::with_root(buffer_pool, page_manager, fanout, root_page, 1);
        let mut root = BTreeNode::new_leaf(root_page, None, None);
        root.sync_header();
        tree.write_node(&root).await?;
        Ok(tree)
    }

    pub fn with_root(
        buffer_pool: Arc<BufferPool>,
        page_manager: Arc<PageManager>,
        fanout: usize,
        root_page: PageId,
        height: u16,
    ) -> Self {
        Self {
            buffer_pool,
            page_manager,
            fanout: fanout.max(3),
            metadata: Arc::new(RwLock::new(TreeMetadata {
                root: root_page,
                height,
            })),
            structure_lock: Arc::new(RwLock::new(())),
            _marker: PhantomData,
        }
    }

    pub async fn root_page_id(&self) -> PageId {
        self.metadata.read().await.root
    }
    pub async fn search(&self, key: &K) -> Result<Vec<RowId>> {
        let _guard = self.structure_lock.read().await;
        let mut node = self.read_root().await?;
        loop {
            if node.is_leaf() {
                match node.keys.binary_search(key) {
                    Ok(idx) => return Ok(node.values[idx].clone()),
                    Err(_) => return Ok(Vec::new()),
                }
            }
            let idx = match node.keys.binary_search(key) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
            let child =
                node.children
                    .get(idx)
                    .copied()
                    .ok_or(StorageEngineError::PageCorruption(
                        "invalid child pointer".to_string(),
                    ))?;
            node = self.read_node(child).await?;
        }
    }

    pub async fn insert(&self, key: K, row_id: RowId) -> Result<()> {
        let _guard = self.structure_lock.write().await;
        let mut node = self.read_root().await?;
        let mut path = Vec::new();
        loop {
            if node.is_leaf() {
                break;
            }
            let idx = match node.keys.binary_search(&key) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
            path.push((node.page_id, idx));
            let child =
                node.children
                    .get(idx)
                    .copied()
                    .ok_or(StorageEngineError::PageCorruption(
                        "invalid child pointer".to_string(),
                    ))?;
            node = self.read_node(child).await?;
        }

        match node.keys.binary_search(&key) {
            Ok(idx) => {
                if !node.values[idx].contains(&row_id) {
                    node.values[idx].push(row_id);
                }
            }
            Err(idx) => {
                node.keys.insert(idx, key.clone());
                node.values.insert(idx, vec![row_id]);
            }
        }
        node.sync_header();

        if node.keys.len() >= self.max_keys() {
            self.split_leaf(node, path).await
        } else {
            self.write_node(&node).await
        }
    }

    pub async fn remove(&self, key: &K, row_id: RowId) -> Result<bool> {
        let _guard = self.structure_lock.write().await;
        let mut node = self.read_root().await?;
        let mut path = Vec::new();
        loop {
            if node.is_leaf() {
                break;
            }
            let idx = match node.keys.binary_search(key) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
            path.push((node.page_id, idx));
            let child =
                node.children
                    .get(idx)
                    .copied()
                    .ok_or(StorageEngineError::PageCorruption(
                        "invalid child pointer".to_string(),
                    ))?;
            node = self.read_node(child).await?;
        }

        let Some(index) = node.keys.binary_search(key).ok() else {
            return Ok(false);
        };

        if let Some(pos) = node.values[index].iter().position(|id| id == &row_id) {
            node.values[index].remove(pos);
            if node.values[index].is_empty() {
                node.keys.remove(index);
                node.values.remove(index);
            }
        } else {
            return Ok(false);
        }
        node.sync_header();

        if path.is_empty() && node.keys.is_empty() {
            self.write_node(&node).await?;
            return Ok(true);
        }

        if node.keys.len() < self.min_keys() {
            self.rebalance_after_delete(node, path).await?;
        } else {
            self.write_node(&node).await?;
        }

        Ok(true)
    }

    pub async fn range_cursor(
        self: &Arc<Self>,
        lower_bound: Option<K>,
        upper_bound: Option<K>,
        upper_inclusive: bool,
    ) -> Result<BTreeCursor<K>> {
        let guard = self.structure_lock.clone().read_owned().await;
        let mut node;
        let mut position;
        if let Some(ref key) = lower_bound {
            let (found, idx) = self.find_leaf_with_key(key).await?;
            node = found;
            position = idx;
        } else {
            node = self.leftmost_leaf().await?;
            position = 0;
        }

        while position >= node.keys.len() {
            if let Some(next) = node.next_leaf() {
                node = self.read_node(next).await?;
                position = 0;
            } else {
                break;
            }
        }

        Ok(BTreeCursor {
            tree: Arc::clone(self),
            lock_guard: guard,
            node,
            position,
            upper_bound,
            upper_inclusive,
        })
    }

    async fn read_root(&self) -> Result<BTreeNode<K>> {
        let root = self.metadata.read().await.root;
        self.read_node(root).await
    }

    async fn read_node(&self, page_id: PageId) -> Result<BTreeNode<K>> {
        let page = self.buffer_pool.get_page(page_id).await?;
        let page_guard = page.read().await;
        let header_end = PAGE_HEADER_SIZE + 32; // Size estimate for NodeHeader
        let header_slice = &page_guard.data[PAGE_HEADER_SIZE..header_end];
        let mut header: NodeHeader = if header_slice.iter().all(|b| *b == 0) {
            NodeHeader::default()
        } else {
            bincode::deserialize(header_slice).unwrap_or_default()
        };

        if header.magic != NODE_MAGIC {
            header = NodeHeader::default();
            header.node_type = if page_guard.header().page_type == PAGE_TYPE_BTREE_INTERNAL {
                0
            } else {
                1
            };
        }

        let header_bytes = bincode::serialize(&header)
            .map_err(|err| StorageEngineError::PageCorruption(err.to_string()))?;
        let payload_start = PAGE_HEADER_SIZE + header_bytes.len();
        let payload_end = payload_start + header.payload_size as usize;
        let payload = if header.payload_size == 0 {
            NodePayload {
                keys: Vec::new(),
                children: Vec::new(),
                values: Vec::new(),
            }
        } else {
            let end = payload_end.min(PAGE_SIZE);
            bincode::deserialize(&page_guard.data[payload_start..end])
                .map_err(|err| StorageEngineError::PageCorruption(err.to_string()))?
        };

        Ok(BTreeNode {
            page_id,
            header,
            keys: payload.keys,
            children: payload.children,
            values: payload.values,
        })
    }

    async fn write_node(&self, node: &BTreeNode<K>) -> Result<()> {
        let page = self.buffer_pool.get_page(node.page_id).await?;
        let mut guard = page.write().await;
        let payload = node.payload();
        let encoded = bincode::serialize(&payload)
            .map_err(|err| StorageEngineError::PageCorruption(err.to_string()))?;
        let mut header = node.header;
        header.payload_size = encoded.len() as u32;
        let header_bytes = bincode::serialize(&header)
            .map_err(|err| StorageEngineError::PageCorruption(err.to_string()))?;
        let payload_start = PAGE_HEADER_SIZE + header_bytes.len();
        let payload_end = payload_start + encoded.len();
        if payload_end > PAGE_SIZE {
            return Err(StorageEngineError::OutOfSpace);
        }
        self.initialize_page_header(&mut guard, node, encoded.len());
        let header_end = PAGE_HEADER_SIZE + header_bytes.len();
        guard.data[PAGE_HEADER_SIZE..header_end].copy_from_slice(&header_bytes);
        guard.data[payload_start..payload_end].copy_from_slice(&encoded);
        if payload_end < PAGE_SIZE {
            for byte in &mut guard.data[payload_end..] {
                *byte = 0;
            }
        }
        guard.dirty = true;
        Ok(())
    }

    fn initialize_page_header(&self, page: &mut Page, node: &BTreeNode<K>, payload_len: usize) {
        let header = page.header_mut();
        header.page_type = if node.is_leaf() {
            PAGE_TYPE_BTREE_LEAF
        } else {
            PAGE_TYPE_BTREE_INTERNAL
        };
        header.row_count = node.keys.len() as u16;
        header.free_space = (PAGE_SIZE - PAGE_HEADER_SIZE - payload_len) as u16;
    }

    fn max_keys(&self) -> usize {
        self.fanout - 1
    }

    fn min_keys(&self) -> usize {
        (self.max_keys() / 2).max(1)
    }

    async fn leftmost_leaf(&self) -> Result<BTreeNode<K>> {
        let mut node = self.read_root().await?;
        loop {
            if node.is_leaf() {
                return Ok(node);
            }
            let child = *node
                .children
                .first()
                .ok_or(StorageEngineError::PageCorruption(
                    "missing child pointer".to_string(),
                ))?;
            node = self.read_node(child).await?;
        }
    }

    async fn find_leaf_with_key(&self, key: &K) -> Result<(BTreeNode<K>, usize)> {
        let mut node = self.read_root().await?;
        loop {
            if node.is_leaf() {
                let idx = match node.keys.binary_search(key) {
                    Ok(pos) => pos,
                    Err(pos) => pos,
                };
                return Ok((node, idx));
            }
            let idx = match node.keys.binary_search(key) {
                Ok(pos) => pos + 1,
                Err(pos) => pos,
            };
            let child =
                node.children
                    .get(idx)
                    .copied()
                    .ok_or(StorageEngineError::PageCorruption(
                        "invalid child pointer".to_string(),
                    ))?;
            node = self.read_node(child).await?;
        }
    }

    async fn split_leaf(
        &self,
        mut node: BTreeNode<K>,
        mut path: Vec<(PageId, usize)>,
    ) -> Result<()> {
        let split_point = node.keys.len() / 2;
        let mut right_keys = node.keys.split_off(split_point);
        let mut right_values = node.values.split_off(split_point);
        let promoted_key =
            right_keys
                .first()
                .cloned()
                .ok_or(StorageEngineError::PageCorruption(
                    "leaf split without keys".to_string(),
                ))?;
        let new_page = self.page_manager.allocate_page().await?;
        let mut right_node = BTreeNode::new_leaf(new_page, Some(node.page_id), node.next_leaf());
        right_node.keys.append(&mut right_keys);
        right_node.values.append(&mut right_values);
        right_node.set_parent(node.parent());
        right_node.sync_header();

        if let Some(next_page) = right_node.next_leaf() {
            let mut next_node = self.read_node(next_page).await?;
            next_node.set_prev_leaf(Some(right_node.page_id));
            next_node.sync_header();
            self.write_node(&next_node).await?;
        }

        node.set_next_leaf(Some(right_node.page_id));
        node.sync_header();

        self.insert_into_parent(node, promoted_key, right_node, path)
            .await
    }

    async fn split_internal(
        &self,
        mut node: BTreeNode<K>,
        mut path: Vec<(PageId, usize)>,
    ) -> Result<()> {
        let split_point = node.keys.len() / 2;
        let promoted_key =
            node.keys
                .get(split_point)
                .cloned()
                .ok_or(StorageEngineError::PageCorruption(
                    "internal split missing promoted key".to_string(),
                ))?;
        let mut right_keys = node.keys.split_off(split_point + 1);
        let mut right_children = node.children.split_off(split_point + 1);

        node.keys.truncate(split_point);
        node.sync_header();

        let new_page = self.page_manager.allocate_page().await?;
        let mut right_node = BTreeNode::new_internal(new_page, node.level(), node.parent());
        right_node.keys.append(&mut right_keys);
        right_node.children.append(&mut right_children);
        right_node.sync_header();

        for child in &right_node.children {
            let mut child_node = self.read_node(*child).await?;
            child_node.set_parent(Some(right_node.page_id));
            child_node.sync_header();
            self.write_node(&child_node).await?;
        }

        self.insert_into_parent(node, promoted_key, right_node, path)
            .await
    }

    async fn insert_into_parent(
        &self,
        mut left: BTreeNode<K>,
        promoted_key: K,
        mut right: BTreeNode<K>,
        mut path: Vec<(PageId, usize)>,
    ) -> Result<()> {
        if path.is_empty() {
            let new_root_page = self.page_manager.allocate_page().await?;
            let mut root = BTreeNode::new_internal(new_root_page, left.level() + 1, None);
            root.keys.push(promoted_key);
            root.children.push(left.page_id);
            root.children.push(right.page_id);
            root.sync_header();
            left.set_parent(Some(root.page_id));
            right.set_parent(Some(root.page_id));
            left.sync_header();
            right.sync_header();
            self.write_node(&left).await?;
            self.write_node(&right).await?;
            self.write_node(&root).await?;
            let mut metadata = self.metadata.write().await;
            metadata.root = root.page_id;
            metadata.height += 1;
            return Ok(());
        }

        let (parent_page, child_index) = path.pop().unwrap();
        let mut parent = self.read_node(parent_page).await?;
        right.set_parent(Some(parent_page));
        parent.keys.insert(child_index, promoted_key);
        parent.children.insert(child_index + 1, right.page_id);
        parent.sync_header();
        left.sync_header();
        right.sync_header();
        self.write_node(&left).await?;
        self.write_node(&right).await?;

        if parent.keys.len() >= self.max_keys() {
            self.split_internal(parent, path).await
        } else {
            self.write_node(&parent).await
        }
    }

    async fn rebalance_after_delete(
        &self,
        mut node: BTreeNode<K>,
        mut path: Vec<(PageId, usize)>,
    ) -> Result<()> {
        if path.is_empty() {
            if node.is_leaf() {
                self.write_node(&node).await?;
                return Ok(());
            }
            if node.keys.is_empty() {
                if let Some(&child) = node.children.first() {
                    let mut child_node = self.read_node(child).await?;
                    child_node.set_parent(None);
                    child_node.sync_header();
                    self.write_node(&child_node).await?;
                    let mut metadata = self.metadata.write().await;
                    metadata.root = child;
                    if metadata.height > 0 {
                        metadata.height -= 1;
                    }
                    return Ok(());
                }
            }
            self.write_node(&node).await?;
            return Ok(());
        }

        if node.keys.len() >= self.min_keys() {
            self.write_node(&node).await?;
            return Ok(());
        }

        let (parent_page, child_index) = path.pop().unwrap();
        let mut parent = self.read_node(parent_page).await?;
        let left_page = if child_index > 0 {
            Some(parent.children[child_index - 1])
        } else {
            None
        };
        let right_page = if child_index + 1 < parent.children.len() {
            Some(parent.children[child_index + 1])
        } else {
            None
        };

        if let Some(left_page) = left_page {
            let mut left = self.read_node(left_page).await?;
            if left.keys.len() > self.min_keys() {
                if node.is_leaf() {
                    let borrowed_key = left.keys.pop().unwrap();
                    let borrowed_vals = left.values.pop().unwrap();
                    node.keys.insert(0, borrowed_key.clone());
                    node.values.insert(0, borrowed_vals);
                    parent.keys[child_index - 1] =
                        node.keys
                            .first()
                            .cloned()
                            .ok_or(StorageEngineError::PageCorruption(
                                "empty node after borrow".to_string(),
                            ))?;
                } else {
                    let borrowed_child = left.children.pop().unwrap();
                    let borrowed_key = left.keys.pop().unwrap();
                    node.keys.insert(0, parent.keys[child_index - 1].clone());
                    node.children.insert(0, borrowed_child);
                    parent.keys[child_index - 1] = borrowed_key.clone();
                    let mut child_node = self.read_node(node.children[0]).await?;
                    child_node.set_parent(Some(node.page_id));
                    child_node.sync_header();
                    self.write_node(&child_node).await?;
                }
                left.sync_header();
                node.sync_header();
                parent.sync_header();
                self.write_node(&left).await?;
                self.write_node(&node).await?;
                self.write_node(&parent).await?;
                return Ok(());
            }
        }

        if let Some(right_page) = right_page {
            let mut right = self.read_node(right_page).await?;
            if right.keys.len() > self.min_keys() {
                if node.is_leaf() {
                    let borrowed_key = right.keys.remove(0);
                    let borrowed_vals = right.values.remove(0);
                    node.keys.push(borrowed_key.clone());
                    node.values.push(borrowed_vals);
                    parent.keys[child_index] = right.keys.first().cloned().unwrap_or(borrowed_key);
                } else {
                    let borrowed_child = right.children.remove(0);
                    let borrowed_key = right.keys.remove(0);
                    node.keys.push(parent.keys[child_index].clone());
                    node.children.push(borrowed_child);
                    parent.keys[child_index] = borrowed_key.clone();
                    if let Some(last_child) = node.children.last().copied() {
                        let mut child_node = self.read_node(last_child).await?;
                        child_node.set_parent(Some(node.page_id));
                        child_node.sync_header();
                        self.write_node(&child_node).await?;
                    }
                }
                right.sync_header();
                node.sync_header();
                parent.sync_header();
                self.write_node(&right).await?;
                self.write_node(&node).await?;
                self.write_node(&parent).await?;
                return Ok(());
            }
        }

        if let Some(left_page) = left_page {
            let mut left = self.read_node(left_page).await?;
            if node.is_leaf() {
                let next_leaf = node.next_leaf();
                left.keys.extend(node.keys);
                left.values.extend(node.values);
                left.set_next_leaf(next_leaf);
                if let Some(next_page) = next_leaf {
                    let mut next_node = self.read_node(next_page).await?;
                    next_node.set_prev_leaf(Some(left.page_id));
                    next_node.sync_header();
                    self.write_node(&next_node).await?;
                }
            } else {
                left.keys.push(parent.keys[child_index - 1].clone());
                left.keys.extend(node.keys);
                left.children.extend(node.children.clone());
                for child in &node.children {
                    let mut child_node = self.read_node(*child).await?;
                    child_node.set_parent(Some(left.page_id));
                    child_node.sync_header();
                    self.write_node(&child_node).await?;
                }
            }
            parent.keys.remove(child_index - 1);
            parent.children.remove(child_index);
            left.sync_header();
            self.write_node(&left).await?;
            node = parent;
            return self.rebalance_after_delete(node, path).await;
        }

        if let Some(right_page) = right_page {
            let mut right = self.read_node(right_page).await?;
            if node.is_leaf() {
                let next_leaf = right.next_leaf();
                node.keys.extend(right.keys);
                node.values.extend(right.values);
                node.set_next_leaf(next_leaf);
                if let Some(next_page) = next_leaf {
                    let mut next_node = self.read_node(next_page).await?;
                    next_node.set_prev_leaf(Some(node.page_id));
                    next_node.sync_header();
                    self.write_node(&next_node).await?;
                }
            } else {
                node.keys.push(parent.keys[child_index].clone());
                node.keys.extend(right.keys);
                node.children.extend(right.children.clone());
                for child in &right.children {
                    let mut child_node = self.read_node(*child).await?;
                    child_node.set_parent(Some(node.page_id));
                    child_node.sync_header();
                    self.write_node(&child_node).await?;
                }
            }
            parent.keys.remove(child_index);
            parent.children.remove(child_index + 1);
            node.sync_header();
            self.write_node(&node).await?;
            return self.rebalance_after_delete(parent, path).await;
        }

        self.write_node(&node).await
    }
}

pub struct BTreeCursor<K: BTreeKey> {
    tree: Arc<BPlusTree<K>>,
    lock_guard: OwnedRwLockReadGuard<()>,
    node: BTreeNode<K>,
    position: usize,
    upper_bound: Option<K>,
    upper_inclusive: bool,
}

impl<K: BTreeKey> BTreeCursor<K> {
    pub async fn next(&mut self) -> Result<Option<(K, Vec<RowId>)>> {
        loop {
            if self.position >= self.node.keys.len() {
                if let Some(next_leaf) = self.node.next_leaf() {
                    self.node = self.tree.read_node(next_leaf).await?;
                    self.position = 0;
                    continue;
                } else {
                    return Ok(None);
                }
            }

            let key = self.node.keys[self.position].clone();
            if let Some(bound) = &self.upper_bound {
                match key.cmp(bound) {
                    Ordering::Greater => return Ok(None),
                    Ordering::Equal if !self.upper_inclusive => return Ok(None),
                    _ => {}
                }
            }
            let values = self.node.values[self.position].clone();
            self.position += 1;
            return Ok(Some((key, values)));
        }
    }
}

#[async_trait]
pub trait IndexMaintenance: Send + Sync {
    async fn on_insert(&self, row: &VectorRow, row_id: RowId) -> Result<()>;
    async fn on_delete(&self, row: &VectorRow, row_id: RowId) -> Result<()>;
    async fn on_update(
        &self,
        old_row: &VectorRow,
        new_row: &VectorRow,
        row_id: RowId,
    ) -> Result<()>;
}

pub struct IndexMaintainer<K: BTreeKey> {
    tree: Arc<BPlusTree<K>>,
    extractor: Arc<dyn Fn(&VectorRow) -> Option<K> + Send + Sync>,
}

impl<K: BTreeKey> IndexMaintainer<K> {
    pub fn new(
        tree: Arc<BPlusTree<K>>,
        extractor: impl Fn(&VectorRow) -> Option<K> + Send + Sync + 'static,
    ) -> Self {
        Self {
            tree,
            extractor: Arc::new(extractor),
        }
    }

    pub fn tree(&self) -> &Arc<BPlusTree<K>> {
        &self.tree
    }
}

#[async_trait]
impl<K: BTreeKey> IndexMaintenance for IndexMaintainer<K> {
    async fn on_insert(&self, row: &VectorRow, row_id: RowId) -> Result<()> {
        if let Some(key) = (self.extractor)(row) {
            self.tree.insert(key, row_id).await?;
        }
        Ok(())
    }

    async fn on_delete(&self, row: &VectorRow, row_id: RowId) -> Result<()> {
        if let Some(key) = (self.extractor)(row) {
            self.tree.remove(&key, row_id).await?;
        }
        Ok(())
    }

    async fn on_update(
        &self,
        old_row: &VectorRow,
        new_row: &VectorRow,
        row_id: RowId,
    ) -> Result<()> {
        let old_key = (self.extractor)(old_row);
        let new_key = (self.extractor)(new_row);
        match (old_key, new_key) {
            (Some(old_key), Some(new_key)) => {
                if old_key != new_key {
                    self.tree.remove(&old_key, row_id).await?;
                    self.tree.insert(new_key, row_id).await?;
                }
            }
            (Some(old_key), None) => {
                self.tree.remove(&old_key, row_id).await?;
            }
            (None, Some(new_key)) => {
                self.tree.insert(new_key, row_id).await?;
            }
            (None, None) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::runtime::Runtime;

    fn create_runtime() -> Runtime {
        Runtime::new().expect("runtime")
    }

    async fn setup_tree(fanout: usize) -> (Arc<BPlusTree<i64>>, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let page_manager = Arc::new(PageManager::new(dir.path()).await.expect("page manager"));
        let mut buffer_pool = BufferPool::new(64).await.expect("buffer");
        buffer_pool.set_page_manager(page_manager.clone());
        let buffer_pool = Arc::new(buffer_pool);
        let tree = BPlusTree::create(buffer_pool.clone(), page_manager.clone(), fanout)
            .await
            .expect("tree");
        (Arc::new(tree), dir)
    }

    #[test]
    fn test_bplus_tree_insert_search_range() {
        let rt = create_runtime();
        rt.block_on(async {
            let (tree, _dir) = setup_tree(6).await;
            for i in 0..50 {
                let row_id = RowId {
                    page_id: 1,
                    slot_id: i as u16,
                };
                tree.insert(i, row_id).await.expect("insert");
            }

            for i in 0..50 {
                let rows = tree.search(&i).await.expect("search");
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].slot_id, i as u16);
            }

            let mut cursor = tree
                .clone()
                .range_cursor(Some(10), Some(20), true)
                .await
                .expect("cursor");
            let mut keys = Vec::new();
            while let Some((key, rows)) = cursor.next().await.expect("next") {
                keys.push((key, rows[0]));
            }
            assert_eq!(keys.len(), 11);
            assert_eq!(keys.first().unwrap().0, 10);
            assert_eq!(keys.last().unwrap().0, 20);
        });
    }

    #[test]
    fn test_bplus_tree_delete_and_merge() {
        let rt = create_runtime();
        rt.block_on(async {
            let (tree, _dir) = setup_tree(4).await;
            for i in 0..20 {
                let row_id = RowId {
                    page_id: 2,
                    slot_id: i as u16,
                };
                tree.insert(i, row_id).await.expect("insert");
            }

            for i in (5..15).step_by(2) {
                let row_id = RowId {
                    page_id: 2,
                    slot_id: i as u16,
                };
                assert!(tree.remove(&i, row_id).await.expect("remove"));
            }

            for i in 0..20 {
                let rows = tree.search(&i).await.expect("search");
                if (5..15).step_by(2).any(|removed| removed == i) {
                    assert!(rows.is_empty());
                } else {
                    assert_eq!(rows.len(), 1);
                }
            }
        });
    }
}
