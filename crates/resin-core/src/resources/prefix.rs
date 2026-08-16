use std::collections::HashMap;

use super::paged::CachedPages;
use super::slots::SlotHandle;

#[derive(Debug)]
pub(crate) struct PrefixSnapshot {
    pub(crate) pages: CachedPages,
    pub(crate) recurrent_checkpoint: Option<SlotHandle>,
}

#[derive(Debug)]
struct Entry<T> {
    snapshot: T,
    last_used: u64,
}

#[derive(Debug)]
struct Node<T> {
    parent: Option<usize>,
    children: HashMap<Box<[u32]>, usize>,
    entry: Option<Entry<T>>,
}

impl<T> Default for Node<T> {
    fn default() -> Self {
        Self {
            parent: None,
            children: HashMap::new(),
            entry: None,
        }
    }
}

/// Exact-token block trie with least-recently-used immutable KV snapshots.
/// Each distinct token block is stored once per parent, avoiding the quadratic
/// metadata of retaining a complete token vector at every checkpoint.
#[derive(Debug)]
pub(crate) struct PrefixCache<T> {
    alignment: u32,
    clock: u64,
    nodes: Vec<Node<T>>,
    free_nodes: Vec<usize>,
    entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<T> PrefixCache<T> {
    pub(crate) fn new(alignment: u32) -> Self {
        debug_assert_ne!(alignment, 0);
        Self {
            alignment,
            clock: 0,
            nodes: vec![Node::default()],
            free_nodes: Vec::new(),
            entries: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub(crate) const fn alignment(&self) -> u32 {
        self.alignment
    }

    pub(crate) fn find(&mut self, tokens: &[u32]) -> Option<(u32, &T)> {
        let alignment = self.alignment as usize;
        let usable = if tokens.len() % alignment == 0 {
            tokens.len().saturating_sub(alignment)
        } else {
            tokens.len() / alignment * alignment
        };
        let mut node = 0usize;
        let mut matched = None;
        for (block_index, block) in tokens[..usable].chunks_exact(alignment).enumerate() {
            let Some(&child) = self.nodes[node].children.get(block) else {
                break;
            };
            node = child;
            if self.nodes[node].entry.is_some() {
                matched = Some((node, (block_index + 1) * alignment));
            }
        }
        let Some((node, length)) = matched else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.clock = self.clock.saturating_add(1);
        let entry = self.nodes[node]
            .entry
            .as_mut()
            .expect("matched prefix node keeps its snapshot");
        entry.last_used = self.clock;
        self.hits = self.hits.saturating_add(1);
        Some((length as u32, &entry.snapshot))
    }

    pub(crate) fn insert(&mut self, tokens: &[u32], snapshot: T) -> Option<T> {
        debug_assert!(!tokens.is_empty() && tokens.len() % self.alignment as usize == 0);
        let mut node = 0usize;
        for block in tokens.chunks_exact(self.alignment as usize) {
            node = match self.nodes[node].children.get(block).copied() {
                Some(child) => child,
                None => {
                    let child = self.allocate_node(node);
                    self.nodes[node].children.insert(block.into(), child);
                    child
                }
            };
        }
        self.clock = self.clock.saturating_add(1);
        let previous = self.nodes[node].entry.replace(Entry {
            snapshot,
            last_used: self.clock,
        });
        if previous.is_none() {
            self.entries += 1;
        }
        previous.map(|entry| entry.snapshot)
    }

    pub(crate) fn evict_lru(&mut self) -> Option<T> {
        let node = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| node.entry.as_ref().map(|entry| (index, entry.last_used)))
            .min_by_key(|(_, last_used)| *last_used)?
            .0;
        let snapshot = self.nodes[node]
            .entry
            .take()
            .expect("selected LRU node keeps its snapshot")
            .snapshot;
        self.entries -= 1;
        self.evictions = self.evictions.saturating_add(1);
        self.prune(node);
        Some(snapshot)
    }

    fn allocate_node(&mut self, parent: usize) -> usize {
        if let Some(index) = self.free_nodes.pop() {
            self.nodes[index] = Node {
                parent: Some(parent),
                ..Node::default()
            };
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Node {
                parent: Some(parent),
                ..Node::default()
            });
            index
        }
    }

    fn prune(&mut self, mut node: usize) {
        while node != 0 && self.nodes[node].entry.is_none() && self.nodes[node].children.is_empty()
        {
            let parent = self.nodes[node]
                .parent
                .expect("non-root prefix nodes keep their parent");
            self.nodes[parent]
                .children
                .retain(|_, child| *child != node);
            self.nodes[node] = Node::default();
            self.free_nodes.push(node);
            node = parent;
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.entries
    }

    pub(crate) const fn hits(&self) -> u64 {
        self.hits
    }

    pub(crate) const fn misses(&self) -> u64 {
        self.misses
    }

    pub(crate) const fn evictions(&self) -> u64 {
        self.evictions
    }
}

/// Exact-token LRU checkpoints for runners whose complete sequence state is
/// represented by one Engine-managed recurrent slot rather than paged KV.
#[derive(Debug)]
pub struct StatePrefixResources {
    cache: PrefixCache<SlotHandle>,
}

impl StatePrefixResources {
    pub fn new(alignment: u32) -> Self {
        Self {
            cache: PrefixCache::new(alignment),
        }
    }

    pub fn restore(&mut self, tokens: &[u32]) -> Option<(u32, SlotHandle)> {
        self.cache
            .find(tokens)
            .map(|(length, checkpoint)| (length, *checkpoint))
    }

    pub fn publish(
        &mut self,
        tokens: &[u32],
        resident_tokens: u32,
        checkpoint: SlotHandle,
    ) -> (bool, Option<SlotHandle>) {
        let length = resident_tokens as usize;
        if length == 0 || length > tokens.len() || resident_tokens % self.cache.alignment() != 0 {
            return (false, None);
        }
        (true, self.cache.insert(&tokens[..length], checkpoint))
    }

    pub fn evict_lru(&mut self) -> Option<SlotHandle> {
        self.cache.evict_lru()
    }

    pub fn clear(&mut self) -> Vec<SlotHandle> {
        let mut checkpoints = Vec::with_capacity(self.cache.len());
        while let Some(checkpoint) = self.evict_lru() {
            checkpoints.push(checkpoint);
        }
        checkpoints
    }

    pub fn metrics(&self) -> (usize, u64, u64, u64) {
        (
            self.cache.len(),
            self.cache.hits(),
            self.cache.misses(),
            self.cache.evictions(),
        )
    }

    pub const fn alignment(&self) -> u32 {
        self.cache.alignment()
    }
}
