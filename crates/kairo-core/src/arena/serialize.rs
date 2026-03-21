//! Binary serialization for the arena graph (§4.3, §15).
//!
//! Field-by-field serialization (not raw memcpy, because SmallVec has variable
//! layout). Still fast: ~100KB for 1,000 nodes, <1ms to write/read.
//!
//! Uses bincode for the actual encoding, with XXH3 integrity hash.

use crate::arena::Arena;
use crate::arena::node::Node;
use crate::itch::ItchRegister;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

// ---------------------------------------------------------------------------
// Serialization errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SerializeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Integrity check failed: expected {expected:#018x}, got {actual:#018x}")]
    IntegrityCheckFailed { expected: u64, actual: u64 },

    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u32),
}

// ---------------------------------------------------------------------------
// Serialization format
// ---------------------------------------------------------------------------

/// Format version for forward compatibility.
const FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify KAIRO arena files.
const MAGIC: [u8; 4] = *b"KXAG"; // KAIRO-X Arena Graph

/// Serializable snapshot of the full arena state.
#[derive(Serialize, Deserialize)]
struct ArenaSnapshot {
    version: u32,
    nodes: Vec<Node>,
    free_list: Vec<u32>,
    title_index: Vec<(u64, u32)>,
    path_index: Vec<(u64, Vec<u32>)>,
    itch_raw: Vec<u64>,
    itch_bit_len: usize,
    specs: Vec<(u32, String)>,
    file_paths: Vec<(u64, String)>,
    current_step: u32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl Arena {
    /// Serialize the arena to a writer (file, buffer, etc.).
    ///
    /// Format: [MAGIC:4][VERSION:4][DATA_LEN:8][DATA:...][XXH3:8]
    pub fn serialize_to<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        let snapshot = ArenaSnapshot {
            version: FORMAT_VERSION,
            nodes: self.nodes.clone(),
            free_list: self.free_list.clone(),
            title_index: self.title_index.iter().map(|(&k, &v)| (k, v)).collect(),
            path_index: self
                .path_index
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            itch_raw: self.itch.as_raw_slice().to_vec(),
            itch_bit_len: self.itch.len(),
            specs: self.specs.iter().map(|(&k, v)| (k, v.clone())).collect(),
            file_paths: self
                .file_paths
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            current_step: self.current_step,
        };

        // Encode to bincode
        let data = bincode::serialize(&snapshot)?;

        // Compute integrity hash
        let hash = xxh3_64(&data);

        // Write: magic + version + data_len + data + hash
        writer.write_all(&MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&(data.len() as u64).to_le_bytes())?;
        writer.write_all(&data)?;
        writer.write_all(&hash.to_le_bytes())?;

        Ok(())
    }

    /// Serialize to a byte vector.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        let mut buf = Vec::new();
        self.serialize_to(&mut buf)?;
        Ok(buf)
    }

    /// Deserialize an arena from a reader.
    pub fn deserialize_from<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        // Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(SerializeError::UnsupportedVersion(0));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FORMAT_VERSION {
            return Err(SerializeError::UnsupportedVersion(version));
        }

        // Read data length
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let data_len = u64::from_le_bytes(len_bytes) as usize;

        // Read data
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        // Read hash
        let mut hash_bytes = [0u8; 8];
        reader.read_exact(&mut hash_bytes)?;
        let expected_hash = u64::from_le_bytes(hash_bytes);

        // Verify integrity
        let actual_hash = xxh3_64(&data);
        if actual_hash != expected_hash {
            return Err(SerializeError::IntegrityCheckFailed {
                expected: expected_hash,
                actual: actual_hash,
            });
        }

        // Decode
        let snapshot: ArenaSnapshot = bincode::deserialize(&data)?;

        // Reconstruct arena
        let mut arena = Arena::new();
        arena.nodes = snapshot.nodes;
        arena.free_list = snapshot.free_list;
        arena.title_index = snapshot.title_index.into_iter().collect();
        arena.path_index = snapshot.path_index.into_iter().collect();
        arena.itch = ItchRegister::from_raw(snapshot.itch_raw, snapshot.itch_bit_len);
        arena.specs = snapshot.specs.into_iter().collect();
        arena.file_paths = snapshot.file_paths.into_iter().collect();
        arena.current_step = snapshot.current_step;

        // Rebuild the pending queue from node states
        // (pending queue is ephemeral; rebuilt from node statuses)
        for (idx, node) in arena.nodes.iter().enumerate().skip(1) {
            if (node.status == crate::arena::node::NodeStatus::Pending
                || node.status == crate::arena::node::NodeStatus::Ready)
                && arena.are_dependencies_resolved(idx as u32)
            {
                    arena
                        .pending_queue
                        .push(crate::arena::priority_queue::PendingEntry {
                            node_idx: idx as u32,
                            priority: node.priority,
                        });
            }
        }

        Ok(arena)
    }

    /// Deserialize from a byte slice.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, SerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::deserialize_from(&mut cursor)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::node::{Node, Priority};

    #[test]
    fn test_round_trip() {
        let mut arena = Arena::new();

        let n1 = arena.alloc(Node::new("JWT Service".to_string(), Priority::Critical));
        let n2 = arena.alloc(Node::new("Auth Middleware".to_string(), Priority::Standard));
        arena.add_dependency(n2, n1);
        arena.set_spec(n1, "Implement JWT token service with RS256".to_string());

        let file_hash = arena.register_file_path("src/auth/jwt.rs");
        arena.get_mut(n1).impl_files.push(file_hash);

        // Serialize
        let bytes = arena.serialize_to_bytes().unwrap();

        // Deserialize
        let restored = Arena::deserialize_from_bytes(&bytes).unwrap();

        // Verify
        assert_eq!(restored.live_count(), 2);
        assert_eq!(restored.get(n1).title.as_str(), "JWT Service");
        assert_eq!(restored.get(n2).title.as_str(), "Auth Middleware");
        assert!(restored.get(n2).dependencies.contains(&n1));
        assert_eq!(
            restored.get_spec(n1).unwrap(),
            "Implement JWT token service with RS256"
        );
        assert_eq!(restored.resolve_file_path(file_hash).unwrap(), "src/auth/jwt.rs");
    }

    #[test]
    fn test_integrity_check() {
        let mut arena = Arena::new();
        arena.alloc(Node::new("Test".to_string(), Priority::Standard));

        let mut bytes = arena.serialize_to_bytes().unwrap();

        // Corrupt one byte in the data section
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;

        let result = Arena::deserialize_from_bytes(&bytes);
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializeError::IntegrityCheckFailed { .. } => {}
            other => panic!("Expected IntegrityCheckFailed, got: {other}"),
        }
    }

    #[test]
    fn test_serialization_size() {
        let mut arena = Arena::new();

        // Create 100 nodes
        for i in 0..100 {
            let n = arena.alloc(Node::new(format!("Node {i}"), Priority::Standard));
            arena.set_spec(n, format!("Spec for node {i}"));
        }

        let bytes = arena.serialize_to_bytes().unwrap();

        // Should be reasonably compact — well under 100KB for 100 nodes
        assert!(
            bytes.len() < 50_000,
            "Serialized size too large: {} bytes for 100 nodes",
            bytes.len()
        );
    }
}
