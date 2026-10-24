//! Flat token rows shared by native runners and borrowed foreign output.

use crate::engine::{Error, Result};
use crate::logprobs::validate_indptr;

/// Validated CSR token rows. The runner retains their storage through completion.
#[derive(Clone, Copy, Debug)]
pub struct TokenRows<'a> {
    token_ids: &'a [u32],
    indptr: &'a [u32],
}

impl<'a> TokenRows<'a> {
    pub fn new(token_ids: &'a [u32], indptr: &'a [u32]) -> Result<Self> {
        let rows = indptr
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::execution("invalid output CSR offsets"))?;
        validate_indptr(indptr, rows, token_ids.len())?;
        Ok(Self { token_ids, indptr })
    }

    pub fn len(self) -> usize {
        self.indptr.len() - 1
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn token_count(self) -> usize {
        self.token_ids.len()
    }

    pub fn row(self, index: usize) -> &'a [u32] {
        &self.token_ids[self.indptr[index] as usize..self.indptr[index + 1] as usize]
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = &'a [u32]> {
        self.indptr
            .windows(2)
            .map(move |pair| &self.token_ids[pair[0] as usize..pair[1] as usize])
    }

    pub fn to_owned(self) -> OwnedTokenRows {
        OwnedTokenRows {
            token_ids: self.token_ids.to_vec(),
            indptr: self.indptr.to_vec(),
        }
    }
}

/// Owned form for synchronous native runners; async runtimes may borrow instead.
#[derive(Clone, Debug)]
pub struct OwnedTokenRows {
    token_ids: Vec<u32>,
    indptr: Vec<u32>,
}

impl OwnedTokenRows {
    pub fn new(token_ids: Vec<u32>, indptr: Vec<u32>) -> Result<Self> {
        TokenRows::new(&token_ids, &indptr)?;
        Ok(Self { token_ids, indptr })
    }

    pub fn view(&self) -> TokenRows<'_> {
        TokenRows {
            token_ids: &self.token_ids,
            indptr: &self.indptr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_borrow_exact_flat_slices_including_empty_rows() {
        let ids = [u32::MAX, 2, 3, 4];
        let offsets = [0, 1, 1, 4];
        let rows = TokenRows::new(&ids, &offsets).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows.token_count(), 4);
        assert_eq!(
            rows.iter().collect::<Vec<_>>(),
            [&ids[..1], &ids[1..1], &ids[1..]]
        );
        assert_eq!(rows.row(2).as_ptr(), ids[1..].as_ptr());
        let owned = rows.to_owned();
        assert_eq!(
            owned.view().iter().collect::<Vec<_>>(),
            rows.iter().collect::<Vec<_>>()
        );
        assert_ne!(owned.view().row(0).as_ptr(), ids.as_ptr());
        assert!(TokenRows::new(&[], &[0]).unwrap().is_empty());
    }

    #[test]
    fn malformed_offsets_are_rejected_before_rows_can_be_sliced() {
        for offsets in [&[][..], &[1, 2], &[0, 3], &[0, 2, 1], &[0, 1]] {
            assert!(TokenRows::new(&[1, 2], offsets).is_err());
            assert!(OwnedTokenRows::new(vec![1, 2], offsets.to_vec()).is_err());
        }
    }
}
