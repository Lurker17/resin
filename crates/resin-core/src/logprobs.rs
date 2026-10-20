//! Borrowed columns from one synchronous forward pass.

use crate::engine::{Error, Result};

#[derive(Clone, Debug)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

#[derive(Clone, Copy)]
pub struct TopLogprobView<'a> {
    pub indptr: &'a [u32],
    pub token_ids: &'a [u32],
    pub values: &'a [f32],
}

/// CSR score rows may be empty to opt out, otherwise they cover the raw token row.
/// Storage belongs to the runner and remains borrowed until events are constructed.
#[derive(Clone, Copy)]
pub struct LogprobView<'a> {
    pub indptr: &'a [u32],
    pub values: &'a [f32],
    pub top: Option<TopLogprobView<'a>>,
}

pub fn validate_indptr(indptr: &[u32], rows: usize, values: usize) -> Result<()> {
    if indptr.len() != rows + 1
        || indptr.first() != Some(&0)
        || indptr.last().map(|&end| end as usize) != Some(values)
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(Error::execution("invalid output CSR offsets"));
    }
    Ok(())
}

impl LogprobView<'_> {
    pub fn validate(&self, token_rows: &[Vec<u32>]) -> Result<()> {
        validate_indptr(self.indptr, token_rows.len(), self.values.len())?;
        for (row, pair) in token_rows.iter().zip(self.indptr.windows(2)) {
            let count = (pair[1] - pair[0]) as usize;
            if count != 0 && count != row.len() {
                return Err(Error::execution(
                    "logprobs must cover the complete raw token row",
                ));
            }
        }
        if let Some(top) = self.top {
            validate_indptr(top.indptr, self.values.len(), top.values.len())?;
            if top.token_ids.len() != top.values.len() {
                return Err(Error::execution("top logprob IDs and scores must align"));
            }
        }
        let valid = |value: &f32| !value.is_nan() && *value <= 0.0;
        if !self.values.iter().all(valid)
            || self.top.is_some_and(|top| !top.values.iter().all(valid))
        {
            return Err(Error::execution(
                "logprobs must be non-positive and not NaN",
            ));
        }
        Ok(())
    }

    /// Called after validation; only committed events acquire owned score storage.
    pub fn token(&self, row: usize, column: usize) -> Option<TokenLogprobs> {
        let start = self.indptr[row] as usize;
        let end = self.indptr[row + 1] as usize;
        if start == end {
            return None;
        }
        let index = start + column;
        let top_logprobs = self.top.map_or_else(Vec::new, |top| {
            let range = top.indptr[index] as usize..top.indptr[index + 1] as usize;
            top.token_ids[range.clone()]
                .iter()
                .copied()
                .zip(top.values[range].iter().copied())
                .collect()
        });
        Some(TokenLogprobs {
            logprob: self.values[index],
            top_logprobs,
        })
    }
}
