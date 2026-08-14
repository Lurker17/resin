use std::fmt;
use std::num::NonZeroU32;

use crate::{Phase, SequenceStep, TokenRange};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanningConfig {
    pub max_batch_tokens: Option<NonZeroU32>,
    pub prefill_chunk_size: NonZeroU32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestProgress {
    pub phase: Phase,
    pub prefill_length: u32,
    pub logical_context: u32,
    pub sample_at_prefill_end: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanningError {
    MixedPhases,
    InvalidPrefillProgress,
    TokenRangeOverflow,
}

impl fmt::Display for PlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MixedPhases => {
                formatter.write_str("scheduler selected requests from different phases")
            }
            Self::InvalidPrefillProgress => {
                formatter.write_str("prefill cursor must be before prompt length")
            }
            Self::TokenRangeOverflow => formatter.write_str("scheduled token range overflows u32"),
        }
    }
}

impl std::error::Error for PlanningError {}

/// Choose the token range for each request in the current dispatch.
///
pub fn schedule_ready_prefix(
    requests: &[RequestProgress],
    config: PlanningConfig,
) -> Result<Vec<SequenceStep>, PlanningError> {
    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    if requests.iter().any(|request| request.phase != first.phase) {
        return Err(PlanningError::MixedPhases);
    }

    let mut remaining_batch_tokens = config.max_batch_tokens.map(NonZeroU32::get);
    let mut steps = Vec::with_capacity(requests.len());
    for request in requests {
        if remaining_batch_tokens == Some(0) {
            break;
        }
        let desired = match request.phase {
            Phase::Prefill => request
                .prefill_length
                .checked_sub(request.logical_context)
                .filter(|remaining| *remaining != 0)
                .ok_or(PlanningError::InvalidPrefillProgress)?
                .min(config.prefill_chunk_size.get()),
            Phase::Decode => 1,
        };
        let count = remaining_batch_tokens.map_or(desired, |budget| desired.min(budget));
        let count = NonZeroU32::new(count).expect("positive request work and budget");
        let end = request
            .logical_context
            .checked_add(count.get())
            .ok_or(PlanningError::TokenRangeOverflow)?;
        steps.push(SequenceStep {
            phase: request.phase,
            tokens: TokenRange {
                start: request.logical_context,
                count,
            },
            sample: match request.phase {
                Phase::Prefill => request.sample_at_prefill_end && end == request.prefill_length,
                Phase::Decode => true,
            },
        });
        if let Some(remaining) = &mut remaining_batch_tokens {
            *remaining -= count.get();
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill(prefill_length: u32, logical_context: u32) -> RequestProgress {
        RequestProgress {
            phase: Phase::Prefill,
            prefill_length,
            logical_context,
            sample_at_prefill_end: true,
        }
    }

    #[test]
    fn token_budget_and_chunk_limit_compose() {
        let steps = schedule_ready_prefix(
            &[prefill(4, 0), prefill(4, 0)],
            PlanningConfig {
                max_batch_tokens: NonZeroU32::new(3),
                prefill_chunk_size: NonZeroU32::new(2).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].tokens.count.get(), 2);
        assert_eq!(steps[1].tokens.count.get(), 1);
        assert!(!steps[0].sample);
        assert!(!steps[1].sample);
    }

    #[test]
    fn token_budget_is_consumed_in_request_order() {
        let steps = schedule_ready_prefix(
            &[prefill(4, 0), prefill(4, 0)],
            PlanningConfig {
                max_batch_tokens: NonZeroU32::new(4),
                prefill_chunk_size: NonZeroU32::new(8).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].tokens.count.get(), 4);
        assert!(steps[0].sample);
    }

    #[test]
    fn prefills_are_not_limited_by_cache_page_boundaries() {
        let steps = schedule_ready_prefix(
            &[prefill(100, 30)],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(50).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(steps[0].tokens.count.get(), 50);
    }

    #[test]
    fn replay_prefill_does_not_sample_an_already_generated_token() {
        let steps = schedule_ready_prefix(
            &[RequestProgress {
                phase: Phase::Prefill,
                prefill_length: 6,
                logical_context: 0,
                sample_at_prefill_end: false,
            }],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(8).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(steps[0].token_end(), Some(6));
        assert!(!steps[0].sample);
    }

    #[test]
    fn phases_must_be_homogeneous() {
        let decode = RequestProgress {
            phase: Phase::Decode,
            prefill_length: 4,
            logical_context: 4,
            sample_at_prefill_end: false,
        };
        assert_eq!(
            schedule_ready_prefix(
                &[prefill(4, 0), decode],
                PlanningConfig {
                    max_batch_tokens: None,
                    prefill_chunk_size: NonZeroU32::new(4).unwrap(),
                },
            ),
            Err(PlanningError::MixedPhases)
        );
    }
}
