//! Conversion only: request planning and validation live in resin-core.
use crate::{BatchPhase, BatchPlan, RequestPayload, cache::CacheGroupPlan};
use resin_core::{Phase, engine::ExecutionPlan};
use std::sync::OnceLock;

pub(super) fn from_core(plan: ExecutionPlan<RequestPayload>) -> BatchPlan {
    let mut sampling_params = Vec::with_capacity(plan.payloads.len());
    let mut media = Vec::with_capacity(plan.payloads.len());
    for payload in plan.payloads {
        sampling_params.push(payload.sampling_params);
        media.push(payload.media);
    }
    // The core constructs this plan from validated request and allocator state.
    // Public Python constructors still validate untrusted caller-provided arrays.
    BatchPlan {
        prompt_score_indices: plan.prompt_score_indices,
        prompt_score_token_ids: plan.prompt_score_token_ids,
        prompt_score_indptr: plan.prompt_score_indptr,
        sequence_ids: plan.sequence_ids,
        context_lengths: plan.context_lengths,
        token_ids: plan.token_ids,
        token_indptr: plan.token_indptr,
        phase: match plan.phase {
            Phase::Prefill => BatchPhase::Prefill,
            Phase::Decode => BatchPhase::Decode,
        },
        batch_id: plan.batch_id,
        batch_key: plan.batch_key,
        cache_groups: plan
            .cache_groups
            .into_iter()
            .map(CacheGroupPlan::from_inner)
            .collect(),
        sampling_mask: plan.sampling_mask,
        sampling_params,
        media,
        positions: OnceLock::new(),
    }
}
