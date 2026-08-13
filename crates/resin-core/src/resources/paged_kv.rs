use std::fmt;

use crate::{RequestId, SequenceStep};

use super::kv::{KvGroupLayout, KvGroupPlan, KvLayout, KvPlanError, KvRetention};
use super::paged::{
    CachePool, CachePoolError, GroupDemand, GroupId, RequestPages, RetireRequestFailure,
    SequenceId, WriterAccessClaim, WriterAccessLease,
};
use super::{BatchAccess, RequestResourceRef};

#[derive(Debug)]
pub struct PagedKvResources {
    layout: KvLayout,
    pool: CachePool,
    batch_active: bool,
}

impl PagedKvResources {
    pub fn new(layout: KvLayout) -> Result<Self, CachePoolError> {
        let pool = CachePool::new(
            layout.parent_capacity().get(),
            &layout.packings().collect::<Vec<_>>(),
        )?;
        Ok(Self {
            layout,
            pool,
            batch_active: false,
        })
    }

    pub const fn pool(&self) -> &CachePool {
        &self.pool
    }

    pub fn attach(
        &mut self,
        request_id: RequestId,
        maximum_context: u32,
    ) -> Result<RequestPages, CachePoolError> {
        let demands = self
            .layout
            .groups()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, group)| {
                let blocks = match group.retention() {
                    KvRetention::FullContext => {
                        maximum_context.div_ceil(group.block_tokens().get())
                    }
                    KvRetention::SlidingWindow { .. } => group
                        .window_resident_blocks()
                        .expect("sliding window has a resident width")
                        .get(),
                };
                GroupDemand::new(GroupId::new(index as u32), blocks as usize)
            })
            .collect::<Vec<_>>();
        let pending = self
            .pool
            .reserve_request(SequenceId::new(request_id.get()), &demands)?;
        Ok(self.pool.promote_request(pending))
    }

    pub fn retire(&mut self, state: RequestPages) -> Result<(), RetireRequestFailure> {
        self.pool.retire_request(state)
    }

    pub fn release(&mut self, lease: WriterAccessLease) {
        assert!(self.batch_active, "released a nonexistent KV batch");
        self.pool.release_writer_access(lease);
        self.batch_active = false;
    }

    fn invalid(message: impl Into<String>) -> PagedKvError {
        PagedKvError::Invalid(message.into())
    }

    pub fn acquire(
        &mut self,
        requests: &[RequestResourceRef<'_, RequestPages>],
        steps: &[SequenceStep],
    ) -> Result<BatchAccess<PagedKvBatchView, WriterAccessLease>, PagedKvError> {
        if self.batch_active {
            return Err(Self::invalid(
                "cannot acquire a KV batch before the current batch is released",
            ));
        }
        if requests.is_empty() || requests.len() != steps.len() {
            return Err(Self::invalid(
                "KV requests and steps must have equal nonzero lengths",
            ));
        }
        for request in requests {
            self.pool.validate_request(request.state)?;
            if request.state.sequence().get() != request.request_id.get() {
                return Err(Self::invalid(format!(
                    "request {} cache pages belong to sequence {}",
                    request.request_id.get(),
                    request.state.sequence().get()
                )));
            }
            if request.state.groups().len() != self.layout.groups().len() {
                return Err(Self::invalid(format!(
                    "request {} has the wrong cache group count",
                    request.request_id.get()
                )));
            }
            for (index, table) in request.state.groups().iter().enumerate() {
                if table.group() != GroupId::new(index as u32) {
                    return Err(Self::invalid(format!(
                        "request {} cache groups are out of order",
                        request.request_id.get()
                    )));
                }
            }
        }

        let append_count = steps
            .iter()
            .map(|step| step.token_count().get() as usize)
            .sum();
        let mut groups = Vec::with_capacity(self.layout.groups().len());
        let mut claims = Vec::<WriterAccessClaim>::new();
        for (group_index, group) in self.layout.groups().iter().copied().enumerate() {
            let mut indptr = Vec::with_capacity(requests.len() + 1);
            let mut block_ids = Vec::new();
            let mut first_blocks = Vec::with_capacity(requests.len());
            let mut append_ids = Vec::with_capacity(append_count);
            let mut append_offsets = Vec::with_capacity(append_count);
            indptr.push(0);

            for (request, step) in requests.iter().zip(steps) {
                let table = &request.state.groups()[group_index];
                let post_length = step
                    .token_end()
                    .ok_or_else(|| Self::invalid("scheduled token range overflows u32"))?;
                let (first, count) = active_blocks(group, post_length);
                if count as usize > table.pages().len() {
                    return Err(Self::invalid(format!(
                        "request {} cache group {group_index} needs {count} blocks but reserved {}",
                        request.request_id.get(),
                        table.pages().len()
                    )));
                }
                first_blocks.push(first);
                for logical in first..first + count {
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    block_ids.push(self.pool.kernel_page_id(page.handle())?);
                    claims.push(request.state.write_claim(page));
                }
                indptr.push(
                    u32::try_from(block_ids.len()).map_err(|_| {
                        Self::invalid("cache block table exceeds the u32 CSR range")
                    })?,
                );

                for position in step.token_start()..post_length {
                    let logical = position / group.block_tokens().get();
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    append_ids.push(self.pool.kernel_page_id(page.handle())?);
                    append_offsets.push(position % group.block_tokens().get());
                }
            }
            groups.push(KvGroupPlan::new(
                group_index as u32,
                indptr,
                block_ids,
                first_blocks,
                append_ids,
                append_offsets,
            )?);
        }

        let prepared = self.pool.prepare_writer_access(claims)?;
        let lease = self.pool.acquire_writer_access(prepared)?;
        self.batch_active = true;
        Ok(BatchAccess {
            view: PagedKvBatchView {
                pool_id: self.pool.id(),
                groups,
            },
            lease,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedKvBatchView {
    pub pool_id: u64,
    pub groups: Vec<KvGroupPlan>,
}

#[derive(Debug)]
pub enum PagedKvError {
    Cache(CachePoolError),
    Invalid(String),
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PagedKvError {}

impl From<CachePoolError> for PagedKvError {
    fn from(error: CachePoolError) -> Self {
        Self::Cache(error)
    }
}

impl From<KvPlanError> for PagedKvError {
    fn from(error: KvPlanError) -> Self {
        Self::Invalid(error.to_string())
    }
}

fn active_blocks(group: KvGroupLayout, post_length: u32) -> (u32, u32) {
    let total = post_length.div_ceil(group.block_tokens().get());
    match group.retention() {
        KvRetention::FullContext => (0, total),
        KvRetention::SlidingWindow { .. } => {
            let count = total.min(group.window_resident_blocks().unwrap().get());
            (total - count, count)
        }
    }
}

fn page_for<R>(group: KvGroupLayout, pages: &[R], logical: u32) -> Option<&R> {
    let index = match group.retention() {
        KvRetention::FullContext => logical as usize,
        KvRetention::SlidingWindow { .. } => logical as usize % pages.len(),
    };
    pages.get(index)
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroU64};

    use crate::{Phase, TokenRange};

    use super::*;

    #[test]
    fn full_and_sliding_groups_share_one_current_batch_lease() {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(
                    2,
                    2,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 12).unwrap();
        let step = SequenceStep {
            phase: Phase::Prefill,
            tokens: TokenRange {
                start: 0,
                count: NonZeroU32::new(4).unwrap(),
            },
            sample: false,
        };
        let requests = [RequestResourceRef {
            request_id: id,
            state: &state,
        }];
        let access = resources.acquire(&requests, &[step]).unwrap();
        assert_eq!(access.view.groups.len(), 2);
        assert!(resources.acquire(&requests, &[step]).is_err());
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }
}
