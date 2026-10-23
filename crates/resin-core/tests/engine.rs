use resin_core::engine::{
    EngineCore, Error, ErrorKind, ExecutionPlan, Runner, RunnerConfig, SchedulerConfig, Submission,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use std::collections::VecDeque;
use std::num::NonZeroU32;

#[derive(Default)]
struct Model {
    plans: Vec<ExecutionPlan<&'static str>>,
    outputs: VecDeque<Result<Vec<Vec<u32>>, Error>>,
    released: Vec<u64>,
    release_fails: bool,
}
impl Runner<&'static str> for Model {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<Vec<Vec<u32>>, Error> {
        let fallback = plan
            .token_indptr
            .windows(2)
            .map(|range| vec![plan.token_ids[range[1] as usize - 1] + 1])
            .collect();
        self.plans.push(plan);
        self.outputs.pop_front().unwrap_or(Ok(fallback))
    }
    fn release(&mut self, ids: &[u64]) -> Result<(), Error> {
        if self.release_fails {
            return Err(Error::execution("release failed"));
        }
        self.released.extend(ids);
        Ok(())
    }
}
fn engine(lookahead: u32, chunk: u32) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    16,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            cache_write_lookahead_tokens: lookahead,
            ..Default::default()
        },
        4,
        128,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..Default::default()
        },
        None,
        false,
    )
    .unwrap()
}
fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "opaque model metadata");
    input.max_new_tokens = Some(limit);
    input
}

#[test]
fn admission_credits_restored_physical_prefix_pages_only() {
    for cache_hit in [true, false] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        8,
                        vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                continuation_prefill: true,
                ..Default::default()
            },
            4,
            32,
            SchedulerConfig {
                max_batch_tokens: NonZeroU32::new(32),
                prefill_chunk_size: NonZeroU32::new(32).unwrap(),
                ..Default::default()
            },
            None,
            true,
        )
        .unwrap();
        let mut model = Model::default();
        let prefix: Vec<u32> = (0..8).collect();
        let seed = engine.submit(input(prefix.clone(), 1)).unwrap();
        while engine.has_work() {
            engine.step(&mut model).unwrap();
        }
        seed.take_result().unwrap();
        let requests: Vec<_> = (0..4)
            .map(|index| {
                let mut prompt = prefix.clone();
                if !cache_hit {
                    prompt[0] = 90;
                }
                prompt.extend([100 + index, 200]);
                engine.submit(input(prompt, 1)).unwrap()
            })
            .collect();
        let events = engine.step(&mut model).unwrap();
        let plan = model.plans.last().unwrap();
        assert_eq!(plan.sequence_ids.len(), if cache_hit { 4 } else { 1 });
        if cache_hit {
            assert_eq!(events.len(), 4);
            assert_eq!(plan.context_lengths, [8; 4]);
            assert_eq!(plan.token_ids.len(), 8);
            let tables: Vec<_> = plan.cache_groups[0]
                .block_slab_ids
                .as_chunks::<5>()
                .0
                .iter()
                .collect();
            assert_eq!(tables.len(), 4);
            for table in &tables[1..] {
                assert_eq!(&table[..4], &tables[0][..4]);
            }
            assert_eq!(
                tables
                    .iter()
                    .map(|table| table[4])
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                4
            );
            for request in requests {
                request.take_result().unwrap();
            }
        }
        engine.close(&mut model).unwrap();
    }
}

#[test]
fn a_native_model_uses_the_complete_lifecycle_without_python() {
    let mut engine = engine(0, 2);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1, 2, 3, 4, 5], 3)).unwrap();
    assert!(engine.step(&mut model).unwrap().is_empty());
    assert!(engine.step(&mut model).unwrap().is_empty());
    let mut events = Vec::new();
    while engine.has_work() {
        events.extend(engine.step(&mut model).unwrap());
    }
    assert_eq!(
        events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>(),
        [6, 7, 8]
    );
    assert_eq!(
        model
            .plans
            .iter()
            .map(|plan| plan.context_lengths[0])
            .collect::<Vec<_>>(),
        [0, 2, 4, 5, 6]
    );
    assert!(
        model
            .plans
            .iter()
            .all(|plan| plan.payloads == ["opaque model metadata"])
    );
    assert_eq!(request.take_result().unwrap(), [6, 7, 8]);
    assert!(request.take_result().is_err());
    assert_eq!(model.released, [request.id()]);
    assert_eq!(engine.cache_free_parent_count(), Some(16));
    engine.close(&mut model).unwrap();
    assert!(engine.closed());
}

#[test]
fn rejected_execution_restores_requests_and_releases_batch_access() {
    let mut engine = engine(0, 8);
    let mut model = Model::default();
    model
        .outputs
        .push_back(Err(Error::rejected("no state changed")));
    let first = engine.submit(input(vec![1, 2], 1)).unwrap();
    let second = engine.submit(input(vec![3, 4], 1)).unwrap();
    assert_eq!(
        engine.step(&mut model).unwrap_err().kind,
        ErrorKind::Rejected
    );
    assert!(!engine.poisoned());
    assert_eq!(engine.ready_request_count(), 2);
    engine.step(&mut model).unwrap();
    assert_eq!(model.plans[0].sequence_ids, model.plans[1].sequence_ids);
    assert!(model.plans[1].batch_id > model.plans[0].batch_id);
    assert_eq!(model.plans[0].cache_groups, model.plans[1].cache_groups);
    assert_eq!(first.take_result().unwrap(), [3]);
    assert_eq!(second.take_result().unwrap(), [5]);
}

#[test]
fn terminal_limits_do_not_hide_unleased_runner_output() {
    for eos in [None, Some(3)] {
        let mut engine = engine(0, 8);
        let mut model = Model::default();
        let mut request_input = input(vec![1], 2);
        request_input.stop_token_ids = eos.map(|token| vec![token]);
        let request = engine.submit(request_input).unwrap();
        engine.step(&mut model).unwrap();
        model.outputs.push_back(Ok(vec![vec![3, 4, 5]]));
        assert!(
            engine
                .step(&mut model)
                .unwrap_err()
                .to_string()
                .contains("leased cache space")
        );
        assert!(engine.poisoned());
        assert!(!request.done());
        assert!(model.released.is_empty());
        assert!(engine.close(&mut model).is_err());
    }
}

#[test]
fn frontend_event_failure_quarantines_before_native_commit() {
    let mut engine = engine(0, 8);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1, 2], 1)).unwrap();
    let error = engine
        .step_with::<()>(&mut model, |_| {
            Err(Error::memory("event allocation failed"))
        })
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Memory);
    assert!(engine.poisoned());
    assert!(!request.done());
    assert!(model.released.is_empty());
    assert_eq!(engine.cache_occupied_block_count(), Some(1));
}

#[test]
fn failed_release_keeps_request_ownership_quarantined() {
    let mut engine = engine(0, 8);
    let mut model = Model {
        release_fails: true,
        ..Default::default()
    };
    let request = engine.submit(input(vec![1], 1)).unwrap();
    assert!(engine.step(&mut model).is_err());
    assert!(engine.poisoned());
    assert!(!request.done());
    assert_eq!(engine.cache_occupied_block_count(), Some(1));
}

#[test]
fn handle_allocation_failure_does_not_submit_a_request() {
    let mut engine = engine(0, 8);
    assert!(
        engine
            .submit_with::<()>(input(vec![1], 1), |_| Err(Error::memory(
                "handle allocation failed"
            )))
            .is_err()
    );
    assert!(!engine.has_work());
    assert_eq!(engine.next_request_id(), 1);
    assert_eq!(engine.cache_free_parent_count(), Some(16));
}

#[test]
fn cancellation_checks_engine_identity_and_returns_native_cache() {
    let mut first = engine(0, 8);
    let mut second = engine(0, 8);
    let mut model = Model::default();
    let request = first.submit(input(vec![1], 3)).unwrap();
    assert_eq!(
        second.cancel(&mut model, &request).unwrap_err().kind,
        ErrorKind::InvalidInput
    );
    first.step(&mut model).unwrap();
    assert!(first.cancel(&mut model, &request).unwrap());
    assert!(request.cancelled());
    assert_eq!(request.take_result().unwrap(), [2]);
    assert_eq!(first.cache_free_parent_count(), Some(16));
}

#[test]
fn a_panicking_native_runner_cannot_leave_a_healthy_engine() {
    struct PanickingRunner {
        score_access: bool,
    }
    impl Runner<&'static str> for PanickingRunner {
        fn forward(&mut self, _plan: ExecutionPlan<&'static str>) -> Result<Vec<Vec<u32>>, Error> {
            assert!(self.score_access, "model state may already have changed");
            Ok(vec![vec![2]])
        }
        fn logprobs(&self) -> Option<resin_core::logprobs::LogprobView<'_>> {
            panic!("score storage access failed after model execution");
        }
    }
    for score_access in [false, true] {
        let mut engine = engine(0, 8);
        let request = engine.submit(input(vec![1], 1)).unwrap();
        let error = engine
            .step(&mut PanickingRunner { score_access })
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::Execution);
        assert!(engine.poisoned());
        assert!(!request.done());
        assert_eq!(engine.cache_occupied_block_count(), Some(1));
    }
}

#[test]
fn a_panicking_payload_clone_restores_the_cohort_and_releases_its_cache_access() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Payload {
        token: u32,
        fail_clone: Arc<AtomicBool>,
    }
    impl Clone for Payload {
        fn clone(&self) -> Self {
            assert!(
                !self.fail_clone.swap(false, Ordering::SeqCst),
                "payload clone failed"
            );
            Self {
                token: self.token,
                fail_clone: Arc::clone(&self.fail_clone),
            }
        }
    }
    #[derive(Default)]
    struct NativeModel {
        calls: usize,
        ids: Vec<u64>,
    }
    impl Runner<Payload> for NativeModel {
        fn forward(&mut self, plan: ExecutionPlan<Payload>) -> Result<Vec<Vec<u32>>, Error> {
            self.calls += 1;
            self.ids = plan.sequence_ids;
            Ok(plan
                .payloads
                .into_iter()
                .map(|payload| vec![payload.token])
                .collect())
        }
    }

    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    8,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            ..Default::default()
        },
        4,
        128,
        SchedulerConfig::default(),
        None,
        false,
    )
    .unwrap();
    let mut handles = Vec::new();
    for (token, fail_clone) in [(11, false), (21, true)] {
        let mut submission = Submission::new(
            vec![1, 2],
            Payload {
                token,
                fail_clone: Arc::new(AtomicBool::new(fail_clone)),
            },
        );
        submission.max_new_tokens = Some(1);
        handles.push(engine.submit(submission).unwrap());
    }
    let mut model = NativeModel::default();
    let error = engine.step(&mut model).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Rejected);
    assert!(error.to_string().contains("payload cloning"));
    assert!(!engine.poisoned());
    assert_eq!(engine.ready_request_count(), 2);
    assert_eq!(model.calls, 0);
    assert!(handles.iter().all(|handle| !handle.done()));

    engine.step(&mut model).unwrap();
    assert_eq!(model.calls, 1);
    assert_eq!(
        model.ids,
        handles.iter().map(|handle| handle.id()).collect::<Vec<_>>()
    );
    assert_eq!(handles[0].take_result().unwrap(), [11]);
    assert_eq!(handles[1].take_result().unwrap(), [21]);
    assert_eq!(engine.cache_free_parent_count(), Some(8));
}

#[test]
fn enqueued_transaction_retains_requests_and_pages_until_harvest() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 2)).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(model.plans.is_empty());
    assert!(engine.has_in_flight_step());
    assert!(engine.has_work());
    assert_eq!(engine.ready_request_count(), 0);
    assert_eq!(engine.cache_free_parent_count(), Some(15));
    assert_eq!(
        engine.next_phase().unwrap(),
        Some(resin_core::Phase::Prefill)
    );
    assert!(!request.done());
    assert!(engine.prepare_step(&mut model, None).is_err());
    assert!(engine.cancel(&mut model, &request).is_err());
    assert!(engine.close(&mut model).is_err());
    assert!(!engine.poisoned());

    let output = model.forward(plan);
    let events = engine.complete_step_with(&mut model, output, Ok).unwrap();
    assert_eq!(events[0].token_id, 11);
    assert!(!engine.has_in_flight_step());
    assert_eq!(engine.ready_request_count(), 1);
    assert!(engine.cancel(&mut model, &request).unwrap());
    assert_eq!(engine.cache_free_parent_count(), Some(16));
}

#[test]
fn rejected_enqueued_transaction_restores_exact_cohort_but_failure_quarantines() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 2)).unwrap();
    engine.prepare_step(&mut model, None).unwrap().unwrap();
    let rejected = engine.complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok);
    assert_eq!(rejected.unwrap_err().kind, ErrorKind::Rejected);
    assert!(!engine.poisoned());
    assert!(!engine.has_in_flight_step());
    assert_eq!(engine.ready_request_count(), 1);
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.sequence_ids, [request.id()]);
    let failed = engine.complete_step_with(&mut model, Err(Error::execution("device failed")), Ok);
    assert_eq!(failed.unwrap_err().kind, ErrorKind::Execution);
    assert!(engine.poisoned());
    assert_eq!(engine.cache_free_parent_count(), Some(15));
    assert!(model.released.is_empty());
    assert!(!request.done());
}

#[test]
fn decode_continuation_transfers_writer_lease_and_commits_exact_feedback_depth() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    4,
                    vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            cache_write_lookahead_tokens: 3,
            ..Default::default()
        },
        4,
        512,
        SchedulerConfig::default(),
        None,
        true,
    )
    .unwrap();
    let mut model = Model::default();
    let request = engine.submit(input(vec![1], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.context_lengths, [1]);
    assert!(engine.retain_continuation(&plan));
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [5]); // GPU corrects conservative host length.
    assert_eq!(next.cache_groups[0].append_block_offsets, [5]);
    let leased = engine.cache_free_parent_count();
    let events = engine
        .complete_continuing_step_with(&mut model, Ok(vec![vec![3, 4]]), Ok)
        .unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>(),
        [3, 4]
    );
    assert!(engine.has_in_flight_step());
    assert_eq!(engine.cache_free_parent_count(), leased);
    let following = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(following.context_lengths, [7]); // Actual 1 + 2, then maximum 4.
    assert_eq!(following.token_ids, [4]);
    engine
        .complete_step_with(&mut model, Ok(vec![vec![5]]), Ok)
        .unwrap();
    assert!(!engine.has_in_flight_step());
    let next = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(next.context_lengths, [4]);
    assert_eq!(next.token_ids, [5]);
    engine
        .complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok)
        .unwrap_err();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(engine.cache_free_parent_count(), Some(4));
}

#[test]
fn unsafe_decode_boundaries_fall_back_before_enqueuing_a_continuation() {
    for (prompt, stop, limit) in [
        (vec![1; 127], None, 40),
        (vec![1], Some(vec![9]), 40),
        (vec![1], None, 4),
    ] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        8,
                        vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                cache_write_lookahead_tokens: 3,
                ..Default::default()
            },
            4,
            512,
            SchedulerConfig::default(),
            None,
            false,
        )
        .unwrap();
        let mut model = Model::default();
        let mut submission = input(prompt, limit);
        submission.stop_token_ids = stop;
        engine.submit(submission).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(!engine.retain_continuation(&plan));
        assert!(engine.continuation_plan().unwrap().is_none());
        let output = model.forward(plan);
        engine.complete_step_with(&mut model, output, Ok).unwrap();
        engine.close(&mut model).unwrap();
    }
}

#[test]
fn failure_after_next_enqueue_keeps_transferred_lease_quarantined() {
    for output in [
        Err(Error::rejected("prior result unavailable")),
        Ok(vec![vec![]]),
    ] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        4,
                        vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                cache_write_lookahead_tokens: 3,
                ..Default::default()
            },
            4,
            512,
            SchedulerConfig::default(),
            None,
            false,
        )
        .unwrap();
        let mut model = Model::default();
        let request = engine.submit(input(vec![1], 40)).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(engine.retain_continuation(&plan));
        engine.continuation_plan().unwrap().unwrap(); // Next GPU work is accepted.
        engine
            .complete_continuing_step_with(&mut model, output, Ok)
            .unwrap_err();
        assert!(engine.poisoned());
        assert_eq!(engine.ready_request_count(), 0);
        assert_eq!(engine.cache_free_parent_count(), Some(3));
        assert!(model.released.is_empty());
        assert!(!request.done());
        assert!(engine.close(&mut model).is_err());
    }
}
