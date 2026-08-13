use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum QueueMode {
    #[default]
    RoundRobin,
    Fifo,
}

impl QueueMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::Fifo => "fifo",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerPolicy {
    pub(crate) queue_mode: QueueMode,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self {
            queue_mode: QueueMode::RoundRobin,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Prefill,
    Decode,
}

/// Native metadata required to order and co-batch a ready item.
///
/// Turn ordinals never cross the Python boundary. A `u128` gives the scheduler
/// a practically inexhaustible monotonically increasing age domain: even one
/// billion new turns per second would take vastly longer than the lifetime of
/// any process to exhaust it.
pub(crate) trait Schedulable {
    fn priority(&self) -> i32;
    fn batch_key(&self) -> Option<u64>;
    fn phase(&self) -> Phase;
    fn turn_ordinal(&self) -> u128;
    fn set_turn_ordinal(&mut self, turn: u128);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectedIdentity {
    index: usize,
    turn: u128,
}

/// The immutable identity of a ready-cohort selection.
///
/// Planning inspects selected requests before the engine pins and removes
/// them. The epoch and identities make the later removal fail loudly if an
/// accidental queue mutation or reordering is inserted between those steps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Selection {
    epoch: u64,
    selected: Vec<SelectedIdentity>,
}

impl Selection {
    /// Retain the oldest prefix of a previously selected homogeneous cohort.
    ///
    /// Token-budget planning happens after the engine has inspected the ready
    /// requests. It may admit fewer rows than the sequence limit, but it must
    /// never skip an older matching row to backfill a later one.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.selected.truncate(len);
    }
}

/// Strict-priority homogeneous scheduling with stable native turn ordinals.
///
/// The oldest request at the highest ready priority anchors a batch. Matching
/// requests share its priority, batch key, and phase. FIFO survivors retain
/// their original age; round-robin survivors receive fresh turns after every
/// request already waiting.
#[derive(Debug)]
pub(crate) struct Scheduler<T> {
    ready: Vec<T>,
    max_running_requests: NonZeroUsize,
    policy: SchedulerPolicy,
    next_turn: u128,
    epoch: u64,
}

impl<T: Schedulable> Scheduler<T> {
    pub(crate) fn new(max_running_requests: NonZeroUsize, policy: SchedulerPolicy) -> Self {
        Self {
            ready: Vec::new(),
            max_running_requests,
            policy,
            next_turn: 1,
            epoch: 0,
        }
    }

    pub(crate) fn max_running_requests(&self) -> usize {
        self.max_running_requests.get()
    }

    pub(crate) fn policy(&self) -> SchedulerPolicy {
        self.policy
    }

    pub(crate) fn len(&self) -> usize {
        self.ready.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    pub(crate) fn enqueue(&mut self, mut item: T) {
        let turn = self.take_fresh_turn();
        item.set_turn_ordinal(turn);
        self.ready.push(item);
        self.bump_epoch();
    }

    pub(crate) fn selected(&self) -> (Vec<&T>, Selection) {
        let selected = self.selected_identities();
        let items = selected
            .iter()
            .map(|identity| &self.ready[identity.index])
            .collect();
        (
            items,
            Selection {
                epoch: self.epoch,
                selected,
            },
        )
    }

    pub(crate) fn take_selected(&mut self, selection: Selection) -> Vec<T> {
        assert_eq!(
            selection.epoch, self.epoch,
            "ready queue changed after its dispatch cohort was selected"
        );
        let current = self.selected_identities();
        assert!(
            selection.selected.len() <= current.len()
                && selection.selected == current[..selection.selected.len()],
            "selected dispatch cohort no longer matches the ready queue"
        );

        let mut selected = Vec::with_capacity(selection.selected.len());
        let mut remaining = Vec::with_capacity(self.ready.len() - selection.selected.len());
        let mut identities = selection.selected.into_iter().peekable();
        for (index, item) in self.ready.drain(..).enumerate() {
            if identities
                .peek()
                .is_some_and(|identity| identity.index == index)
            {
                let identity = identities.next().expect("peeked selected identity");
                assert_eq!(item.turn_ordinal(), identity.turn);
                selected.push(item);
            } else {
                remaining.push(item);
            }
        }
        assert!(
            identities.next().is_none(),
            "selected index is out of range"
        );
        self.ready = remaining;
        self.bump_epoch();
        selected
    }

    pub(crate) fn restore_rejected(&mut self, selected: Vec<T>) {
        if selected.is_empty() {
            return;
        }
        self.ready.extend(selected);
        self.sort_by_turn();
        self.bump_epoch();
    }

    pub(crate) fn requeue_survivors(&mut self, mut survivors: Vec<T>) {
        if survivors.is_empty() {
            return;
        }
        if self.policy.queue_mode == QueueMode::RoundRobin {
            for survivor in &mut survivors {
                let turn = self.take_fresh_turn();
                survivor.set_turn_ordinal(turn);
            }
        }
        self.ready.extend(survivors);
        self.sort_by_turn();
        self.bump_epoch();
    }

    pub(crate) fn position(&self, mut predicate: impl FnMut(&T) -> bool) -> Option<usize> {
        self.ready.iter().position(&mut predicate)
    }

    pub(crate) fn remove(&mut self, index: usize) -> T {
        let item = self.ready.remove(index);
        self.bump_epoch();
        item
    }

    pub(crate) fn restore(&mut self, index: usize, item: T) {
        self.ready.insert(index, item);
        self.bump_epoch();
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.ready.iter()
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.bump_epoch();
        self.ready.drain(..)
    }

    fn selected_identities(&self) -> Vec<SelectedIdentity> {
        let Some(max_priority) = self.ready.iter().map(Schedulable::priority).max() else {
            return Vec::new();
        };
        let anchor = self
            .ready
            .iter()
            .find(|item| item.priority() == max_priority)
            .expect("maximum ready priority has an anchor");
        let anchor_key = anchor.batch_key();
        let anchor_phase = anchor.phase();
        self.ready
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                item.priority() == max_priority
                    && item.batch_key() == anchor_key
                    && item.phase() == anchor_phase
            })
            .take(self.max_running_requests.get())
            .map(|(index, item)| SelectedIdentity {
                index,
                turn: item.turn_ordinal(),
            })
            .collect()
    }

    fn take_fresh_turn(&mut self) -> u128 {
        let turn = self.next_turn;
        self.next_turn = self
            .next_turn
            .checked_add(1)
            .expect("practically inexhaustible u128 scheduler turn domain exhausted");
        turn
    }

    fn sort_by_turn(&mut self) {
        self.ready.sort_by_key(Schedulable::turn_ordinal);
    }

    fn bump_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{Phase, QueueMode, Schedulable, Scheduler, SchedulerPolicy};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Item {
        id: u32,
        priority: i32,
        key: Option<u64>,
        phase: Phase,
        turn: u128,
    }

    impl Schedulable for Item {
        fn priority(&self) -> i32 {
            self.priority
        }

        fn batch_key(&self) -> Option<u64> {
            self.key
        }

        fn phase(&self) -> Phase {
            self.phase
        }

        fn turn_ordinal(&self) -> u128 {
            self.turn
        }

        fn set_turn_ordinal(&mut self, turn: u128) {
            self.turn = turn;
        }
    }

    fn item(id: u32, priority: i32, key: Option<u64>, phase: Phase) -> Item {
        Item {
            id,
            priority,
            key,
            phase,
            turn: 0,
        }
    }

    fn scheduler(limit: usize, policy: SchedulerPolicy, items: Vec<Item>) -> Scheduler<Item> {
        let mut scheduler = Scheduler::new(NonZeroUsize::new(limit).unwrap(), policy);
        for item in items {
            scheduler.enqueue(item);
        }
        scheduler
    }

    fn default_scheduler(limit: usize, count: u32) -> Scheduler<Item> {
        scheduler(
            limit,
            SchedulerPolicy::default(),
            (0..count)
                .map(|id| item(id, 0, None, Phase::Prefill))
                .collect(),
        )
    }

    fn ready_ids(scheduler: &Scheduler<Item>) -> Vec<u32> {
        scheduler.iter().map(|item| item.id).collect()
    }

    fn ready_id_turns(scheduler: &Scheduler<Item>) -> Vec<(u32, u128)> {
        scheduler.iter().map(|item| (item.id, item.turn)).collect()
    }

    fn selected_ids(scheduler: &Scheduler<Item>) -> Vec<u32> {
        scheduler
            .selected()
            .0
            .into_iter()
            .map(|item| item.id)
            .collect()
    }

    #[test]
    fn default_policy_preserves_every_legacy_prefix_and_completion_mask() {
        for queue_len in 0_u32..=8 {
            for limit in 1_usize..=8 {
                let scheduler = default_scheduler(limit, queue_len);
                assert_eq!(
                    selected_ids(&scheduler),
                    (0..queue_len.min(limit as u32)).collect::<Vec<_>>()
                );
                if queue_len == 0 {
                    continue;
                }
                let selected_len = (queue_len as usize).min(limit);
                for retired_mask in 0_u32..(1_u32 << selected_len) {
                    let mut scheduler = default_scheduler(limit, queue_len);
                    let (_, selection) = scheduler.selected();
                    let selected = scheduler.take_selected(selection);
                    let survivors = selected
                        .into_iter()
                        .enumerate()
                        .filter_map(|(row, request)| {
                            ((retired_mask & (1 << row)) == 0).then_some(request)
                        })
                        .collect::<Vec<_>>();
                    let survivor_ids = survivors.iter().map(|item| item.id).collect::<Vec<_>>();
                    scheduler.requeue_survivors(survivors);

                    let mut expected = (selected_len as u32..queue_len).collect::<Vec<_>>();
                    expected.extend(survivor_ids);
                    assert_eq!(
                        ready_ids(&scheduler),
                        expected,
                        "len={queue_len}, limit={limit}, retired={retired_mask:#b}"
                    );
                }
            }
        }
    }

    #[test]
    fn exhaustive_small_queues_match_anchor_and_homogeneity_reference() {
        let classes = [
            (-1, None, Phase::Prefill),
            (-1, Some(7), Phase::Decode),
            (0, None, Phase::Decode),
            (0, Some(7), Phase::Prefill),
            (1, None, Phase::Prefill),
            (1, None, Phase::Decode),
            (1, Some(7), Phase::Prefill),
            (1, Some(7), Phase::Decode),
        ];
        for queue_len in 1_u32..=4 {
            let case_count = classes.len().pow(queue_len);
            for mut case in 0..case_count {
                let mut items = Vec::new();
                for id in 0..queue_len {
                    let (priority, key, phase) = classes[case % classes.len()];
                    case /= classes.len();
                    items.push(item(id, priority, key, phase));
                }
                for limit in 1..=queue_len as usize {
                    let scheduler = scheduler(limit, SchedulerPolicy::default(), items.clone());
                    let max_priority = items.iter().map(|item| item.priority).max().unwrap();
                    let anchor = items
                        .iter()
                        .find(|item| item.priority == max_priority)
                        .unwrap();
                    let expected = items
                        .iter()
                        .filter(|candidate| {
                            candidate.priority == max_priority
                                && candidate.key == anchor.key
                                && candidate.phase == anchor.phase
                        })
                        .take(limit)
                        .map(|item| item.id)
                        .collect::<Vec<_>>();
                    assert_eq!(selected_ids(&scheduler), expected);
                }
            }
        }
    }

    #[test]
    fn priority_is_strict_and_never_backfills_an_incompatible_cohort() {
        let scheduler = scheduler(
            4,
            SchedulerPolicy::default(),
            vec![
                item(0, 0, None, Phase::Prefill),
                item(1, 2, Some(9), Phase::Prefill),
                item(2, 1, Some(9), Phase::Prefill),
                item(3, 2, None, Phase::Prefill),
            ],
        );
        assert_eq!(selected_ids(&scheduler), vec![1]);
    }

    #[test]
    fn none_is_a_real_key_and_phase_is_always_homogeneous() {
        let items = vec![
            item(0, 0, None, Phase::Prefill),
            item(1, 0, Some(0), Phase::Prefill),
            item(2, 0, None, Phase::Decode),
        ];
        let phase_homogeneous = scheduler(3, SchedulerPolicy::default(), items);
        assert_eq!(selected_ids(&phase_homogeneous), vec![0]);
    }

    #[test]
    fn fifo_survivors_retain_age_ahead_of_tail_and_inflight_submissions() {
        let mut scheduler = scheduler(
            2,
            SchedulerPolicy {
                queue_mode: QueueMode::Fifo,
            },
            (0..3).map(|id| item(id, 0, None, Phase::Prefill)).collect(),
        );
        let original_turns = ready_id_turns(&scheduler);
        let (_, selection) = scheduler.selected();
        let selected = scheduler.take_selected(selection);
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        scheduler.requeue_survivors(vec![selected[0].clone(), selected[1].clone()]);
        assert_eq!(ready_ids(&scheduler), vec![0, 1, 2, 3]);
        assert_eq!(&ready_id_turns(&scheduler)[..2], &original_turns[..2]);
    }

    #[test]
    fn round_robin_survivors_get_fresh_age_after_waiters_and_submissions() {
        let mut scheduler = default_scheduler(2, 3);
        let (_, selection) = scheduler.selected();
        let selected = scheduler.take_selected(selection);
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        scheduler.requeue_survivors(selected);
        assert_eq!(ready_ids(&scheduler), vec![2, 3, 0, 1]);
        assert!(
            ready_id_turns(&scheduler)
                .windows(2)
                .all(|pair| pair[0].1 < pair[1].1)
        );
    }

    #[test]
    fn rejection_restores_exact_noncontiguous_items_and_ordinals() {
        let mut scheduler = scheduler(
            3,
            SchedulerPolicy::default(),
            vec![
                item(0, 1, None, Phase::Prefill),
                item(1, 0, None, Phase::Prefill),
                item(2, 1, None, Phase::Prefill),
                item(3, 1, Some(4), Phase::Prefill),
                item(4, 1, None, Phase::Prefill),
            ],
        );
        let original = ready_id_turns(&scheduler);
        assert_eq!(selected_ids(&scheduler), vec![0, 2, 4]);
        let (_, selection) = scheduler.selected();
        let rejected = scheduler.take_selected(selection);
        scheduler.restore_rejected(rejected);
        assert_eq!(ready_id_turns(&scheduler), original);
    }

    #[test]
    fn truncated_noncontiguous_prefix_takes_and_restores_exact_identities() {
        let mut scheduler = scheduler(
            4,
            SchedulerPolicy::default(),
            vec![
                item(0, 1, Some(7), Phase::Prefill),
                item(1, 1, None, Phase::Prefill),
                item(2, 1, Some(7), Phase::Prefill),
                item(3, 0, Some(7), Phase::Prefill),
                item(4, 1, Some(7), Phase::Prefill),
            ],
        );
        let original = ready_id_turns(&scheduler);
        let (selected, mut selection) = scheduler.selected();
        assert_eq!(
            selected.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 2, 4]
        );

        selection.truncate(2);
        let taken = scheduler.take_selected(selection);
        assert_eq!(
            taken.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(ready_ids(&scheduler), vec![1, 3, 4]);

        scheduler.restore_rejected(taken);
        assert_eq!(ready_id_turns(&scheduler), original);
    }

    #[test]
    fn cancellation_rollback_restores_exact_index_and_ordinal() {
        for index in 0..5 {
            let mut scheduler = default_scheduler(3, 5);
            let original = ready_id_turns(&scheduler);
            let request = scheduler.remove(index);
            scheduler.restore(index, request);
            assert_eq!(ready_id_turns(&scheduler), original);
        }
    }

    #[test]
    #[should_panic(expected = "ready queue changed")]
    fn selection_cannot_be_taken_after_an_intervening_mutation() {
        let mut scheduler = default_scheduler(2, 3);
        let (_, selection) = scheduler.selected();
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        let _ = scheduler.take_selected(selection);
    }

    #[test]
    fn drain_preserves_global_turn_order_and_empties_scheduler() {
        let mut scheduler = default_scheduler(2, 5);
        assert_eq!(
            scheduler.drain().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert!(scheduler.is_empty());
    }
}
