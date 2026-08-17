//! PyO3-free scheduling and resource ownership for Resin.
//!
//! The crate separates three concerns:
//! - [`Scheduler`] chooses a phase-homogeneous, key-homogeneous cohort.
//! - [`schedule_ready_prefix`] freezes each row's current token range.
//! - [`resources`] owns allocator state and just-in-time batch leases.
//!
//! The KV coordinator is deliberately single-dispatch, so it cannot plan or
//! lock a future batch before the current lease is released.

#![forbid(unsafe_code)]

mod batch;
mod ids;
mod planning;
pub mod resources;
mod scheduler;

pub use batch::{SequenceStep, TokenRange};
pub use ids::RequestId;
pub use planning::{PlanningConfig, PlanningError, RequestProgress, schedule_ready_prefix};
pub use scheduler::{Phase, Schedulable, Scheduler, Selection};
