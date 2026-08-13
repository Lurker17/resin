use std::ops::{Deref, DerefMut};

use crate::RequestId;
use crate::Schedulable;

pub mod kv;
pub mod paged;
pub mod paged_kv;
pub mod slots;

/// An owner and the resource state whose lifetime exactly matches it.
#[derive(Debug)]
pub struct WithResources<T, R> {
    pub inner: T,
    pub resources: R,
}

impl<T, R> Deref for WithResources<T, R> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, R> DerefMut for WithResources<T, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: Schedulable, R> Schedulable for WithResources<T, R> {
    fn priority(&self) -> i32 {
        self.inner.priority()
    }

    fn batch_key(&self) -> Option<u64> {
        self.inner.batch_key()
    }

    fn phase(&self) -> crate::Phase {
        self.inner.phase()
    }

    fn turn_ordinal(&self) -> u128 {
        self.inner.turn_ordinal()
    }

    fn set_turn_ordinal(&mut self, turn: u128) {
        self.inner.set_turn_ordinal(turn);
    }
}

/// Attention and recurrent state at one ownership boundary.
#[derive(Debug)]
pub struct ResourceSet<A, R> {
    pub attention: A,
    pub recurrent: R,
}

#[derive(Clone, Copy, Debug)]
pub struct RequestResourceRef<'a, S> {
    pub request_id: RequestId,
    pub state: &'a S,
}

/// The executor view and the capability that keeps it valid.
#[derive(Debug)]
#[must_use = "batch access must be released or quarantined"]
pub struct BatchAccess<V, L> {
    pub view: V,
    pub lease: L,
}
