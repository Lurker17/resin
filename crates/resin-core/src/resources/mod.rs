use crate::RequestId;

pub mod kv;
pub mod paged;
pub mod paged_kv;
mod prefix;

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
