use std::num::NonZeroU32;

use crate::Phase;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenRange {
    pub start: u32,
    pub count: NonZeroU32,
}

impl TokenRange {
    pub fn end(self) -> Option<u32> {
        self.start.checked_add(self.count.get())
    }
}

impl SequenceStep {
    pub const fn token_start(self) -> u32 {
        self.tokens.start
    }

    pub const fn token_count(self) -> NonZeroU32 {
        self.tokens.count
    }

    pub fn token_end(self) -> Option<u32> {
        self.tokens.end()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceStep {
    pub phase: Phase,
    pub tokens: TokenRange,
    pub sample: bool,
}
