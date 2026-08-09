from __future__ import annotations

import unittest

from resin import StatePool


class StatePoolTests(unittest.TestCase):
    def test_reference_counts_and_bulk_allocation(self) -> None:
        pool = StatePool(3)
        first, second = pool.allocate(2)
        self.assertEqual((pool.free_count, pool.active_count), (1, 2))
        pool.retain([first, first])
        self.assertEqual(pool.counts(first), (3, 0))
        pool.release([first, first, second])
        self.assertEqual(pool.counts(first), (1, 0))
        self.assertFalse(pool.contains(second))

    def test_completion_pin_defers_reuse(self) -> None:
        pool = StatePool(1)
        first = pool.allocate()[0]
        pool.pin([first])
        pool.release([first])
        self.assertEqual((pool.free_count, pool.retiring_count), (0, 1))
        with self.assertRaises(MemoryError):
            pool.allocate()

        pool.unpin([first])
        second = pool.allocate()[0]
        self.assertEqual(first.slot, second.slot)
        self.assertNotEqual(first.generation, second.generation)
        self.assertFalse(pool.contains(first))
        with self.assertRaisesRegex(ValueError, "stale"):
            pool.counts(first)

    def test_pool_identity_and_failed_bulk_operation_are_safe(self) -> None:
        first_pool = StatePool(1)
        second_pool = StatePool(1)
        handle = first_pool.allocate()[0]
        with self.assertRaisesRegex(ValueError, "different pool"):
            second_pool.retain([handle])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            first_pool.release([handle, handle])
        self.assertEqual(first_pool.counts(handle), (1, 0))


if __name__ == "__main__":
    unittest.main()
