#!/usr/bin/env python

import numpy as np
import unittest
import time


class RingBuffer:
    """Each row contains `number_entries` entries. The ring buffer can add entries to and pop_back
    entries from each row. In addition it can read (without removing) entries form all rows.

    DEFINITIONS:
        Back: This is the entry that was added first, eg when adding the buffer contains [1, 2, 5]
         for a given row the back entry is 1.
        Front: This is the entry that was added last, eg in the above example the front entry is 5

    You can assume that all entries are integers.

    Assumption: Butter shall raise error on overflow.
    """
    def __init__(self, number_rows: int, number_entries: int, zeros=False):
        self.number_rows = number_rows
        # Buffer allocates one more element than `number_entries` to be able to assume
        # `begin`==`end` means an empty buffer and not a full one.
        self.row_length = number_entries + 1
        self.buf = (np.zeros if zeros else np.empty)(shape=(number_rows, self.row_length),
                                                     dtype=int)
        self.begin = np.zeros(number_rows, dtype=int)
        self.end = np.zeros(number_rows, dtype=int)

    def push_front(self, rows: np.ndarray, entries: np.ndarray):
        """Add new entries to the front of the respective row.

        Args:
            rows: Where the entries should be added, row indices must be unique.
            entries: values of new entries.
        """
        if np.unique(rows).size != len(rows.tolist()):
            raise ValueError('Row indices must be unique')
        rows_begin = self.begin[rows]
        rows_end = self.end[rows]
        rows_new = rows_end
        rows_end_after = (self.end[rows] + 1) % self.row_length
        if (rows_begin == rows_end_after).any():
            raise IndexError('Buffer is full')
        self.buf[rows, rows_new] = entries
        self.end[rows] = rows_end_after

    def push_back(self, rows: np.ndarray, entries: np.ndarray):
        """Add new entries to the back of the respective row.

        Args:
            rows: Where the entries should be added, row indices must be unique.
            entries: values of new entries.
        """
        rows_end = self.end[rows]
        rows_new = (self.begin[rows] - 1) % self.row_length
        rows_begin_after = rows_new
        if (rows_begin_after == rows_end).any():
            raise IndexError('Buffer is full')
        self.buf[rows, rows_new] = entries
        self.begin[rows] = rows_begin_after
        pass

    def pop_back(self, rows: np.ndarray) -> np.ndarray:
        """Returns the entry at the back and removes it from the ringbuffer.

        Args:
            rows: For which rows the back entry should be popped, row indices must be unique.
        Returns:
            Entries for given rows.
        """
        rows_begin = self.begin[rows]
        rows_end = self.end[rows]
        if (rows_begin == rows_end).any():
            raise IndexError('Buffer is empty')
        values = self.buf[rows, rows_begin]
        rows_begin_after = (self.begin[rows] + 1) % self.row_length
        self.begin[rows] = rows_begin_after
        return values

    def get_entries(self, offset: int) -> np.ndarray:
        """Returns the entries of each row with the given offset

        Args:
            offset: Offset with respect to the front (offset = 0 returns the front entry)

        Returns:
            Values in the ringbuffer (return -1 when there is no entry)
        """
        not_empty = np.logical_and(self.end != self.begin,
                                   offset <= (self.begin - self.end) % self.row_length)
        return np.where(
            not_empty,
            self.buf[np.arange(self.number_rows), (self.end - offset - 1) % self.row_length],
            -1)

    def get_number_of_entries(self) -> np.ndarray:
        """Returns the number of entries that are currently in the buffer for each row."""
        return (self.end - self.begin) % self.row_length


class RingBufferTest(unittest.TestCase):
    def test_push_pop(self):
        buffer = RingBuffer(number_entries=3, number_rows=4, zeros=True)

        buffer.push_front(np.array([1, 2]), np.array([5, 6]))
        self.assertEqual(buffer.get_number_of_entries().tolist(), [0, 1, 1, 0])

        buffer.push_front(np.array([0, 1, 2, 3]), np.array([8, 4, 8, 3]))
        self.assertEqual(buffer.get_number_of_entries().tolist(), [1, 2, 2, 1])

        extracted = buffer.pop_back(np.array([2, 0]))
        self.assertEqual(extracted.tolist(), [6, 8])
        self.assertEqual(buffer.get_number_of_entries().tolist(), [0, 2, 1, 1])

        buffer.push_front(np.array([1, 3, 2]), np.array([9, 0, 4]))
        self.assertEqual(buffer.get_number_of_entries().tolist(), [0, 3, 2, 2])
        self.assertEqual(buffer.get_entries(offset=0).tolist(), [-1, 9, 4, 0])
        self.assertEqual(buffer.get_entries(offset=1).tolist(), [-1, 4, 8, 3])
        self.assertEqual(buffer.get_entries(offset=10).tolist(), [-1, -1, -1, -1])

    def test_push_back(self):
        buffer = RingBuffer(number_rows=3, number_entries=3)
        buffer.push_back(np.array([0, 1]), np.array([2, 3]))
        self.assertEqual(buffer.pop_back(np.array([1, 0])).tolist(), [3, 2])

    def test_push_front_time_complexity(self):
        R = 5
        for N in [1000, 10_000, 100_000, 1_000_000]:
            start = time.time()
            buffer = RingBuffer(number_entries=N, number_rows=R)
            for i in range(N):
                buffer.push_front(np.arange(R), 42)
            end = time.time()
            elapsed = end - start
            self.assertEqual(buffer.get_number_of_entries().tolist(), [N, N, N, N, N])
            print('Filling with {:10,d} elements takes {:.6f} s.'.format(N, elapsed))


if __name__ == '__main__':
    unittest.main(exit=False)
