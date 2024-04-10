from __future__ import annotations

import unittest

from . import sparsity


class SparsityAnalyzerTest(unittest.TestCase):
    def test_init(self):
        sparsity_test = sparsity.SparsityAnalyzer(threshold=0.5)
        self.assertEqual(sparsity_test.threshold, 0.5)
