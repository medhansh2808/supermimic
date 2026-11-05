import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_eval import summarize


class SummaryTests(unittest.TestCase):
    def test_prefers_zero_failure_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eval.csv"
            path.write_text("iteration,train_wall_h,norm_return,n_failed\n10,0.2,0.90,2\n20,0.4,0.85,0\n")
            summary = summarize(path)
            self.assertIn("Selected checkpoint iteration: **20**", summary)
            self.assertIn("0/40", summary)
            self.assertIn("0.9000", summary)

    def test_no_zero_failure_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eval.csv"
            path.write_text("iteration,norm_return,n_failed\n10,0.7,4\n")
            summary = summarize(path)
            self.assertIn("No zero-failure", summary)
            self.assertIn("Selected checkpoint iteration: **10**", summary)


if __name__ == "__main__":
    unittest.main()
