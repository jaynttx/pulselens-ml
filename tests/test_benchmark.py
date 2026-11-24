import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('analyze', Path(__file__).resolve().parents[1] / 'benchmark/analyze.py')
analyze = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyze)

class BenchmarkTests(unittest.TestCase):
    def test_summary(self):
        result = analyze.summarize([4, -2])
        self.assertEqual(result['mae_bpm'], 3)
        self.assertEqual(result['bias_bpm'], 1)
        self.assertEqual(result['rmse_bpm'], math.sqrt(10))

    def test_invalid_reference_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('condition,method,app_bpm,reference_bpm\nGood light,pos,72,nan\n')
            with self.assertRaisesRegex(ValueError, 'line 2'):
                analyze.analyze(path)

if __name__ == '__main__':
    unittest.main()
