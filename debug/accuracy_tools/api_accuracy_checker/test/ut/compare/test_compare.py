import unittest
from api_accuracy_checker.compare.compare import Comparator
from unittest.mock import patch, MagicMock

class TestComparator(unittest.TestCase):
    @patch('os.path')
    def setUp(self, mock_path):
        mock_path.exists.return_value = False
        self.comparator = Comparator('test_path')

    def test_register_compare_algorithm(self):
        self.comparator.register_compare_algorithm('Test Algorithm', lambda x: x, None)
        self.assertIn('Test Algorithm', self.comparator.compare_alg)

    @patch('compare.compare_core')
    def test__compare_core_wrapper(self, mock_compare_core):
        mock_compare_core.return_value = ('test_result', True, 'bench_dtype', 'npu_dtype', 'shape')
        result, detailed_result = self.comparator._compare_core_wrapper('bench_out', 'npu_out')
        self.assertTrue(result)
        self.assertEqual(detailed_result[0], ('bench_dtype', 'npu_dtype', 'shape', 'test_result', 'True'))

    def test__compare_dropout(self):
        class MockTensor:
            def numel(self):
                return 200
            def sum(self):
                return 50
            def cpu(self):
                return self
        bench_out = MockTensor()
        npu_out = MockTensor()
        result, detailed_result = self.comparator._compare_dropout(bench_out, npu_out)
        self.assertTrue(result)
        self.assertEqual(detailed_result, 1)

    @patch('compare.write_csv')
    def test_write_summary_csv(self, mock_write_csv):
        self.comparator.stack_info = {'test_api': ['stack_info']}
        self.comparator.write_summary_csv(('test_api', 'SKIP', 'SKIP', 'test_message'))
        mock_write_csv.assert_called_once()

    @patch('compare.write_csv')
    def test_write_detail_csv(self, mock_write_csv):
        self.comparator.write_detail_csv(('test_api', 'SKIP', 'SKIP', ['test_fwd_result'], ['test_bwd_result']))
        mock_write_csv.assert_called_once()

    @patch('compare.write_csv')
    def test_record_results(self, mock_write_csv):
        self.comparator.record_results('test_api', 'SKIP', 'SKIP', ['test_fwd_result'], ['test_bwd_result'])
        self.assertEqual(mock_write_csv.call_count, 2)

    @patch('compare.compare_core')
    def test_compare_output(self, mock_compare_core):
        mock_compare_core.return_value = ('test_result', True, 'bench_dtype', 'npu_dtype', 'shape')
        fwd_success, bwd_success = self.comparator.compare_output('test_api', 'bench_out', 'npu_out')
        self.assertTrue(fwd_success)
        self.assertEqual(bwd_success, 'NA')

if __name__ == '__main__':
    unittest.main()