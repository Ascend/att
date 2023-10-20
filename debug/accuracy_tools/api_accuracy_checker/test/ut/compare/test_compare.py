import unittest
from api_accuracy_checker.api_accuracy_checker.compare.compare import Comparator
from unittest.mock import patch, MagicMock

class TestComparator(unittest.TestCase):
    @patch('os.path')
    @patch('api_accuracy_checker.common.utils.get_json_contents')
    def setUp(self, mock_get_json_contents, mock_path):
        mock_path.exists.return_value = False
        mock_get_json_contents.return_value = {}
        self.comparator = Comparator('test_path')

    def test_register_compare_algorithm(self):
        self.comparator.register_compare_algorithm('Test Algorithm', lambda x: x, None)
        self.assertIn('Test Algorithm', self.comparator.compare_alg)

    @patch('api_accuracy_checker.compare.compare_core')
    def test__compare_core_wrapper(self, mock_compare_core):
        mock_compare_core.return_value = ([], True, 'float32', 'float32', [])
        result, detailed_result = self.comparator._compare_core_wrapper([], [])
        self.assertTrue(result)
        self.assertEqual(detailed_result, [])

    def test__compare_dropout(self):
        class MockTensor:
            def numel(self):
                return 100
            def sum(self):
                return 50
            def __eq__(self, other):
                return self
        result, detailed_result = self.comparator._compare_dropout(MockTensor(), MockTensor())
        self.assertTrue(result)
        self.assertEqual(detailed_result, 1)

    @patch('api_accuracy_checker.compare.write_csv')
    def test_write_summary_csv(self, mock_write_csv):
        self.comparator.write_summary_csv(['test_api', 'PASS', 'PASS', [], []])
        mock_write_csv.assert_called()

    @patch('api_accuracy_checker.compare.write_csv')
    def test_write_detail_csv(self, mock_write_csv):
        self.comparator.write_detail_csv(['test_api', 'PASS', 'PASS', [], []])
        mock_write_csv.assert_called()

    @patch('api_accuracy_checker.compare.write_csv')
    def test_record_results(self, mock_write_csv):
        self.comparator.record_results('test_api', 'PASS', 'PASS', [], [])
        self.assertEqual(mock_write_csv.call_count, 2)

    @patch('api_accuracy_checker.compare.compare_core')
    def test_compare_output(self, mock_compare_core):
        mock_compare_core.return_value = ([], True, 'float32', 'float32', [])
        fwd_success, bwd_success = self.comparator.compare_output('test_api', [], [])
        self.assertTrue(fwd_success)
        self.assertEqual(bwd_success, 'NA')

if __name__ == '__main__':
    unittest.main()