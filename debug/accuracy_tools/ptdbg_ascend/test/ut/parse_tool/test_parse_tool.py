import unittest
from unittest.mock import patch, MagicMock
from ptdbg_ascend.parse_tool.lib.parse_tool import ParseTool

class TestParseTool(unittest.TestCase):
    def setUp(self):
        self.parse_tool = ParseTool()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.Util.create_dir')
    def test_prepare(self, mock_create_dir):
        self.parse_tool.prepare()
        mock_create_dir.assert_called_once()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.argparse.ArgumentParser')
    def test_do_vector_compare(self, mock_argparse):
        mock_args = MagicMock()
        mock_args.my_dump_path = '/path/to/my_dump'
        mock_args.golden_dump_path = '/path/to/golden_dump'
        mock_args.output_path = None
        mock_args.ascend_path = None
        mock_argparse.parse_args.return_value = mock_args
        self.parse_tool.do_vector_compare()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.argparse.ArgumentParser')
    def test_do_convert_dump(self, mock_argparse):
        mock_args = MagicMock()
        mock_args.path = '/path/to/dump'
        mock_args.format = 'npy'
        mock_args.output_path = None
        mock_args.ascend_path = None
        mock_argparse.parse_args.return_value = mock_args
        self.parse_tool.do_convert_dump()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.argparse.ArgumentParser')
    def test_do_print_data(self, mock_argparse):
        mock_args = MagicMock()
        mock_args.path = '/path/to/data'
        mock_argparse.parse_args.return_value = mock_args
        self.parse_tool.do_print_data()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.argparse.ArgumentParser')
    def test_do_parse_pkl(self, mock_argparse):
        mock_args = MagicMock()
        mock_args.file_name = '/path/to/file.pkl'
        mock_args.api_name = 'api_name'
        mock_argparse.parse_args.return_value = mock_args
        self.parse_tool.do_parse_pkl()

    @patch('ptdbg_ascend.parse_tool.lib.parse_tool.argparse.ArgumentParser')
    def test_do_compare_data(self, mock_argparse):
        mock_args = MagicMock()
        mock_args.my_dump_path = '/path/to/my_dump'
        mock_args.golden_dump_path = '/path/to/golden_dump'
        mock_args.count = 20
        mock_args.save = False
        mock_args.atol = 0.001
        mock_args.rtol = 0.001
        mock_argparse.parse_args.return_value = mock_args
        self.parse_tool.do_compare_data(argv=None)
