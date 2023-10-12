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
