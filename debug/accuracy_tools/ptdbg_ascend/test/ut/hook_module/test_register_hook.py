import unittest
from unittest.mock import patch, MagicMock
from ptdbg_ascend.hook_module import register_hook

class TestRegisterHook(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.hook = MagicMock()

    def test_register_hook(self):
        with patch('ptdbg_ascend.hook_module.register_hook.register_hook_core') as mock_core:
            register_hook.register_hook(self.model, self.hook)
            mock_core.assert_called_once()