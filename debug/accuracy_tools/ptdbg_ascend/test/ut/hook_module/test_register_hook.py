import unittest
from unittest.mock import patch, MagicMock
from ptdbg_ascend.hook_module import register_hook
from ptdbg_ascend.dump.dump import acc_cmp_dump

class TestRegisterHook(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.hook = acc_cmp_dump

    def test_register_hook(self):
        with patch('ptdbg_ascend.hook_module.register_hook.register_hook_core') as mock_core:
            register_hook.register_hook(self.model, self.hook)
            mock_core.assert_called_once()