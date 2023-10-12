import unittest
from unittest.mock import patch, MagicMock
from ptdbg_ascend.debugger.precision_debugger import PrecisionDebugger
from ptdbg_ascend.dump.dump import DumpUtil

class TestPrecisionDebugger(unittest.TestCase):

    def setUp(self):
        self.precision_debugger = PrecisionDebugger(dump_path='test_path', hook_name='dump')

    def test_init(self):
        self.assertEqual(self.precision_debugger.config.dump_path, 'test_path')
        self.assertEqual(self.precision_debugger.config.hook_name, 'dump')

    def test_get_configure_hook_dump(self):
        hook = self.precision_debugger.get_configure_hook('dump')
        self.assertEqual(hook, self.precision_debugger.configure_full_dump)

    def test_get_configure_hook_overflow(self):
        hook = self.precision_debugger.get_configure_hook('overflow_check')
        self.assertEqual(hook, self.precision_debugger.configure_overflow_dump)

    def test_configure_full_dump(self):
        self.assertRaises(ValueError, self.precision_debugger.configure_full_dump, mode='acl', acl_config=None)

    def test_configure_overflow_dump(self):
        self.assertRaises(ValueError, self.precision_debugger.configure_overflow_dump, overflow_nums='invalid')

    @patch('ptdbg_ascend.debugger.precision_debugger.register_hook_core')
    def test_start(self, mock_register_hook_core):
        PrecisionDebugger.first_start = True
        PrecisionDebugger.start()
        mock_register_hook_core.assert_called_once()

    @patch('ptdbg_ascend.debugger.precision_debugger.write_to_disk')
    def test_stop(self, mock_write_to_disk):
        PrecisionDebugger.stop()
        mock_write_to_disk.assert_called_once()

    def test_step(self):
        initial_iter_num = DumpUtil.iter_num
        PrecisionDebugger.step()
        self.assertEqual(DumpUtil.iter_num, initial_iter_num + 1)

    @patch('ptdbg_ascend.debugger.precision_debugger.PrecisionDebugger.step')
    @patch('ptdbg_ascend.debugger.precision_debugger.PrecisionDebugger.start')
    def test_incr_iter_num_maybe_exit(self, mock_start, mock_step):
        PrecisionDebugger.incr_iter_num_maybe_exit()
        mock_step.assert_called_once()
        mock_start.assert_called_once()

