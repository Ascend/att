import unittest
from ptdbg_ascend.debugger.debugger_config import DebuggerConfig

class TestDebuggerConfig(unittest.TestCase):
    def setUp(self):
        self.dump_path = "/path/to/dump"
        self.hook_name = "dump"
        self.rank = 0
        self.step = [1, 2, 3]

    def test_init(self):
        debugger_config = DebuggerConfig(self.dump_path, self.hook_name, self.rank, self.step)
        self.assertEqual(debugger_config.dump_path, self.dump_path)
        self.assertEqual(debugger_config.hook_name, self.hook_name)
        self.assertEqual(debugger_config.rank, self.rank)
        self.assertEqual(debugger_config.step, self.step)

    def test_check_hook_name(self):
        debugger_config = DebuggerConfig(self.dump_path, self.hook_name, self.rank, self.step)
        with self.assertRaises(ValueError):
            debugger_config.hook_name = "invalid_hook_name"
            debugger_config._check_hook_name()

    def test_check_rank(self):
        debugger_config = DebuggerConfig(self.dump_path, self.hook_name, self.rank, self.step)
        with self.assertRaises(ValueError):
            debugger_config.rank = -1
            debugger_config._check_rank()

    def test_check_step(self):
        debugger_config = DebuggerConfig(self.dump_path, self.hook_name, self.rank, self.step)
        with self.assertRaises(ValueError):
            debugger_config.step = "invalid_step"
            debugger_config._check_step()

