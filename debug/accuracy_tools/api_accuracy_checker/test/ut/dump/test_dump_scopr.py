import unittest
from api_accuracy_checker.dump.dump_scope import *

class TestDumpScope(unittest.TestCase):
    def test_iter_tracer(self):
        def dummy_func():
            return "Hello, World!"
        
        wrapped_func = iter_tracer(dummy_func)
        result = wrapped_func()
        self.assertEqual(DumpUtil.dump_switch, "ON")
        self.assertEqual(result, "Hello, World!")

        def another_dummy_func():
            return 123
        wrapped_func = iter_tracer(another_dummy_func)
        result = wrapped_func()
        self.assertEqual(DumpUtil.dump_switch, "ON")
        self.assertEqual(result, 123)
