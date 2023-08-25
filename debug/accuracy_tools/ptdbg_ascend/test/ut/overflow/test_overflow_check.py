# coding=utf-8
import os
import pytest
import unittest
from ptdbg_ascend.overflow_check import overflow_check
from ptdbg_ascend.overflow_check import utils
from ptdbg_ascend.overflow_check.utils import OverFlowUtil, dump_overflow

ON = "ON"
OFF = "OFF"
ERROR_PID = 1


class TestUtilsMethods(unittest.TestCase):

    def test_check_overflow_environment_1(self):
        utils.set_overflow_check_switch(OFF, OFF)
        OverFlowUtil.get_overflow_check_switch()
        res = overflow_check.check_overflow_environment(ERROR_PID)
        self.assertEqual(res, False)

    def test_check_overflow_environment_2(self):
        utils.set_overflow_check_switch(ON, ON)
        OverFlowUtil.get_overflow_check_switch()
        res = overflow_check.check_overflow_environment(ERROR_PID)
        self.assertEqual(res, False)

    def test_check_overflow_environment_3(self):
        utils.set_overflow_check_switch(ON, ON)
        OverFlowUtil.get_overflow_check_switch()
        pid = os.getpid()
        overflow_check.is_gpu = True
        res = overflow_check.check_overflow_environment(pid)
        self.assertEqual(res, False)

    def test_check_overflow_environment_4(self):
        utils.set_overflow_check_switch(ON, ON)
        OverFlowUtil.get_overflow_check_switch()
        pid = os.getpid()
        overflow_check.is_gpu = False
        overflow_check.backward_init_status = True
        res = overflow_check.check_overflow_environment(pid)
        self.assertEqual(res, False)

    def test_check_overflow_environment_5(self):
        utils.set_overflow_check_switch(ON, ON)
        OverFlowUtil.get_overflow_check_switch()
        pid = os.getpid()
        overflow_check.is_gpu = False
        overflow_check.backward_init_status = False
        res = overflow_check.check_overflow_environment(pid)
        self.assertEqual(res, True)

