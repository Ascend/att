# coding=utf-8
import pytest
import unittest
from ptdbg_ascend.overflow_check import utils
from ptdbg_ascend.overflow_check.utils import OverFlowUtil, dump_overflow

ON = "ON"
OFF = "OFF"


class TestUtilsMethods(unittest.TestCase):

    def test_set_overflow_check_switch_error1(self):
        with pytest.raises(Exception) as error:
            res = OverFlowUtil.set_overflow_check_switch("abc")
            self.assertEqual(error.type, TypeError)

    def test_set_overflow_check_switch_error2(self):
        with pytest.raises(Exception) as error:
            res = utils.set_overflow_check_switch("abc")
            self.assertEqual(error.type, AssertionError)

    def test_set_overflow_check_switch_error3(self):
        with pytest.raises(Exception) as error:
            res = utils.set_overflow_check_switch(ON, "abc")
            self.assertEqual(error.type, AssertionError)

    def test_OverFlowUtil_set_overflow_check_switch(self):
        OverFlowUtil.set_overflow_check_switch(ON, OFF)
        self.assertEqual(OverFlowUtil.overflow_check_switch, ON)
        self.assertEqual(OverFlowUtil.overflow_filter_switch, OFF)

    def test_get_overflow_check_switch(self):
        res = OverFlowUtil.get_overflow_check_switch()
        self.assertEqual(res, True)

    def test_inc_overflow_dump_times(self):
        OverFlowUtil.inc_overflow_dump_times()
        self.assertEqual(OverFlowUtil.real_overflow_dump_times, 1)

    def test_check_overflow_dump_times(self):
        res = OverFlowUtil.check_overflow_dump_times(100)
        self.assertEqual(res, True)

    def test_set_overflow_check_switch_success1(self):
        utils.set_overflow_check_switch(OFF, OFF)
        self.assertEqual(OverFlowUtil.overflow_check_switch, OFF)
        self.assertEqual(OverFlowUtil.overflow_filter_switch, OFF)

    def test_set_overflow_check_switch_success2(self):
        utils.set_overflow_check_switch(ON)
        self.assertEqual(OverFlowUtil.overflow_check_switch, ON)
        self.assertEqual(OverFlowUtil.overflow_filter_switch, ON)

    def test_set_overflow_check_switch_success3(self):
        utils.set_overflow_check_switch(ON, ON)
        self.assertEqual(OverFlowUtil.overflow_check_switch, ON)
        self.assertEqual(OverFlowUtil.overflow_filter_switch, ON)
