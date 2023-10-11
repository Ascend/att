#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import shutil
import unittest
from ptdbg_ascend.advisor.advisor import Advisor
from ptdbg_ascend.common.utils import CompareException


class TestAdvisor(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs("test_result/output", exist_ok=True)
        self.output_path = os.path.abspath("test_result/output")

    def tearDown(self) -> None:
        shutil.rmtree("test_result/", ignore_errors=True)

    def test_analysis_when_csv_path_is_not_exist(self):
        advisor = Advisor("resources/compare/test.pkl", self.output_path)
        self.assertRaises(CompareException, advisor.analysis)

    def test_analysis_when_csv_path_is_invalid(self):
        advisor = Advisor("resources/compare/npu_test_1.pkl", self.output_path)
        self.assertRaises(CompareException, advisor.analysis)

    def test_analysis_when_csv_is_valid(self):
        advisor = Advisor("resources/compare/compare_result_20230703104808.csv", self.output_path)
        advisor.analysis()
        filenames = os.listdir(self.output_path)
        self.assertEqual(len(filenames), 1)
