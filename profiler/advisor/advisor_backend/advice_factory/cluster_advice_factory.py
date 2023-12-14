# Copyright (c) 2023, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys


from common_func.path_manager import PathManager
from advice_factory.advice_factory import AdviceFactory
from cluster_advice.slow_link_advice import SlowLinkAdvice
from cluster_advice.slow_rank_advice import SlowRankAdvice
from cluster_advice.kernel_cluster_advice import KernelClusterAdvice
from common_func_advisor.constant import Constant


class ClusterAdviceFactory(AdviceFactory):
    ADVICE_LIB = {
        Constant.SLOW_RANK: SlowRankAdvice,
        Constant.SLOW_LINK: SlowLinkAdvice,
        Constant.KERNEL: KernelClusterAdvice
    }

    def __init__(self, collection_path: str):
        super().__init__(collection_path)

    def path_check(self):
        """
        check whether input path is valid
        """
        PathManager.check_input_directory_path(self.collection_path)

    def produce_advice(self, advice: str):
        """
        produce data for input mode and advice
        """
        self.advice_check(advice)
        return self.ADVICE_LIB.get(advice)(self.collection_path).run()
