import unittest
from unittest.mock import patch, MagicMock
from ptdbg_ascend.online_dispatch.utils import *
import torch
import numpy as np
import os
import logging
from ptdbg_ascend.common.utils import CompareConst

COLOR_YELLOW='\033[33m'
COLOR_RED='\033[31m'
COLOR_RESET='\033[0m'

class TestUtils(unittest.TestCase):

    def test_get_callstack(self):
        stack = get_callstack()
        self.assertTrue(isinstance(stack, list))
        self.assertTrue(all(isinstance(item, list) for item in stack))

    @patch('numpy.save')
    def test_np_save_data(self, mock_save):
        data = np.array([1, 2, 3])
        np_save_data(data, 'test', '/tmp')
        mock_save.assert_called_once()
        data = torch.tensor([1, 2, 3])
        np_save_data(data, 'test', '/tmp')
        mock_save.assert_called()

    def test_data_to_cpu(self):
        tensor = torch.tensor([1, 2, 3], device='npu:0')
        result = data_to_cpu(tensor, 0, [])
        self.assertTrue(result.device.type == 'cpu')

    @patch('logging.getLogger')
    def test_get_mp_logger(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger = get_mp_logger()
        self.assertTrue(callable(logger))

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_debug(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_debug('test')
        mock_logger.assert_called_once_with('DEBUG test')

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_info(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_info('test')
        mock_logger.assert_called_once_with('INFO test')

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_warn(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_warn('test')
        mock_logger.assert_called_once_with(f'{COLOR_YELLOW}WARNING test {COLOR_RESET}')

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_error('test')
        mock_logger.assert_called_once_with(f'{COLOR_RED}ERROR test {COLOR_RESET}')

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_user(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_user('test')
        mock_logger.assert_called_once_with('test')

    @patch('ptdbg_ascend.online_dispatch.utils.get_mp_logger')
    def test_logger_logo(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        logger_logo()
        mock_logger.assert_called_once()

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_get_sys_info(self, mock_cpu_percent, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(total=8*1024*1024, available=4*1024*1024, used=4*1024*1024)
        mock_cpu_percent.return_value = 50
        sys_info = get_sys_info()
        self.assertIn('Total:', sys_info)
        self.assertIn('Free:', sys_info)
        self.assertIn('Used:', sys_info)
        self.assertIn('CPU:', sys_info)


if __name__ == '__main__':
    unittest.main()