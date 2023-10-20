import unittest
import os
import fcntl
from unittest.mock import patch
from api_accuracy_checker.dump.api_info import APIInfo, ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.dump.info_dump import write_api_info_json, write_json, initialize_output_json
from api_accuracy_checker.common.utils import check_file_or_directory_path, initialize_save_path

class TestInfoDump(unittest.TestCase):
    # def test_write_api_info_json_forward(self):
    #     api_info = ForwardAPIInfo("test_forward_api", [1, 2, 3], {"a": 1, "b": 2})
    #     with patch('api_accuracy_checker.dump.info_dump.write_json') as mock_write_json:
    #         write_api_info_json(api_info)
    #         rank = os.getpid()
    #         mock_write_json.assert_called_with('./forward_info_rank.json', api_info.api_info_struct)
    #         mock_write_json.assert_called_with('./stack_info_rank.json', api_info.stack_info_struct, indent=4)

    def test_write_api_info_json_backward(self):
        api_info = BackwardAPIInfo("test_backward_api", [1, 2, 3])
        with patch('api_accuracy_checker.dump.info_dump.write_json') as mock_write_json:
            write_api_info_json(api_info)
            rank = os.getpid()
            mock_write_json.assert_called_with(f'./backward_info_{rank}.json', api_info.grad_info_struct)

    def test_write_api_info_json_invalid_type(self):
        api_info = APIInfo("test_api", True, True, "save_path")
        with self.assertRaises(ValueError):
            write_api_info_json(api_info)

    def test_write_json(self):
        file_path = 'dump_path/test.json'
        data = {"key": "value"}
        
    #     with patch('dump.info_dump.check_file_or_directory_path'), \
    #          patch('builtins.open', create=True), \
    #          patch('fcntl.flock'), \
    #          patch('builtins.json.dumps', return_value='{"key": "value"}'):
    #         write_json(file_path, data)
    #         # Assert that the file is opened in 'a+' mode
    #         open.assert_called_with(file_path, 'a+')
    #         # Assert that the file is locked and unlocked
    #         fcntl.flock.assert_called_with(open.return_value, fcntl.LOCK_EX)
    #         fcntl.flock.assert_called_with(open.return_value, fcntl.LOCK_UN)
    #         # Assert that the data is written to the file
    #         open.return_value.write.assert_called_with('{"key": "value"}')

    # def test_initialize_output_json(self):
    #     dump_path = 'dump_path'
    #     with patch('os.path.realpath', return_value=dump_path), \
    #          patch('dump.info_dump.check_file_or_directory_path'), \
    #          patch('os.path.exists', return_value=False), \
    #          patch('dump.info_dump.initialize_save_path'):
    #         initialize_output_json()
    #         # Assert that the dump path is checked and created
    #         check_file_or_directory_path.assert_called_with(dump_path, True)
    #         # Assert that the save paths are initialized if real_data is True
    #         initialize_save_path.assert_called_with(dump_path, 'forward_real_data')
    #         initialize_save_path.assert_called_with(dump_path, 'backward_real_data')
    #         # Assert that the files are checked and raise an error if they exist
    #         os.path.exists.assert_called_with('dump_path/forward_info.json')
    #         os.path.exists.assert_called_with('dump_path/backward_info.json')
    #         os.path.exists.assert_called_with('dump_path/stack_info.json')
    #         self.assertRaises(ValueError)



if __name__ == '__main__':
    unittest.main()