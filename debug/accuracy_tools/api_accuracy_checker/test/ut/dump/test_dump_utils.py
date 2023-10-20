import unittest
import os
import numpy as np
from api_accuracy_checker.dump.utils import create_folder, write_npy

class TestUtils(unittest.TestCase):
    def test_create_folder(self):
        path = "test_folder"
        self.assertFalse(os.path.exists(path))
        create_folder(path)
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)

        path = "test_folder"
        os.makedirs(path, mode=0o750)
        self.assertTrue(os.path.exists(path))
        create_folder(path)
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)

    def test_write_npy(self):
        file_path = "test.npy"
        tensor = np.array([1, 2, 3])
        write_npy(file_path, tensor)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

        file_path = "test.npy"
        tensor = np.array([1, 2, 3])
        np.save(file_path, tensor)
        self.assertTrue(os.path.exists(file_path))
        with self.assertRaises(ValueError):
            write_npy(file_path, tensor)
        os.remove(file_path)

if __name__ == '__main__':
    unittest.main()