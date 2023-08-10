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
import csv
import json
import shutil
from common_func.constant import Constant


class FileManager:

    @classmethod
    def check_file_or_directory_path(cls, path, isdir=False):
        """
        Function Description:
            check whether the path is valid
        Parameter:
            path: the path to check
            isdir: the path is dir or file
        Exception Description:
            when invalid data throw exception
        """
        if isdir:
            if not os.path.exists(path):
                raise RuntimeError('The path {} is not exist.'.format(path))

            if not os.path.isdir(path):
                raise RuntimeError('The path {} is not a directory.'.format(path))

            if not os.access(path, os.W_OK):
                raise RuntimeError('The path {} does not have permission to write. '
                                   'Please check the path permission'.format(path))
        else:
            if not os.path.isfile(path):
                raise RuntimeError('{} is an invalid file or non-exist.'.format(path))

        if not os.access(path, os.R_OK):
            raise RuntimeError(
                'The path {} does not have permission to read. Please check the path permission'.format(path))

    @classmethod
    def read_csv_file(cls, file_path: str, class_bean: any) -> list:
        cls.check_file_or_directory_path(file_path)
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > Constant.MAX_CSV_SIZE:
            print(f"The file size exceeds the preset value {Constant.MAX_CSV_SIZE / 1024 / 1024}MB, "
                  f"please check the file: {file_path}")
            return []
        result_data = []
        try:
            with open(file_path, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    result_data.append(class_bean(row))
        except Exception:
            raise RuntimeError(f"Failed to read the file: {file_path}")
        return result_data

    @classmethod
    def read_json_file(cls, file_path: str) -> dict:
        cls.check_file_or_directory_path(file_path)
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return {}
        if file_size > Constant.MAX_JSON_SIZE:
            print(f"The file size exceeds the preset value {Constant.MAX_CSV_SIZE / 1024 / 1024}MB, "
                  f"please check the file: {file_path}")
            return {}
        try:
            with open(file_path, "r") as json_file:
                result_data = json.load(json_file)
        except Exception:
            raise RuntimeError(f"Failed to read the file: {file_path}")
        return result_data

    @classmethod
    def create_csv_file(cls, profiler_path: str, data: list, file_name: str, headers: list = None) -> None:
        if not data:
            return
        file_path = os.path.join(profiler_path, Constant.OUTPUT_DIR, file_name)
        try:
            with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w",
                           newline="") as file:
                writer = csv.writer(file)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
        except Exception:
            raise RuntimeError(f"Can't create file: {file_path}")

    @classmethod
    def create_json_file(cls, profiler_path: str, data: list, file_name: str) -> None:
        if not data:
            return
        file_path = os.path.join(profiler_path, Constant.OUTPUT_DIR, file_name)
        cls.create_json_file_by_path(file_path, data)

    @classmethod
    def create_json_file_by_path(cls, output_path: str, data: list) -> None:
        dir_name = os.path.dirname(output_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, mode=Constant.DIR_AUTHORITY)
            except Exception:
                raise RuntimeError(f"Can't create directory: {dir_name}")
        try:
            with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w") as file:
                json.dump(data, file)
        except Exception:
            raise RuntimeError(f"Can't create file: {output_path}")

    @classmethod
    def remove_and_make_output_dir(cls, profiler_path) -> None:
        output_path = os.path.join(profiler_path, Constant.OUTPUT_DIR)
        if os.path.isdir(output_path):
            try:
                shutil.rmtree(output_path)
                os.makedirs(output_path, mode=Constant.DIR_AUTHORITY)
            except Exception:
                raise RuntimeError(f"Can't delete files in the directory: {output_path}")
            return
        try:
            os.makedirs(output_path, mode=Constant.DIR_AUTHORITY)
        except Exception:
            raise RuntimeError(f"Can't create directory: {output_path}")