# Copyright 2023 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

#############################################
## Module             : GeoLlama/test_main ##
## Created            : Neville Yee        ##
## Date created       : 23-Apr-2024        ##
## Date last modified : 23-Apr-2024        ##
#############################################


import sys
import os
import tempfile
import unittest

import numpy as np
import mrcfile

from GeoLlama import main


class MainTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.tmpdir = tempfile.TemporaryDirectory()
        os.mkdir(f"{self.tmpdir.name}/data")


    def setUp(self):
        pass


    def test_check_cli_input(self):
        # Test for None path exception handling
        with self.assertRaises(ValueError) as cm:
            main._check_cli_input(
                path=None,
                pixel_size=1,
            )
        self.assertNotEqual(cm.exception, 0)

        # Test for non-directory path exception handling
        with self.assertRaises(NotADirectoryError) as cm:
            main._check_cli_input(
                path="random string",
                pixel_size=1,
            )
        self.assertNotEqual(cm.exception, 0)

        # Test for None pixel size exception handling
        with self.assertRaises(ValueError) as cm:
            main._check_cli_input(
                path=f"{self.tmpdir.name}/data",
                pixel_size=None,
            )
        self.assertNotEqual(cm.exception, 0)


    @classmethod
    def tearDownClass(self):
        self.tmpdir.cleanup()


    def tearDown(self):
        pass
