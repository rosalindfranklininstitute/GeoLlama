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
import unittest.mock as mock

import numpy as np
import pandas as pd
import starfile

from GeoLlama import (main, evaluate)


class MainTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.tmpdir = tempfile.TemporaryDirectory()
        os.mkdir(f"{self.tmpdir.name}/data")
        os.mkdir(f"{self.tmpdir.name}/anlys")


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


    def test_generate_config(self):
        os.chdir(f"{self.tmpdir.name}/anlys")

        # Test if correct file is generated with default output path
        main.generate_config()
        self.assertTrue(os.path.exists("./config.yaml"))

        # Test if correct file is generated with default output path
        main.generate_config(f"./new_config.yaml")
        self.assertTrue(os.path.exists("./new_config.yaml"))


    def test_main(self):
        os.chdir(f"{self.tmpdir.name}/anlys")

        # Create random dataframe as mock output from eval_batch
        mock_df = pd.DataFrame(np.random.randint(0, 100, size=(10, 9)),
                               columns=[
                                   "filename",
                                   "Mean_thickness_nm",
                                   "Thickness_s.d._nm",
                                   "Mean_X-tilt_degs",
                                   "X-tilt_s.d._degs",
                                   "Mean_Y-tilt_degs",
                                   "Y-tilt_s.d._degs",
                                   "thickness_anomaly",
                                   "xtilt_anomaly"
                               ]
        )

        with mock.patch.object(evaluate, "eval_batch", return_value=(mock_df, mock_df)) as m:
            main.main(
                user_path = "../data",
                pixel_size = 1,
                out_csv = "./test.csv",
                out_star = "./test.star",
            )

            # Test if files are created
            self.assertTrue(os.path.exists("./test.csv"))
            self.assertTrue(os.path.exists("./test.star"))

            # Read in created files
            csv_df = pd.read_csv("./test.csv", index_col=False)
            star_df = starfile.read("./test.star")

            # Test if exported values are correct
            pd.testing.assert_frame_equal(mock_df, csv_df)
            pd.testing.assert_frame_equal(mock_df, star_df)


    @classmethod
    def tearDownClass(self):
        self.tmpdir.cleanup()


    def tearDown(self):
        pass
