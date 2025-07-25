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

###############################################
## Module             : GeoLlama/test_config ##
## Created            : Neville Yee          ##
## Date created       : 03-May-2024          ##
## Date last modified : 03-May-2024          ##
###############################################


import sys
import os
from pathlib import Path
import shutil
import unittest
import multiprocessing as mp

import numpy as np
import pandas as pd
import starfile

from geollama import config, objects

# from geollama.templates.report_template import params_str


class ConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.orig_path = Path(os.getcwd())
        self.tmpdir = Path(os.getcwd() + "/temp/")
        self.tmp_data = Path(self.tmpdir, "data")
        self.tmp_anlys = Path(self.tmpdir, "anlys")

        # Create folders
        self.tmpdir.mkdir(exist_ok=True)
        self.tmp_data.mkdir(exist_ok=True)
        self.tmp_anlys.mkdir(exist_ok=True)

    def setUp(self):
        pass

    def test_generate_config(self):
        os.chdir(self.tmp_anlys)

        config.generate_config(f"./new_config.yaml")
        self.assertTrue(os.path.exists("./new_config.yaml"))

    def test_read_config(self):
        # Generate default config file
        os.chdir(self.tmp_anlys)
        config.generate_config("./config.yaml")

        # Test if config file is read correctly
        params_dict = config.read_config("./config.yaml")
        self.assertTrue(isinstance(params_dict, dict))

        params = objects.Config(**params_dict)
        self.assertIsNone(params.data_path)
        self.assertIsNone(params.pixel_size_nm)
        self.assertEqual(params.binning, 0)
        self.assertTrue(params.autocontrast)
        self.assertTrue(params.adaptive)
        self.assertFalse(params.bandpass)
        self.assertEqual(params.num_cores, 1)
        self.assertIsNone(params.output_csv_path)
        self.assertEqual(params.output_star_path, "./output.star")
        self.assertTrue(params.generate_report)
        self.assertEqual(params.thickness_lower_limit, 120)
        self.assertEqual(params.thickness_upper_limit, 300)
        self.assertEqual(params.thickness_std_limit, 15)
        self.assertEqual(params.xtilt_std_limit, 5)
        self.assertEqual(params.displacement_limit, 25)
        self.assertEqual(params.displacement_std_limit, 5)

    def test_check_config(self):
        params_dict = dict(
            data_path=self.tmp_data,
            pixel_size_nm=1,
            binning=1,
            num_cores=1,
            autocontrast=False,
            adaptive=False,
            bandpass=False,
            output_csv_path=None,
            output_star_path=None,
            output_mask=False,
            printout=False,
            generate_report=False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )

        # Test for None path exception handling
        params = objects.Config(**params_dict)
        params.data_path = None
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        # Test for non-directory path exception handling
        params = objects.Config(**params_dict)
        params.data_path = "random string"
        with self.assertRaises(NotADirectoryError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        # Tests for wrong pixel size exception handling
        params = objects.Config(**params_dict)
        params.pixel_size_nm = None
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        params = objects.Config(**params_dict)
        params.pixel_size_nm = -1
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        # Tests for wrong number of cores exception handling
        params = objects.Config(**params_dict)
        params.num_cores = 1.5
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        params = objects.Config(**params_dict)
        params.num_cores = mp.cpu_count() + 1
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

        # Test for report generation exception handling
        params = objects.Config(**params_dict)
        params.generate_report = True
        with self.assertRaises(ValueError) as cm:
            params.validate()
        self.assertNotEqual(cm.exception, 0)

    @classmethod
    def tearDownClass(self):
        os.chdir(self.orig_path)
        shutil.rmtree(self.tmp_data, ignore_errors=True)
        shutil.rmtree(self.tmp_anlys, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tearDown(self):
        pass
