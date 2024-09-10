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
import tempfile
import unittest
import multiprocessing as mp

import numpy as np
import pandas as pd
import starfile

from GeoLlama import (config, objects)


class ConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.tmpdir = tempfile.TemporaryDirectory()
        os.mkdir(f"{self.tmpdir.name}/data")
        os.mkdir(f"{self.tmpdir.name}/anlys")


    def setUp(self):
        pass


    def test_generate_config(self):
        os.chdir(f"{self.tmpdir.name}/anlys")

        config.generate_config(f"./new_config.yaml")
        self.assertTrue(os.path.exists("./new_config.yaml"))


    def test_read_config(self):
        # Generate default config file
        os.chdir(f"{self.tmpdir.name}/anlys")
        config.generate_config("./config.yaml")

        # Test if config file is read correctly
        params = config.read_config("./config.yaml")
        print(params)

        self.assertTrue(isinstance(params, objects.Config))
        self.assertIsNone(params.data_path)
        self.assertIsNone(params.pixel_size_nm)
        self.assertEqual(params.binning, 0)
        self.assertTrue(params.autocontrast)
        self.assertTrue(params.adaptive)
        self.assertFalse(params.bandpass)
        self.assertEqual(params.num_cores, 1)
        self.assertIsNone(params.output_csv_path)
        self.assertIsNone(params.output_star_path)
        self.assertEqual(params.thickness_lower_limit, 120)
        self.assertEqual(params.thickness_upper_limit, 300)
        self.assertEqual(params.thickness_std_limit, 15)
        self.assertEqual(params.xtilt_std_limit, 5)
        self.assertEqual(params.displacement_limit, 25)
        self.assertEqual(params.displacement_std_limit, 5)


    def test_objectify_user_input(self):
        params = config.objectify_user_input(
            autocontrast = False,
            adaptive = False,
            bandpass = False,
            data_path = None,
            pixel_size_nm = None,
            binning = 1,
            num_cores = 1,
            output_csv_path = None,
            output_star_path = None,
            output_mask = False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )

        self.assertTrue(isinstance(params, objects.Config))
        self.assertEqual(params.data_path, None)
        self.assertEqual(params.pixel_size_nm, None)
        self.assertEqual(params.binning, 1)
        self.assertFalse(params.autocontrast)
        self.assertFalse(params.adaptive)
        self.assertFalse(params.bandpass)
        self.assertEqual(params.num_cores, 1)
        self.assertEqual(params.output_csv_path, None)
        self.assertEqual(params.output_star_path, None)
        self.assertFalse(params.output_mask)
        self.assertEqual(params.thickness_lower_limit, 120)
        self.assertEqual(params.thickness_upper_limit, 300)
        self.assertEqual(params.thickness_std_limit, 15)
        self.assertEqual(params.xtilt_std_limit, 5)
        self.assertEqual(params.displacement_limit, 25)
        self.assertEqual(params.displacement_std_limit, 5)


    def test_check_config(self):
        params = config.objectify_user_input(
            data_path = f"{self.tmpdir.name}/data",
            pixel_size_nm = 1,
            binning = 1,
            num_cores = 1,
            autocontrast = False,
            adaptive = False,
            bandpass = False,
            output_csv_path = None,
            output_star_path = None,
            output_mask = False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )

        # Test for None path exception handling
        params.data_path = None
        with self.assertRaises(ValueError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)

        # Test for non-directory path exception handling
        params.data_path = "random string"
        with self.assertRaises(NotADirectoryError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)

        params.data_path = f"{self.tmpdir.name}/data" # Reset params data_path to acceptable value

        # Tests for wrong pixel size exception handling
        params.pixel_size_nm = None
        with self.assertRaises(ValueError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)

        params.pixel_size_nm = -1
        with self.assertRaises(ValueError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)

        params.pixel_size_nm = 1 # Reset params pixel_size_nm to acceptable value

        # Tests for wrong number of cores exception handling
        params.num_cores = 1.5
        with self.assertRaises(ValueError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)

        params.num_cores = mp.cpu_count() + 1
        with self.assertRaises(ValueError) as cm:
            config.check_config(params)
        self.assertNotEqual(cm.exception, 0)


    @classmethod
    def tearDownClass(self):
        self.tmpdir.cleanup()


    def tearDown(self):
        pass
