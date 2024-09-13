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

###########################################
## Module             : GeoLlama/test_io ##
## Created            : Neville Yee      ##
## Date created       : 02-Oct-2023      ##
## Date last modified : 02-Oct-2023      ##
###########################################


import sys
import os
import tempfile
import unittest

import numpy as np
import mrcfile

from GeoLlama import (io, config)


class IOSmokeTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.params = config.objectify_user_input(
            autocontrast=True,
            adaptive=False,
            bandpass=False,
            data_path=None,
            num_cores=1,
            binning=1,
            pixel_size_nm=1.0,
            output_csv_path=None,
            output_star_path=None,
            output_mask=False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        self.test_data = np.full(
            shape = (100, 100, 100),
            fill_value = 1.0,
            dtype = np.float32
        )

        tmp_address = f"{self.tmpdir.name}/test.mrc"
        with mrcfile.new(tmp_address) as f:
            f.set_data(self.test_data)


    def test_read_mrc(self):
        os.chdir(self.tmpdir.name)
        # Test whether image remains unchanged if no binning
        data_out, px_size, shape_in, binning, data_in = io.read_mrc(fname="./test.mrc",
                                                           params=self.params,
        )
        self.assertIsInstance(data_out, np.ndarray)
        self.assertEqual(shape_in, (100, 100, 100))
        self.assertEqual(binning, 1)
        self.assertEqual(data_in, None)

        # Test whether image has been correctly binned
        self.params.binning = 2
        data_ds, px_size_ds, shape_in, binning, data_in = io.read_mrc(fname="./test.mrc",
                                                             params=self.params
        )
        self.assertEqual(data_ds.shape, (50, 50, 50))
        self.assertAlmostEqual(px_size_ds, 2.0, places=7)
        self.assertEqual(binning, 2)
        self.assertTrue(np.array_equal(data_in, self.test_data))


    @classmethod
    def tearDownClass(self):
        pass


    def tearDown(self):
        self.tmpdir.cleanup()
