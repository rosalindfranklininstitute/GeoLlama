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


import os
import shutil
import unittest
import platform
import tempfile
from pathlib import Path

import numpy as np
import mrcfile

from geollama import io, config, objects


class IOSmokeTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.orig_path = Path(os.getcwd())

        if mysys := platform.system() == "Windows":
            self.tmpdir = Path(self.orig_path, "temp")
            self.tmpdir.mkdir(exist_ok=True)
        else:
            self.tmpdir = Path(tempfile.mkdtemp())
        self.tmp_data = Path(self.tmpdir, "data")
        self.tmp_anlys = Path(self.tmpdir, "anlys")

        # Create folders
        self.tmp_data.mkdir(exist_ok=True)
        self.tmp_anlys.mkdir(exist_ok=True)

        self._params_dict = dict(
            autocontrast=True,
            adaptive=False,
            bandpass=False,
            data_path=None,
            num_cores=1,
            binning=1,
            pixel_size_nm=1.0,
            output_csv_path=None,
            output_star_path=None,
            generate_report=None,
            output_mask=False,
            printout=False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )
        self.params = objects.Config(**self._params_dict)

        self.test_data = np.full(
            shape=(100, 100, 100), fill_value=1.0, dtype=np.float32
        )

        tmp_address = Path(self.tmp_data, "test.mrc")
        with mrcfile.new(tmp_address) as f:
            f.set_data(self.test_data)

    def setUp(self):
        pass

    def test_read_mrc(self):
        os.chdir(self.tmpdir)
        # Test whether image remains unchanged if no binning
        data_out, px_size, shape_in, binning, data_in = io.read_mrc(
            fname=Path(self.tmp_data, "test.mrc"),
            params=self.params,
        )
        self.assertIsInstance(data_out, np.ndarray)
        self.assertEqual(shape_in, (100, 100, 100))
        self.assertEqual(binning, 1)
        self.assertEqual(data_in, None)

        # Test whether image has been correctly binned
        self.params.binning = 2
        data_ds, px_size_ds, shape_in, binning, data_in = io.read_mrc(
            fname=Path(self.tmp_data, "test.mrc"), params=self.params
        )
        self.assertEqual(data_ds.shape, (50, 50, 50))
        self.assertAlmostEqual(px_size_ds, 2.0, places=7)
        self.assertEqual(binning, 2)
        self.assertTrue(np.array_equal(data_in, self.test_data))

    @classmethod
    def tearDownClass(self):
        os.chdir(self.orig_path)
        shutil.rmtree(self.tmp_data, ignore_errors=True)
        shutil.rmtree(self.tmp_anlys, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tearDown(self):
        pass
