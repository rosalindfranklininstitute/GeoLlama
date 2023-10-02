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

from GeoLlama import io


class IOSmokeTest(unittest.TestCase):
    def _create_dummy_image(self):
        tmpdir = tempfile.TemporaryDirectory()

        test_data = np.full(
            shape = (100, 100, 100),
            fill_value = 1.0,
            dtype = np.float32
        )

        tmp_address = f"{tmpdir.name}/test.mrc"
        with mrcfile.new(tmp_address) as f:
            f.set_data(test_data)

        return tmpdir, test_data


    def test_read_mrc(self):
        tmpdir, raw_img = self._create_dummy_image()
        test_px_size = 1.0

        os.chdir(tmpdir.name)
        # Test whether image remains unchanged if no binning
        data, px_size = io.read_mrc(fname="./test.mrc",
                                    px_size_nm=test_px_size,
                                    downscale=1
        )
        self.assertIsInstance(data, np.ndarray)
        self.assertTrue(data.shape==raw_img.shape)
        self.assertTrue(np.array_equal(data, raw_img))

        # Test whether image has been correctly binned
        data_ds, px_size_ds = io.read_mrc(fname="./test.mrc",
                                          px_size_nm=test_px_size,
                                          downscale=2
        )
        self.assertTrue(data_ds.shape==(50, 50, 50))
        self.assertAlmostEqual(px_size_ds, 2.0, places=7)

        tmpdir.cleanup()
