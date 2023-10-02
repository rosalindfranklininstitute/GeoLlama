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

######################################################
## Module             : GeoLlama/test_calc_by_slice ##
## Created            : Neville Yee                 ##
## Date created       : 02-Oct-2023                 ##
## Date last modified : 02-Oct-2023                 ##
######################################################

import unittest
import numpy as np

from GeoLlama import calc_by_slice as CBS


class CalcBySliceSmokeTest(unittest.TestCase):
    def test_create_slice_views(self):
        # Create dummy data
        data = np.random.random((100, 200, 300))

        # Create 2D views
        view_xy, view_zy, view_zx = CBS.create_slice_views(
            volume=data,
            coords=[50, 100, 150],
        )

        self.assertEqual(view_xy.shape, (200, 300))
        self.assertEqual(view_zy.shape, (100, 300))
        self.assertEqual(view_zx.shape, (100, 200))
