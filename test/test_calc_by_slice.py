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
## Date last modified : 25-Apr-2024                 ##
######################################################

import unittest
from types import NoneType

import numpy as np

from GeoLlama import calc_by_slice as CBS


class CalcBySliceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass


    def setUp(self):
        self.image_2d = np.random.random((200, 200))
        self.image_3d = np.random.random((100, 200, 300))
        self.coords = np.random.random((20, 3))


    def test_filter_bandpass(self):
        out = CBS.filter_bandpass(
            image=self.image_2d
        )

        self.assertEqual(out.shape, self.image_2d.shape)


    def test_autocontrast_slice(self):
        out = CBS.autocontrast_slice(
            image=self.image_2d
        )

        self.assertEqual(out.shape, self.image_2d.shape)


    def test_create_slice_views(self):
        view_zy, view_zx = CBS.create_slice_views(
            volume=self.image_3d,
            sliding_window_width=10,
        )

        self.assertEqual(view_zy.shape, (191, 100, 300))
        self.assertEqual(view_zx.shape, (291, 100, 200))


    def test_interpolate_surface(self):
        # Create dummy data
        data = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=float)

        # Interpolate
        x_grid, y_grid, surface_mesh = CBS.interpolate_surface(
            mesh_points=data,
            n_points=10
        )

        # Tests
        self.assertIsInstance(x_grid, np.ndarray)
        self.assertIsInstance(y_grid, np.ndarray)
        self.assertIsInstance(surface_mesh, np.ndarray)

        self.assertEqual(x_grid.shape, (10, 10))
        self.assertEqual(y_grid.shape, (10, 10))
        self.assertEqual(surface_mesh.shape, (10, 10))


    def test_generalised_theil_sen_fit(self):
        # Run TS fit
        grad_norm, ptl_median, mse = CBS.generalised_theil_sen_fit(
            contour_pts=self.coords
        )

        # Tests
        self.assertIsInstance(grad_norm, np.ndarray)
        self.assertIsInstance(ptl_median, np.ndarray)
        self.assertIsInstance(mse, float)

        self.assertEqual(grad_norm.shape, (3,))
        self.assertEqual(ptl_median.shape, (3,))


    def test_leave_one_out(self):
        # Run leave_one_out
        out = CBS.leave_one_out(
            contour_pts=self.coords
        )

        # Tests
        self.assertIsInstance(out, int | NoneType)


    def test_refine_contour_LOO(self):
        # Run refine_contour_LOO
        out = CBS.refine_contour_LOO(
            contour_pts=self.coords
        )

        # Tests
        self.assertIsInstance(out, list)


    def test_evaluate_slice(self):
        # Run evaluate_slice
        (displacement, breadth, thickness, angle, num_points,
         surface_t1, surface_t2,
         surface_b1, surface_b2) = CBS.evaluate_slice(
             view_input=self.image_2d,
             pixel_size_nm=1.0
         )

        # Tests
        self.assertIsInstance(displacement, float)
        self.assertTrue(0 <= displacement <= 100, "Lamella centroid displacement must be between 0 and 100%.")

        self.assertIsInstance(breadth, float)
        self.assertGreaterEqual(breadth, 0, "Lamella breadth must be >= 0 (nm).")

        self.assertIsInstance(thickness, float)
        self.assertGreaterEqual(thickness, 0, "Lamella thickness must be >= 0 (nm).")

        self.assertIsInstance(angle, float)
        self.assertTrue(-180 <= angle <= 180, "Lamella angle must be between -180 and 180 (degs).")

        self.assertIsInstance(num_points, int)
        self.assertGreaterEqual(num_points, 0, "Number of features picked must be >= 0.")

        self.assertIsInstance(surface_t1, np.ndarray)
        self.assertEqual(surface_t1.shape, (2,))

        self.assertIsInstance(surface_t2, np.ndarray)
        self.assertEqual(surface_t2.shape, (2,))

        self.assertIsInstance(surface_b1, np.ndarray)
        self.assertEqual(surface_b1.shape, (2,))

        self.assertIsInstance(surface_b2, np.ndarray)
        self.assertEqual(surface_b2.shape, (2,))


    @unittest.skip("Wrapper for functions")
    def test_evaluate_full_lamella(self):
        pass


    @classmethod
    def tearDownClass(self):
        pass


    def tearDown(self):
        pass
