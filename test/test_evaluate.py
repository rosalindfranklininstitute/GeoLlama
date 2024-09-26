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

#################################################
## Module             : GeoLlama/test_evaluate ##
## Created            : Neville Yee            ##
## Date created       : 11-Apr-2024            ##
## Date last modified : 11-Apr-2024            ##
#################################################


import sys
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import mrcfile

from GeoLlama import evaluate as EV
from GeoLlama import config


class EvaluateTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up temp folder structure
        self.tmpdir = tempfile.TemporaryDirectory()
        os.mkdir(f"{self.tmpdir.name}/data")
        os.mkdir(f"{self.tmpdir.name}/anlys")
        os.mkdir(f"{self.tmpdir.name}/anlys/surface_models")

        self._data_folder = Path(f"{self.tmpdir.name}/data")
        self._anlys_folder = Path(f"{self.tmpdir.name}/anlys")
        self._models_folder = Path(f"{self.tmpdir.name}/anlys/surface_models")

        self._test_data = np.random.random(
            size=(100, 100, 100),
        )

        self._data_address = Path(f"{self._data_folder}/test.mrc")
        with mrcfile.new(self._data_address) as f:
            f.set_data(self._test_data.astype(np.float32))

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
            generate_report=False,
            output_mask=False,
            printout=False,
            thickness_lower_limit=120,
            thickness_upper_limit=300,
            thickness_std_limit=15,
            xtilt_std_limit=5,
            displacement_limit=25,
            displacement_std_limit=5,
        )

        # Run single analysis to get output for subsequent tests
        os.chdir(self._anlys_folder)
        self._single_out = EV.eval_single(
            fname=self._data_address,
            params=self.params
        )

        # Set up a dummy Lamella object
        self._my_lamella = EV.Lamella(
            centroid = [50, 50, 50],
            breadth = 100,
            thickness = 50,
            xtilt = 0,
            ytilt = 0
        )


    def setUp(self):
        pass


    def test_find_files(self):
        filelist = EV.find_files(self._data_folder)

        # Tests
        self.assertIsInstance(filelist, list)
        self.assertEqual(len(filelist), 1)


    def test_save_figure(self):
        surface_info = self._single_out[6]
        fig_path = Path(f"{self._models_folder}/test_fig.png")
        EV.save_figure(
            surface_info=surface_info,
            save_path=fig_path,
            binning=1
        )

        self.assertTrue(fig_path.is_file())


    def test_save_text_model(self):
        surface_info = self._single_out[6]
        fig_path = Path(f"{self._models_folder}/test_fig.txt")
        EV.save_text_model(
            surface_info=surface_info,
            save_path=fig_path,
            binning=1
        )

        self.assertTrue(fig_path.is_file())


    def test_eval_single(self):
        self.assertEqual(len(self._single_out), 8)

        stats_1, stats_2, mean_1, mean_2, std_1, std_2, surface, binning = self._single_out
        self.assertIsInstance(stats_1, np.ndarray)
        self.assertIsInstance(stats_2, np.ndarray)
        self.assertIsInstance(mean_1, np.ndarray)
        self.assertIsInstance(mean_2, np.ndarray)
        self.assertIsInstance(std_1, np.ndarray)
        self.assertIsInstance(std_2, np.ndarray)
        self.assertIsInstance(binning, int)


    def test_eval_batch(self):
        os.chdir(self._anlys_folder)
        filelist = EV.find_files(self._data_folder)

        out = EV.eval_batch(
            filelist=filelist,
            params=self.params
        )
        self.assertEqual(len(out), 3)

        raw_df, analytics_df, show_df = out
        self.assertIsInstance(raw_df, pd.DataFrame)
        self.assertIsInstance(analytics_df, pd.DataFrame)
        self.assertIsInstance(show_df, pd.DataFrame)
        self.assertEqual(len(raw_df), 1)
        self.assertEqual(len(analytics_df), 1)
        self.assertEqual(len(show_df), 1)


    def test_get_lamella_orientations(self):
        ref_vertex, cell_vects = EV.get_lamella_orientations(self._my_lamella)

        self.assertEqual(ref_vertex.shape, (3,))
        self.assertEqual(cell_vects.shape, (3, 3))


    def test_get_intersection_mask(self):
        mask_out = EV.get_intersection_mask(
            tomo_shape = [100, 100, 100],
            lamella_obj = self._my_lamella
        )

        self.assertEqual(mask_out.shape, (100,)*3)
        self.assertAlmostEqual(mask_out.mean(), 0.51, places=4)


    @classmethod
    def tearDownClass(self):
        self.tmpdir.cleanup()


    def tearDown(self):
        pass
