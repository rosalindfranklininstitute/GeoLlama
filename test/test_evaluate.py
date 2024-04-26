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

        # Run single analysis to get output for subsequent tests
        os.chdir(self._anlys_folder)
        self._single_out = EV.eval_single(
            fname=self._data_address,
            pixel_size=1.0,
            binning=1,
            cpu=1,
            bandpass=False,
            autocontrast=True,
            adaptive=False,
        )


    def setUp(self):
        pass


    def test_find_files(self):
        filelist = EV.find_files(self._data_folder)

        # Tests
        self.assertIsInstance(filelist, list)
        self.assertEqual(len(filelist), 1)


    def test_save_figure(self):
        surface_info = self._single_out[-1]
        fig_path = Path(f"{self._models_folder}/test_fig.png")
        EV.save_figure(
            surface_info=surface_info,
            save_path=fig_path,
            binning=1
        )

        self.assertTrue(fig_path.is_file())


    def test_save_text_model(self):
        surface_info = self._single_out[-1]
        fig_path = Path(f"{self._models_folder}/test_fig.txt")
        EV.save_text_model(
            surface_info=surface_info,
            save_path=fig_path,
            binning=1
        )

        self.assertTrue(fig_path.is_file())


    def test_eval_single(self):
        self.assertEqual(len(self._single_out), 7)

        stats_1, stats_2, mean_1, mean_2, std_1, std_2, surface = self._single_out
        self.assertIsInstance(stats_1, np.ndarray)
        self.assertIsInstance(stats_2, np.ndarray)
        self.assertIsInstance(mean_1, np.ndarray)
        self.assertIsInstance(mean_2, np.ndarray)
        self.assertIsInstance(std_1, np.ndarray)
        self.assertIsInstance(std_2, np.ndarray)


    def test_eval_batch(self):
        os.chdir(self._anlys_folder)
        filelist = EV.find_files(self._data_folder)

        out = EV.eval_batch(
            filelist=filelist,
            pixel_size=1.0,
            binning=1,
            cpu=1,
            bandpass=False,
            autocontrast=True,
            adaptive=False,
        )
        self.assertEqual(len(out), 2)

        raw_df, show_df = out
        self.assertIsInstance(raw_df, pd.DataFrame)
        self.assertIsInstance(show_df, pd.DataFrame)
        self.assertEqual(len(raw_df), 1)
        self.assertEqual(len(show_df), 1)


    @classmethod
    def tearDownClass(self):
        self.tmpdir.cleanup()


    def tearDown(self):
        pass
