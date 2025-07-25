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
## Date last modified : 03-May-2024        ##
#############################################


import os
import tempfile
import platform
from pathlib import Path
import shutil
import unittest
import unittest.mock as mock
import multiprocessing as mp

import numpy as np
import pandas as pd
import starfile

from geollama import main, config, evaluate


class MainTest(unittest.TestCase):

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

        # Create folders
        self.tmpdir.mkdir(exist_ok=True)
        self.tmp_data.mkdir(exist_ok=True)
        self.tmp_anlys.mkdir(exist_ok=True)

    def setUp(self):
        pass

    def test_generate_config(self):
        os.chdir(self.tmp_anlys)

        with mock.patch.object(config, "generate_config") as m:
            main.generate_config()
            self.assertTrue(m.called)

    def test_main(self):
        os.chdir(self.tmp_anlys)

        # Create random dataframe as mock output from eval_batch
        raw_df = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 9)),
            columns=[
                "filename",
                "Mean_thickness_nm",
                "Thickness_SEM_nm",
                "Mean_X-tilt_degs",
                "X-tilt_SEM_degs",
                "Mean_Y-tilt_degs",
                "Y-tilt_SEM_degs",
                "Mean_drift_perc",
                "Drift_SEM_perc",
            ],
        )

        analytics_df = pd.DataFrame(
            np.random.choice(a=[True, False], size=(10, 9), p=[0.5, 0.5]),
            columns=[
                "filename",
                "Anom_too_thin",
                "Anom_too_thick",
                "Anom_thick_uncertain",
                "Anom_xtilt_out_of_range",
                "Anom_xtilt_uncertain",
                "Anom_centroid_displaced",
                "Anom_wild_drift",
                "Num_possible_anomalies",
            ],
        )
        analytics_df["Num_possible_anomalies"] = np.random.randint(7, size=(10,))

        show_df = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 6)),
            columns=[
                "filename",
                "Thickness (nm)",
                "X-tilt (degs)",
                "Y-tilt (degs)",
                "Centroid drift (%)",
                "Num_possible_anomalies",
            ],
        )
        adaptive_count = 0

        with mock.patch.object(
            evaluate,
            "eval_batch",
            return_value=(raw_df, analytics_df, show_df, adaptive_count),
        ) as m:
            main.main(
                data_path=self.tmp_data,
                pixel_size_nm=1,
                output_csv_path=Path(self.tmp_anlys, "./test.csv"),
                output_star_path=Path(self.tmp_anlys, "./test.star"),
                report=False,
                printout=False,
            )

            # Test if files are created
            self.assertTrue(os.path.exists(Path(self.tmp_anlys, "./test.csv")))
            self.assertTrue(os.path.exists(Path(self.tmp_anlys, "./test.star")))

    @classmethod
    def tearDownClass(self):
        os.chdir(self.orig_path)
        shutil.rmtree(self.tmp_data, ignore_errors=True)
        shutil.rmtree(self.tmp_anlys, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tearDown(self):
        pass
