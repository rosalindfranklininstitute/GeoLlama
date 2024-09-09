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
## Module             : GeoLlama.config  ##
## Created            : Neville Yee      ##
## Date created       : 03-May-2024      ##
## Date last modified : 03-May-2024      ##
###########################################

from dataclasses import dataclass
import typing


@dataclass()
class Config:
    data_path: typing.Optional[str] = None
    pixel_size_nm: typing.Optional[float] = None
    binning: typing.Optional[int] = None
    autocontrast: typing.Optional[bool] = None
    adaptive: typing.Optional[bool] = None
    bandpass: typing.Optional[bool] = None
    num_cores: typing.Optional[int] = None
    output_csv_path: typing.Optional[str] = None
    output_star_path: typing.Optional[str] = None
    output_mask: typing.Optional[bool] = None
    thickness_lower_limit: typing.Optional[float] = None
    thickness_upper_limit: typing.Optional[float] = None
    thickness_std_limit: typing.Optional[float] = None
    xtilt_std_limit: typing.Optional[float] = None
    displacement_limit : typing.Optional[float] = None
    displacement_std_limit : typing.Optional[float] = None
