[![codecov](https://codecov.io/gh/rosalindfranklininstitute/GeoLlama/graph/badge.svg?token=I0JZ7YUYI4)](https://codecov.io/gh/rosalindfranklininstitute/GeoLlama)

## Installation
You can install the package by running the following command:
```
pip3 install .
```

## Testing
Package tests can be run by using the following commands if python-pytest is installed:
```
pip3 install .[test]
pytest
```

If pytest is not available, you can also run tests from the package root folder by using the command:
```
python3 -m unittest discover
```

## Quickstarter -- running GeoLlama
To use GeoLlama, first create a YAML configuration file by running the command
```
geollama generate-config [--output-path PATH]
```
A YAML configuration file will be generated with default parameters. If the `--output-path` parameter is not provided, the file will be generated at `./config.yaml` by default.

Once the parameters are set, GeoLlama calculations can be started using the command
```
geollama main -i [config]
```
where `[config]` is the path to the YAML configuration file generated and modified.

Alternatively, should the user wish to provide the parameters on the command line and run GeoLlama YAML-free, flags corresponding to the parameters in the YAML file can be obtained through
```
geollama main --help
```

GeoLlama defaults any error reporting (traceback) to a simplified format. Should the user wish to access more detailed error logs, developer mode can be activated via
```
geollama -d main -i [config]
```
**Please note that the `-d` flag must be added before `main`.**

## Model outputs and their usage
By default, GeoLlama creates a subfolder in the current working directory (`./surface_models/`) with estimated lamella surface models (in .txt plaintext format) and their 3D representation in PNG format.

Moreover, in the same subfolder (`./surface_model/`), GeoLlama creates a bash script named `p2m_convert.sh` which the user can use the command
```
source p2m_convert.sh
```
to batch convert the generated plain text models to corresponding IMOD binary models using IMOD's `point2model` utility. **Note that IMOD utilities must be accessible in the user's PATH variable in order for this script to work.**

## Reporting feature
A STAR file that contains GeoLlama job specification and statistical results from GeoLlama evaluation of tomograms can be optionally created. The STAR file is useful in producing a GeoLlama report, presenting to the user the results in a more formatted and more human-readable fashion. The automatic production of the STAR file and the report can be enabled by setting appropriate variables in the configuration file or flags on the command-line. **Note that STAR file output must be enabled for automatic generation of GeoLlama report.**

The GeoLlama report can also be generated separately using a pre-exising _valid_ STAR file that follows the GeoLlama format (i.e. with appropriate data blocks and columns). The command is
```
geollama generate-report [star_path] [--report_path PATH] [--no-html]
```
where the arguments and flags are as follow:
- `star_path`: path to the input STAR file (_MUST BE PROVIDED_)
- `--report_path PATH`: path to the compiled report _Jupyter Notebook_ (default path:`./GeoLlama_report.ipynb`)
- `--no-html`: compile report as Jupyter Notebook only, without further conversion into HTML format



## Issues
Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/GeoLlama/issues) to submit bug reports or feature requests.

## License
Copyright Rosalind Franklin Institute 2024. Distributed under the terms of the Apache-2.0 license.