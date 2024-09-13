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

## Quickstarter
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


## Issues
Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/GeoLlama/issues) to submit bug reports or feature requests.

## License
Copyright Rosalind Franklin Institute 2024. Distributed under the terms of the Apache-2.0 license.