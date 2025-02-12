# MetaShape
Metashape is a design tool developed for non-imaging metasurfaces. It calculates the phase profile of a metasurface (in 1D or 2D) that produces a specific target pattern at a known distance, given the intensity distribution of the incident beam. When used in combination with the [diffractsim](https://github.com/rafael-fuente/diffractsim) repository by rafael-fuente, one can simulate the resulting beam shaping performed by the newly designed metasurface.

## Compatibility
This software was written and tested on Windows 11 Enterprise (Version 23H2 OS build 22631.4602).

The current version requires all input to be on the form of an analytical expression. Any experimental data thus requires fitting prior to using MetaShape.

## Installation
The repository can be downloaded by cloning
```bash
git clone https://github.com/kespoul/MetaShape.git
```

This folder takes up 140 kB on your disk and should take less than 1 minute to download on standard hardware and internet.

The entire code is open-source in accordance with the license and runs on Python with the external libraries.

The software package has been written using Python 3.11.5 and Spyder 5.4.3.

The libraries necessary for running the software is as follows:

| Package:      | Version:      |
| ------------- | ------------- |
| matplotlib    | 3.7.2         |
| NumPy         | 1.24.3        |
| SciPy         | 1.11.1        |
| SymPy         | 1.11.1        |
| diffractsim   | 2.2.3         |

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install relevant packages.

```bash
pip install matplotlib
pip install numpy
pip install scipy
pip install sympy
pip install diffractsim
pip install pyinstaller
```

## Examples
Examples can be found in the "Examples" subdirectory. Make sure to have the "MetaShape" folder as your active directory before running the examples.

## License
BSD 3-clause (see file)
