// --------------------------------------------------------------------------
// Copyright (C) 2020 by Sebastian Stark
//
// This file is part of the IncrementalFE library
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

The IncrementalFE library builds on the GalerkinTools library and adds some functionality related to the solution of transient problems by one step time integration algorithms. In particular, it includes a Newton-Raphson based solver for the non-linear problems arising in each time step.

Currently, restriction is made to problems, which involve only first order time derivatives.

The library requires the library GalerkinTools (https://github.com/starki0815/GalerkinTools) to be installed on your system.

Installation of the IncrementalFE library is through cmake:

(1) place library source files into some folder /path/to/folder/IncrementalFE (you can use git clone https://github.com/starki0815/IncrementalFE.git for this)
(2) cd /path/to/folder/
(3) mkdir build
(4) cd build
(5) cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir ../IncrementalFE
(6) make install
(7) optionally set an environment variable to INCREMENTAL_FE_DIR=/path/to/install/dir
(8) optionally run the tests (first cd /path/to/folder/build, then ctest)

Acknowledgements:
The IncrementalFE library has been developed during a project supported by the Deutsche Forschungsgemeinschaft (DFG) under Grants STA 1593/1-1 and STA 1593/2-1.
