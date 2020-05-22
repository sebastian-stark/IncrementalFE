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
