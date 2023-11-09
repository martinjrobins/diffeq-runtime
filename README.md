# diffeq-runtime

## Configure and build for WASM using Emscripten

### Install Emscripten SDK

~~~bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
git pull
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
~~~

### Configure and build OpenBLAS

```bash
cd build
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
git checkout v0.3.24
mkdir build
cd build
cmake -DCMAKE_LIBRARY_PATH=${PWD}/../../lib -DCMAKE_INSTALL_PREFIX=${PWD}/../.. -DBUILD_SHARED_LIBS=OFF -DTARGET=generic  -DC_LAPACK=ON -DNOFORTRAN=ON  ..
make
make install
```

### Configure and build SuiteSparse

~~~bash
cd build
git clone https://github.com/martinjrobins/SuiteSparse.git
cd SuiteSparse
git checkout i424-build-shared-libs 
mkdir build
cd build
cmake -DCMAKE_LIBRARY_PATH=${PWD}/../../lib -DCMAKE_INSTALL_PREFIX=${PWD}/../.. -DSUITESPARSE_ENABLE_PROJECTS=klu -DBUILD_SHARED_LIBS=OFF -DTARGET=generic -DNOPENMP=ON -DNPARTITION=ON ..
make
make install
~~~

### Configure project

~~~bash
cmake -DCMAKE_LIBRARY_PATH=${PWD}/lib -DCMAKE_INCLUDE_PATH=${PWD}/include -DKLU_LIBRARY_DIR=${PWD}/lib -DBUILD_SHARED_LIBS=OFF -DENABLE_KLU=ON -DTARGET=generic -DNOFORTRAN=1 -DNO_LAPACK=1 -DUSE_THREAD=0 ..
~~~

### Build project

~~~bash
cmake --build .
~~~