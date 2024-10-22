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

For native you can use cmake:

```bash
cd build
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
git checkout v0.3.24
cd build
cmake -DCMAKE_INSTALL_PREFIX=${PWD}/../.. ..
```

For emscripten use make:

```bash
cd build
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
git checkout v0.3.24
emmake make libs shared CC=emcc HOSTCC=gcc TARGET=RISCV64_GENERIC NOFORTRAN=1 C_LAPACK=1 USE_THREAD=0 NO_SHARED=1 PREFIX=${EMSDK}/upstream/emscripten/cache/sysroot 
emmake make install libs shared CC=emcc HOSTCC=gcc TARGET=RISCV64_GENERIC NOFORTRAN=1 C_LAPACK=1 USE_THREAD=0 NO_SHARED=1 PREFIX=${EMSDK}/upstream/emscripten/cache/sysroot 
```

### Configure and build SuiteSparse

~~~bash
cd build
git clone https://github.com/martinjrobins/SuiteSparse.git
cd SuiteSparse
git checkout i424-build-shared-libs 
mkdir build
cd build
cmake -DSUITESPARSE_ENABLE_PROJECTS=klu -DBUILD_SHARED_LIBS=OFF -DTARGET=generic -DNOPENMP=ON -DNPARTITION=ON -DNFORTRAN=ON -DBLA_STATIC=ON ..
make
make install
~~~

Note: for emscripten use `emcmake cmake ...`


### Configure project

~~~bash
cmake -DKLU_LIBRARY_DIR=${EMSDK}/upstream/emscripten/cache/sysroot/lib -DKLU_INCLUDE_DIR=${EMSDK}/upstream/emscripten/cache/sysroot/include -DBUILD_SHARED_LIBS=OFF -DENABLE_KLU=ON -DTARGET=generic -DNOFORTRAN=1 -DNO_LAPACK=1 -DUSE_THREAD=0 -DSUNDIALS_INDEX_SIZE=32 ..
~~~

### Build project

~~~bash
cmake --build .
~~~