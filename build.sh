# rm -rf build
# rm -rf conan.lock

# conan lock create conanfile.py --version="1.0" --update
# conan lock create conanfile.py --version "1.0" --lockfile=conan.lock --lockfile-out=build/conan.lock
conan install conanfile.py --lockfile=build/conan.lock -of build --build=missing

cmake --preset conan-release \
         -DCMAKE_VERBOSE_MAKEFILE=ON \
         -DGLOBAL_BUILD=ON \
         -DCMAKE_BUILD_TYPE=Release

cmake --build --preset conan-release -j4 \
         -DCMAKE_VERBOSE_MAKEFILE=ON \
         -DGLOBAL_BUILD=ON \
         -DCMAKE_BUILD_TYPE=Release
