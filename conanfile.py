from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class uxto_set_experimentsRecipe(ConanFile):
    name = "uxto-set-experiments"
    version = "1.0"
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = "CMakeLists.txt", "src/*"

    def requirements(self):
        self.requires("node/0.57.0", transitive_headers=True, transitive_libs=True)
        self.requires("fmt/11.1.3", transitive_headers=True, transitive_libs=True)
        self.requires("abseil/20250127.0", transitive_headers=True, transitive_libs=True)
        self.requires("robin-hood-hashing/3.11.5", transitive_headers=True, transitive_libs=True)
        self.requires("unordered_dense/4.5.0", transitive_headers=True, transitive_libs=True)
        self.requires("svector/1.0.3", transitive_headers=True, transitive_libs=True)
        self.requires("boost/1.86.0", transitive_headers=True, transitive_libs=True)

    def configure(self):
        # self.options["node/*"].march_id = "ZLm9Pjh"
        self.options["node/*"].march_strategy = "optimized"        
        self.options["fmt/*"].header_only = True

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()




