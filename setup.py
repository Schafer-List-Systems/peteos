from setuptools import setup, Extension, find_packages, Command
import tomllib
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize
import glob
import os
import re

# Discover all .py files under peteos/ -- these become compiled extensions
py_files = glob.glob("peteos/**/*.py", recursive=True)

# Collect module extensions and their .py sources
extensions = []
for py_path in py_files:
    name = py_path.replace("/", ".").replace(".py", "")
    extensions.append(Extension(name, sources=[py_path]))

class clean(Command):
    """Custom clean: remove .c, .so, build/, dist/, *.egg-info/."""
    _DESC = re.compile(r"\.cpython-\d+")
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove Cython-generated .c files from peteos/
        for root, dirs, files in os.walk("peteos"):
            for f in files:
                if f.endswith(".c"):
                    os.remove(os.path.join(root, f))
        # Remove compiled .so/.pyd files from peteos/
        for root, dirs, files in os.walk("peteos"):
            for f in files:
                if self._DESC.search(f):
                    os.remove(os.path.join(root, f))
        # Remove build/, dist/, egg-info
        import shutil
        for d in ("build", "dist"):
            if os.path.isdir(d):
                shutil.rmtree(d)
        for d in os.listdir("."):
            if d.endswith(".egg-info"):
                shutil.rmtree(d)

class make_dist(Command):
    """Build a distribution archive: wheel + README + docs."""
    description = "Create a peteos-X.X.X.tar.gz distribution archive"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import shutil
        import tarfile
        import tomllib

        # 1. Read version from pyproject.toml
        with open("pyproject.toml", "rb") as f:
            meta = tomllib.load(f)
        version = meta["project"]["version"]
        name = meta["project"]["name"]

        # 2. Build the wheel
        print("=== Building wheel ===")
        self.run_command("bdist_wheel")

        # 3. Find the wheel
        dist_dir = "dist"
        wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
        if not wheels:
            raise RuntimeError("No wheel found in dist/")
        wheel_path = os.path.join(dist_dir, wheels[0])
        print(f"=== Wheel: {wheel_path} ===")

        # 4. Create tar.gz archive
        archive_name = f"{name}-{version}.tar.gz"
        print(f"=== Creating {archive_name} ===")
        with tarfile.open(archive_name, "w:gz") as tar:
            # Add the wheel
            tar.add(wheel_path, arcname=f"{name}-{version}/{wheels[0]}")

            # Add README and LICENSE if they exist
            for fname in ("README.md", "LICENSE", "LICENSE.txt", "peteos.json.example"):
                if os.path.exists(fname):
                    tar.add(fname, arcname=f"{name}-{version}/{fname}")

            # Add docs directory
            if os.path.isdir("docs"):
                for root, dirs, files in os.walk("docs"):
                    for f in files:
                        src = os.path.join(root, f)
                        arcname = src.replace("docs", f"{name}-{version}/docs")
                        tar.add(src, arcname=arcname)

            # Add pyproject.toml and setup.py for reference
            for fname in ("pyproject.toml", "setup.py"):
                if os.path.exists(fname):
                    tar.add(fname, arcname=f"{name}-{version}/{fname}")

            # Add dependencies and extras
            for fname in ("requirements.txt", "requirements-dev.txt"):
                if os.path.exists(fname):
                    tar.add(fname, arcname=f"{name}-{version}/{fname}")

        print(f"=== Done: {archive_name} ===")

_PY_RE = re.compile(r"^peteos/.*\.py$")

class build_py(_build_py):
    """Skip .py files that are compiled to extensions."""
    def get_source_files(self):
        src = super().get_source_files()
        return [f for f in src if not _PY_RE.search(f)]

class build_ext(_build_ext):
    """Compile extensions, then remove .c and .py files from the build dir."""
    def run(self):
        super().run()
        build_dir = self.build_lib
        if not os.path.isdir(build_dir):
            return
        for root, dirs, files in os.walk(build_dir):
            for f in files:
                if f.endswith((".c", ".py")):
                    os.remove(os.path.join(root, f))

setup(
    name="peteos",
    packages=find_packages(include=["peteos", "peteos.*"]),
    ext_modules=cythonize(
        extensions,
        language_level="3str",
        compiler_directives={"annotation_typing": False},
    ),
    zip_safe=False,
    cmdclass={
        "build_py": build_py,
        "build_ext": build_ext,
        "clean": clean,
        "make_dist": make_dist,
    },
)
