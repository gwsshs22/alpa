import glob
import os
import re
import shutil
import subprocess
import sys

from setuptools import setup, find_packages

IS_WINDOWS = sys.platform == "win32"
ROOT_DIR = os.path.dirname(__file__)


def get_cuda_version(cuda_home):
    """Locate the CUDA version."""
    version_file = os.path.join(cuda_home, "version.txt")
    try:
        if os.path.isfile(version_file):
            with open(version_file, "r") as f_version:
                version_str = f_version.readline().replace("\n", "").replace(
                    "\r", "")
                return version_str.split(" ")[2][:4]
        else:
            version_str = subprocess.check_output(
                [os.path.join(cuda_home, "bin", "nvcc"), "--version"])
            version_str = str(version_str).replace("\n", "").replace("\r", "")
            idx = version_str.find("release")
            return version_str[idx + len("release "):idx + len("release ") + 4]
    except RuntimeError:
        raise RuntimeError("Cannot read cuda version file")


def locate_cuda():
    """Locate the CUDA environment on the system."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        try:
            which = "where" if IS_WINDOWS else "which"
            nvcc = subprocess.check_output([which,
                                            "nvcc"]).decode().rstrip("\r\n")
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    version = get_cuda_version(cuda_home)
    cudaconfig = {
        "home":
            cuda_home,
        "include":
            os.path.join(cuda_home, "include"),
        "lib64":
            os.path.join(cuda_home,
                         os.path.join("lib", "x64") if IS_WINDOWS else "lib64"),
    }
    if not all([os.path.exists(v) for v in cudaconfig.values()]):
        raise EnvironmentError(
            "The CUDA  path could not be located in $PATH, $CUDA_HOME or $CUDA_PATH. "
            "Either add it to your path, or set $CUDA_HOME or $CUDA_PATH.")

    return cudaconfig, version


def get_cuda_version_str(no_dot=False):
    """Return the cuda version in the format of [x.x]."""
    ver = locate_cuda()[1]
    if no_dot:
        ver = ver.replace(".", "")
    return ver


install_require_list = [
    "tqdm",
    "ray>=1.13.0",
    "jax==0.3.15",
    "chex==0.1.4",
    "flax==0.5.2",
    "pulp>=2.6.0",
    "numpy>=1.20",
    "numba",
]

dev_require_list = [
    f"cupy-cuda{get_cuda_version_str(no_dot=True)}", "yapf==0.32.0",
    "pylint==2.14.0", "cmake", "pybind11"
]

doc_require_list = [
    "sphinx", "sphinx-rtd-theme", "sphinx-gallery", "matplotlib"
]


def get_alpa_version():
    with open(os.path.join(ROOT_DIR, "alpa", "version.py")) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def build():
    """Build the custom pipeline marker API."""
    # Check cuda version
    build_command = []
    if "CUDACXX" in os.environ and os.path.exists(os.environ["CUDACXX"]):
        cudacxx_path = os.environ["CUDACXX"]
    else:
        # infer CUDACXX
        cuda_version = get_cuda_version_str()
        cudacxx_path = f"/usr/local/cuda-{cuda_version}/bin/nvcc"
        if not os.path.exists(cudacxx_path):
            raise ValueError("Cannot find CUDACXX compiler.")

    # Enter the folder and build
    build_command += [f"cd alpa/pipeline_parallel/xla_custom_call_marker; "]
    build_command += [f"CUDACXX={cudacxx_path} ./build.sh"]
    build_command = " ".join(build_command)
    print(build_command)
    ret = subprocess.call(build_command, shell=True)
    if ret != 0:
        raise RuntimeError("Failed to build the pipeline markers "
                           f"with exit code {ret}")


def move_file(target_dir, filename):
    source = filename
    destination = os.path.join(
        target_dir, "alpa/pipeline_parallel/xla_custom_call_marker/build",
        filename.split('/')[-1])
    # Create the target directory if it doesn't already exist.
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print("Copying {} to {}.".format(source, destination))
        if IS_WINDOWS:
            # Does not preserve file mode (needed to avoid read-only bit)
            shutil.copyfile(source, destination, follow_symlinks=True)
        else:
            # Preserves file mode (needed to copy executable bit)
            shutil.copy(source, destination, follow_symlinks=True)


def build_and_move(build_ext):
    build()
    files_to_include = glob.glob(
        "alpa/pipeline_parallel/xla_custom_call_marker/build/*.so")
    for filename in files_to_include:
        move_file(build_ext.build_lib, filename)


if __name__ == "__main__":
    import setuptools
    import setuptools.command.build_ext
    from setuptools.command.install import install

    class build_ext(setuptools.command.build_ext.build_ext):

        def run(self):
            return build_and_move(self)

    class BinaryDistribution(setuptools.Distribution):

        def has_ext_modules(self):
            return True

    class InstallPlatlib(install):

        def finalize_options(self):
            install.finalize_options(self)
            if self.distribution.has_ext_modules():
                self.install_lib = self.install_platlib

    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="alpa",
        version=get_alpa_version(),
        author="Alpa Developers",
        author_email="",
        description=
        "Alpa automatically parallelizes large tensor computation graphs and "
        "runs them on a distributed cluster.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/alpa-projects/alpa",
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords=("alpa distributed parallel machine-learning model-parallelism"
                  "gpt-3 deep-learning language-model python"),
        packages=find_packages(exclude=["playground"]),
        python_requires='>=3.7',
        cmdclass={
            "build_ext": build_ext,
            "install": InstallPlatlib
        },
        distclass=BinaryDistribution,
        install_requires=install_require_list,
        extras_require={
            'dev': dev_require_list,
            'doc': doc_require_list + dev_require_list,
        },
    )
