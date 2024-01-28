import os
import importlib.util
import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

def check_and_install(package, import_name=""):
    if import_name == "":
        import_name = package
    try:
        importlib.import_module(import_name)
        print(f"{import_name} is already installed.")
    except ImportError:
        print(f"Installing {import_name}...")
        install_package(package)
        
def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir