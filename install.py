import os
import importlib.util
import sys
import subprocess
import sys
from importlib import import_module
# python3.10 hack for attrdict
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
    
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

try:
    from packaging import version
except:
    install_package("packaging")

def check_and_install_version(package_name, required_version, up_version=True, import_name=None):
    if import_name is  None:
        import_name = package_name
    try:
        # 尝试导入包以检查是否已安装
        package = import_module(import_name)
        installed_version = package.__version__
        if up_version and version.parse(installed_version) >= version.parse(required_version):
            print(f"{package_name}的当前版本{installed_version}满足要求，无需安装{required_version}版本。")
            return
        elif installed_version == required_version:
            print(f"{package_name}的当前版本{installed_version}满足要求，无需安装{required_version}版本。")
            return
        else:
            print(f"{package_name}的当前版本{installed_version}低于要求的{required_version}，将尝试安装。")
    except ImportError:
        print(f"{package_name}未安装，将尝试安装{required_version}版本。")
    except AttributeError:
        print(f"无法确定{package_name}的版本，将尝试安装{required_version}版本。")

    # 安装或更新至指定版本
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}=={required_version}'])
        print(f"{package_name}已安装/更新至版本{required_version}。")
    except subprocess.CalledProcessError as e:
        print(f"安装{package_name}时出错：{e}")



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