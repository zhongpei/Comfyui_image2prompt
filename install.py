import os
import importlib.util
import sys
import subprocess
import sys


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

try:
    from packaging import version
except:
    install_package("packaging")

def check_and_install_version(package_name, required_version, up_version=True):
    try:
        # 检查包是否已安装及其版本
        result = subprocess.run(['pip', 'show', package_name], stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # 解析已安装的版本
            installed_version = None
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    installed_version = line.split(':')[1].strip()
                    break

            if installed_version == required_version:
                print(f"{package_name}已安装，且版本为{required_version}。")
                return
            elif up_version and installed_version and version.parse(installed_version) >= version.parse(required_version):
                print(f"{package_name}的当前版本{installed_version}满足要求，无需安装{required_version}版本。")
                return
            else:
                print(f"{package_name}已安装，但版本{installed_version}不符合要求的{required_version}，将尝试安装正确版本。")
        else:
            print(f"{package_name}未安装，将尝试安装。")

        # 安装或更新至指定版本
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}=={required_version}'])
        print(f"{package_name}已更新至版本{required_version}。")
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