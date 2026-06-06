# mamba2_for_windows

这个目录整理了在 Windows 上安装 `mamba-ssm` / Mamba2 所需的本地包与验证脚本，适合在不能直接使用 Linux 版 Triton 和原版源码的 Windows CUDA 环境中复现安装。

本说明主要参考：

- [win10系统完美配置mamba-ssm全整合方案](https://divertingpan.github.io/post/win-mamba/)
- [scott-yjyang/Vivim](https://github.com/scott-yjyang/Vivim)

## 目录结构

```text
mamba2_for_windows/
├── triton-3.1.0/
│   └── triton-3.1.0-cp310-cp310-win_amd64.whl
├── causal-conv1d-1.4.0/
│   ├── setup.py
│   └── dist/causal_conv1d-1.4.0-py3.10-win-amd64.egg
├── mamba-2.2.2/
│   ├── setup.py
│   └── dist/mamba_ssm-2.2.2-py3.10-win-amd64.egg
└── test_scripts/
    ├── triton_test.py
    └── mamba_test.py
```

## 推荐环境

参考教程中的可用组合如下：

| 项目 | 推荐版本 |
| --- | --- |
| 系统 | Windows 10 x64；Windows 11 通常同理，但原教程以 Windows 10 实测为准 |
| Python | 3.10 x64 |
| GPU | NVIDIA GPU，建议 Pascal / GTX 10 系及以上，计算能力不低于 6.1 |
| CUDA | 12.4 |
| PyTorch | 2.4.1 + cu124 |
| Triton | 3.1.0 Windows wheel，本目录已提供 |
| causal-conv1d | 1.4.0 |
| mamba-ssm | 2.2.2 |

注意：`triton-3.1.0-cp310-cp310-win_amd64.whl` 只适配 Python 3.10。不要用 Python 3.11/3.12 安装这个 wheel。

## 安装前准备

1. 安装 Anaconda 或 Miniconda。
2. 安装 NVIDIA 显卡驱动，并确保 `nvidia-smi` 可用。
3. 安装 CUDA Toolkit 12.4.0。
4. 安装 Visual Studio Build Tools 2022：
   - `MSVC v143 - VS 2022 C++ x64/x86 build tools`
   - `Windows 10 SDK`，例如 `10.0.20348.0` 或更高版本
5. 建议安装或修复 `vc_redist.x64.exe`。

编译时推荐使用“x64 Native Tools Command Prompt for VS 2022”启动命令行，这样 `cl`、`rc.exe`、`INCLUDE`、`LIB` 等路径通常会自动配置好。

如果不用 VS 的 Native Tools 命令行，需要手动检查：

```bat
cl
nvcc --version
```

`cl` 能输出 MSVC 编译器信息，`nvcc --version` 能看到 CUDA 12.4，才继续后续步骤。

如果编译时报找不到 `rc.exe` 或 `rcdll.dll`，可把 Windows Kits 目录中的这两个文件复制到 MSVC 编译器目录。路径中的版本号请按本机实际情况调整：

```bat
copy "C:\Program Files (x86)\Windows Kits\10\bin\10.0.20348.0\x64\rc.exe" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64\"
copy "C:\Program Files (x86)\Windows Kits\10\bin\10.0.20348.0\x64\rcdll.dll" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64\"
```

## 安装步骤

以下命令建议在 CMD 或 “x64 Native Tools Command Prompt for VS 2022” 中执行。

### 1. 创建环境

```bat
conda create -n mamba2 python=3.10 -y
conda activate mamba2

python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

conda install nvidia/label/cuda-12.4.0::cuda-nvcc -y
conda install nvidia/label/cuda-12.4.0::cuda-cccl -y

pip install ninja packaging setuptools==68.2.2 einops transformers
```

如已有错误安装过 Linux 版 Triton 或其他版本的 Mamba 依赖，建议先清理：

```bat
pip uninstall -y triton causal-conv1d mamba-ssm
```

### 2. 进入本目录

CMD：

```bat
cd /d "D:\PycharmProjects\mamba2_for_windows"
```

PowerShell：

```powershell
Set-Location -LiteralPath "D:\PycharmProjects\mamba2_for_windows"
```

### 3. 安装 Windows 版 Triton

```bat
pip install .\triton-3.1.0\triton-3.1.0-cp310-cp310-win_amd64.whl
```

验证 Triton：

```bat
python .\test_scripts\triton_test.py
```

正常情况下会输出 PyTorch、Triton、CUDA、GPU 信息，并以 `Triton CUDA test OK` 结束。

### 4. 编译安装 causal-conv1d

CMD：

```bat
cd /d "D:\PycharmProjects\mamba2_for_windows\causal-conv1d-1.4.0"
set CAUSAL_CONV1D_FORCE_BUILD=TRUE
python setup.py install
```

PowerShell：

```powershell
Set-Location -LiteralPath "D:\PycharmProjects\mamba2_for_windows\causal-conv1d-1.4.0"
$env:CAUSAL_CONV1D_FORCE_BUILD = "TRUE"
python setup.py install
```

安装完成后检查：

```bat
pip show causal-conv1d
```

版本应为 `1.4.0`。

### 5. 编译安装 mamba-ssm

CMD：

```bat
cd /d "D:\PycharmProjects\mamba2_for_windows\mamba-2.2.2"
set MAMBA_FORCE_BUILD=TRUE
python setup.py install
```

PowerShell：

```powershell
Set-Location -LiteralPath "D:\PycharmProjects\mamba2_for_windows\mamba-2.2.2"
$env:MAMBA_FORCE_BUILD = "TRUE"
python setup.py install
```

安装完成后检查：

```bat
pip show mamba-ssm
```

版本应为 `2.2.2`。

### 6. 最终验证

回到本目录：

```bat
cd /d "D:\PycharmProjects\mamba2_for_windows"
python .\test_scripts\mamba_test.py
```

正常输出应包含：

```text
causal_conv1d import OK
mamba_ssm import OK
Mamba CUDA forward test OK
```

## 在 PyCharm 中使用

1. 打开项目后进入 `Settings` -> `Project` -> `Python Interpreter`。
2. 选择刚创建的 conda 环境，例如 `mamba2`。
3. 在 PyCharm Terminal 中确认：

```bat
python -c "import torch, triton, causal_conv1d, mamba_ssm; print(torch.__version__); print(triton.__version__)"
```

如果终端能正常导入，项目代码也应能使用 `mamba_ssm`。

## 常见问题

### 误装了原版 Triton

Windows 下不要直接安装 Linux 原版 Triton。先卸载再安装本目录 wheel：

```bat
pip uninstall -y triton
pip install .\triton-3.1.0\triton-3.1.0-cp310-cp310-win_amd64.whl
```

### `cl` 不是内部或外部命令

说明 MSVC 编译环境没有生效。请使用 “x64 Native Tools Command Prompt for VS 2022”，或检查 Visual Studio Build Tools、`Path`、`INCLUDE`、`LIB` 环境变量。

### 找不到 `rc.exe` 或 `rcdll.dll`

按“安装前准备”中的方法，把 Windows Kits 目录里的 `rc.exe`、`rcdll.dll` 复制到 MSVC 的 `bin\Hostx64\x64` 目录。

### `No .egg-info directory found`

本目录建议使用：

```bat
python setup.py install
```

不要优先使用 `pip install .`。

### `causal_conv1d_cuda` 或 `selective_scan_cuda` 导入失败

通常是编译阶段没有成功生成 `.pyd`。重点检查：

- 是否在 Python 3.10 环境中安装；
- PyTorch 是否是 `2.4.1+cu124`；
- `nvcc --version` 是否为 CUDA 12.4；
- 是否在 x64 MSVC 编译环境中执行；
- 是否设置了 `CAUSAL_CONV1D_FORCE_BUILD=TRUE` 和 `MAMBA_FORCE_BUILD=TRUE` 后重新安装。

### 第一次运行脚本时生成编译缓存

首次运行 Triton/Mamba 相关 CUDA 代码时，可能会生成一些编译缓存文件，这是正常现象。

## 许可说明

本目录包含的第三方项目及源码以各子目录中的 `LICENSE` 文件为准。
