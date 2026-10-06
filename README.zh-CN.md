# VanillaSort 🐾

[English](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/README.md) | **简体中文**

**📦 Python package · SpikeInterface API · 命令行分选**

VanillaSort 结合 **VanillaDet** 尖峰检测、**HuiduRep** 波形表征与 **VanillaCluster** 聚类。通过 Python API 或 `vanillasort` 命令处理细胞外记录，获得尖峰时间和候选神经元单元归属。

预训练流程处理 **30 kHz 采样的四通道记录**。SpikeInterface API 还为较大的二维探针提供实验性的四近邻适配。论文代码与方法开发说明见 [main 分支](https://github.com/IgarashiAkatuki/VanillaSort/tree/main)。

**📄 论文：** Zishuo Feng 与 Feng Cao，[*Spike Sorting with VanillaSort*](https://www.biorxiv.org/content/10.64898/2026.09.18.752552)，bioRxiv，2026。**使用 VanillaSort 必须引用该论文。** [BibTeX](#引用)

<p align="center">
  <a href="https://raw.githubusercontent.com/IgarashiAkatuki/VanillaSort/feat/inference-package/assets/vanilla.jpg"><img src="https://raw.githubusercontent.com/IgarashiAkatuki/VanillaSort/feat/inference-package/assets/vanilla.jpg" alt="我们的缅因猫 Vanilla" width="320"></a>
  <br>
  <em>这是 Vanilla，我们可爱的缅因猫！🤍</em>
</p>

[📦 安装](#安装) · [🚀 快速开始](#快速开始) · [🐍 Python API](#python-api) · [💻 命令行](#命令行用法) · [📁 输出](#输出结果) · [🧠 架构](#模型架构) · [📊 结果](#实验结果)

## 安装

环境要求：**Python 3.10+**、**PyTorch 2.7.0+**；GPU 运行使用 **CUDA 12.8+** 及兼容的 NVIDIA 驱动。**在兼容的情况下，请优先使用最新稳定版本。**

创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

使用 GPU 时，在 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)中选择 **Stable**，并选择与环境兼容的最新 CUDA 版本，最低为 **12.8**。将对应的 wheel 索引用于安装命令。下表包含 CUDA 12.8 的安装示例：

| 平台 | 命令 |
| --- | --- |
| Linux / Windows，CPU | `python -m pip install --upgrade "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu` |
| Linux / Windows，NVIDIA GPU，CUDA 12.8 示例 | `python -m pip install --upgrade "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128` |
| macOS，Apple 芯片，CPU | `python -m pip install --upgrade "torch>=2.7.0"` |

从 PyPI 安装 VanillaSort：

```bash
python -m pip install --upgrade vanillasort
```

安装时会自动安装运行依赖。两个模型权重合计约 36 MiB，首次使用时从 [Hugging Face](https://huggingface.co/Kohaku2580/VanillaSort) 下载并缓存。以下命令可在任意目录运行。

## 快速开始

用内置的两秒合成记录运行完整流程，混合模型的组分数设为 4：

```bash
vanillasort --self-test --device cpu --output output/self-test
```

该命令检查检测、波形编码、聚类和结果导出。成功后会打印 `Done:`，并在 `output/self-test/` 中生成 `events.npz` 和 `run.json`。

处理自己的记录时，`--device auto` 会优先使用 CUDA，否则使用 CPU。每次运行请选择新建或空的输出目录。

## Python API

将带有通道坐标的 SpikeInterface `BaseRecording` 传入 `vanillasort.sort()`：

```python
import vanillasort
from spikeinterface.core import load

# recording 是你的 SpikeInterface BaseRecording。
sorting = vanillasort.sort(
    recording,
    components=22,  # 按当前记录选择 GMM 组分数
    output_folder="output/recording",
    device="auto",
    seed=0,
)

for unit_id in sorting.unit_ids:
    samples = sorting.get_unit_spike_train(unit_id, segment_index=0)
    print(unit_id, len(samples))

restored = load("output/recording/sorting")
```

返回值是 SpikeInterface `BaseSorting`。尖峰序列使用各 segment 内从 0 开始的采样点索引；`main_channel_id` 记录每个单元对应的主要记录通道。

**输入要求：** 原始记录采样率为 30,000 Hz（容差 ±1 Hz），每个 segment 至少 100 个采样点，通道坐标为以微米表示的有限二维坐标。每个通道组或 shank 至少包含四个触点。有通道 gain/offset 时会自动换算，否则各通道使用共同电压尺度。程序内部完成滤波。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `components` | `22` | GMM 组分数 K；根据记录选择 K ≥ 2。 |
| `output_folder` | `None` | 将结果保留在内存，或保存到新建/空目录。 |
| `device` | `"auto"` | 优先使用 CUDA，否则使用 CPU；也可指定 `"cpu"` 或 `"cuda:0"`。 |
| `seed` | `0` | 可复现的初始化；CLI 默认值为 `30`。 |
| `profile` | `"d1"` | 选择 `"d1"` 或 `"canonical"` 预处理。 |
| `model`、`model_path` | `"default"`、`None` | 使用预训练模型或本地权重。 |
| `detector_batch`、`embedding_batch` | `8`、`128` | 推理批大小。 |
| `verbose` | `True` | 打印进度。 |

K 分别用于每个邻域和 segment。非空邻域需要至少 K 个可分选事件；检测结果为空时返回空 sorting。

**较大探针：** 实验性适配器构建四近邻，遵守 group/shank 边界，并在 12 个采样点范围内对重叠检测去重。各 segment 使用独立单元 ID。漂移校正、跨邻域和跨 segment 单元匹配仍待实现；漂移单元可能被拆分，相近的同时事件可能被抑制。

**内存：** 每次以最多 300,000 个采样点为一块读取一个四通道邻域。精确滤波和中位数/MAD 归一化需要完整 segment 驻留内存，运行前会检查可用内存。波形存储随事件数增长；CUDA 显存不足时会缩小推理批次并重试。

## 命令行用法

### 准备输入

将数据保存为 NumPy `.npz` 文件，包含以下字段：

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| `traces` | `(T, 4)` | 原始电压采样值，第一维为时间，建议使用 `float32`。 |
| `coords` | `(4, 2)` | 通道物理坐标，单位为微米，顺序与 `traces` 的通道顺序一致。 |
| `fs_hz` | 标量 | 采样频率，单位为 Hz，设为 `30000.0`。 |

数组中的数值应为有限值，记录至少包含 100 个时间采样点，各通道使用一致的电压尺度。程序会完成滤波和归一化。通道数更多的探针可使用 Python API，或导出一个四通道组及其对应坐标供 CLI 使用。

例如，将已有的信号和坐标数组打包：

```python
import numpy as np

traces = np.load("traces.npy")  # (T, 4)
coords = np.load("coords.npy")  # (4, 2)，单位为微米

np.savez(
    "recording.npz",
    traces=traces.astype(np.float32),
    coords=coords.astype(np.float32),
    fs_hz=30000.0,
)
```

### 运行分选

```bash
vanillasort --input recording.npz --components 22 --output output/recording-k22
```

通过 `--components` 指定适合当前记录的高斯混合模型组分数 **K**。K 至少为 2，可用于分选的事件数应不少于 K。仓库中的 Hybrid Janelia 配置为：记录 **11/12** 使用 **22**，记录 **21/22/31/32** 使用 **24**。

也可以分别传入两个 `.npy` 文件：

```bash
vanillasort --input traces.npy --coords coords.npy --fs 30000 --components 22 --output output/recording-npy
```

NPZ 中的 `fs_hz` 优先于 `--fs`；`--coords` 指定的坐标优先于 NPZ 内的坐标。

仅导出尖峰检测结果：

```bash
vanillasort --input recording.npz --detect-only --output output/detections
```

## 输出结果

### Python API 输出

指定 `output_folder` 后，除返回 `BaseSorting` 外，还会保存：

| 路径 | 内容 |
| --- | --- |
| `sorting/` | 可通过 `spikeinterface.core.load()` 重新加载的分选对象。 |
| `events.npz` | 对齐的 `sample_index`、`unit_id`、`segment_index` 数组。 |
| `segmentN_patchM.npz` | 每个 segment 和邻域的详细数组。 |
| `run.json` | 参数、通道坐标、模型哈希、依赖版本和处理诊断信息。 |

使用 `restored.register_recording(recording)` 可将重新加载的分选结果与原始记录关联。

### CLI 输出

每次运行生成两个文件：

| 文件 | 内容 |
| --- | --- |
| `events.npz` | 事件时间、检测分数，以及分选运行的标签和模型输出。 |
| `run.json` | 配置、采样率、事件数量、归一化统计量、耗时、依赖版本、权重 SHA-256 哈希及处理诊断信息。 |

完成聚类后，`events.npz` 包含以下数组。**D** 表示通过筛选的检测事件数，**N** 表示可用于分选的事件数，**K** 表示指定的组分数。

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `detection_samples` | `(D,)` | 通过筛选的检测时间，以从 0 开始的采样点索引表示。 |
| `detection_scores` | `(D,)` | 检测器的 sigmoid 分数。 |
| `detection_snrs` | `(D, 4)` | 各事件在四个通道上的信噪比估计。 |
| `samples` | `(N,)` | 保留用于分选的事件采样点索引。 |
| `scores` | `(N,)` | 与 `samples` 对齐的检测分数。 |
| `labels`、`initial_labels` | 各为 `(N,)` | 最终归属与 GMM 初始归属，编号范围为 `0` 至 `K - 1`。 |
| `features` | `(N, 32)` | 在当前记录的事件间完成标准化的 HuiduRep 表征。 |
| `auxiliary` | `(N, 3)` | 标准化后的相对振幅对比特征。 |
| `templates` | `(2, K, 60, 4)` | 波形模板，依次按目标折、组分、采样点和通道索引。 |
| `core_counts` | `(2, K)` | 用于判断模板是否可用的核心事件数量。 |
| `means` | `(K, 35)` | GMM 在组合特征空间中的组分均值。 |
| `covariances`、`weights` | `(K, 35, 35)`、`(K,)` | GMM 的完整协方差矩阵及混合权重。 |

分选阶段保留满足 `30 <= sample < T - 30` 的检测事件，提取相对事件时间偏移为 `-30` 至 `+29` 的波形，因此 D 和 N 可能不同。仅检测模式保存三个 `detection_*` 数组。

读取以秒为单位的尖峰时间，并选择一个单元：

```python
import json
from pathlib import Path
import numpy as np

output = Path("output/recording-k22")
metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
with np.load(output / "events.npz") as events:
    spike_times_s = events["samples"] / metadata["fs_hz"]
    unit_ids = events["labels"]

unit_0_times_s = spike_times_s[unit_ids == 0]
units, spike_counts = np.unique(unit_ids, return_counts=True)
print(dict(zip(units.tolist(), spike_counts.tolist())))
```

时间以输入记录的起点为基准。可结合波形、放电率和尖峰间隔评估候选单元。`run.json` 的 `clustering` 字段记录 GMM 收敛状态、警告、模板构建情况及归属修正结果。

## 模型与缓存

默认模型为 `hybrid-janelia-2026.09`，托管于 [Kohaku2580/VanillaSort](https://huggingface.co/Kohaku2580/VanillaSort)。固定 revision 和 SHA-256 哈希记录在 [`default_model.json`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/configs/default_model.json) 中，每次加载都会校验权重哈希。

提前下载并校验权重：

```bash
vanillasort --verify-only
```

程序使用 Hugging Face 标准缓存（支持 `HF_HOME` / `HF_HUB_CACHE`），也可通过 `VANILLASORT_MODEL_CACHE` 指定其他 Hub 缓存根目录。缓存准备好后，设置 `HF_HUB_OFFLINE=1` 即可离线运行。

使用本地模型：

```python
sorting = vanillasort.sort(recording, model_path="/path/to/model.pt", components=22)
```

合并的 `.pt` 文件包含 `detector_state_dict`、`huidurep_state_dict` 和可选的 `config`。模型目录则包含 `config.json`，以及 `detector_checkpoint` 和 `huidurep_checkpoint` 指定的两个权重文件。加载时使用 `weights_only=True` 并严格匹配模型结构。

## 配置

常用命令行参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--components K` | 处理自己的数据时必填 | GMM 组分数。 |
| `--profile d1` | `d1` | 选择 `d1` 或 `canonical` 预处理配置。 |
| `--seed 30` | `30` | 本次运行及单次 GMM 初始化的随机种子。 |
| `--device auto` | `auto` | 使用 `auto`、`cpu`、`cuda` 或 `cuda:0` 等设备。 |
| `--seconds 10` | 完整记录 | 处理前 10 秒，指定时长应在记录范围内。 |
| `--detector-batch 8` | `8` | 每批检测的信号块数，每块含 2,500 个采样点。 |
| `--embedding-batch 128` | `128` | 每批送入 HuiduRep 的波形数。 |
| `--gpu-fraction 0.4` | `0.4` | PyTorch CUDA 分配器可使用的设备显存比例上限。 |
| `--output PATH` | `output` | 新建或空的结果目录。 |

遇到 CUDA 显存不足时，程序会将对应推理批次的大小减半并重试。滤波和归一化在主机内存中处理整段选定记录。运行前会检查可用内存是否达到约 `12 × 输入数组字节数 + 512 MiB`；波形和特征所需内存还会随事件数增长。

预处理与模型设置见 [`configs/default.json`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/configs/default.json)：

| 设置 | `d1`（默认） | `canonical` |
| --- | --- | --- |
| 带通滤波 | 300–5,000 Hz，5 阶 | 200–6,000 Hz，3 阶 |
| 陷波滤波 | 60 Hz | 关闭 |
| 检测阈值 | 0.039 | 0.038 |
| 直接接受阈值 | 0.045 | 0.041 |
| 波形来源 | 滤波后的电压 | 滤波后按通道进行中位数/MAD 归一化的电压 |

两种配置都会对检测器输入按通道进行中位数/MAD 归一化。事件信噪比筛选使用 canonical 的 200–6,000 Hz 信号。使用 `--seconds` 处理前缀时，归一化和聚类也在该前缀上拟合。请将 `run.json` 与结果一起保存，以记录这些选择。

## 模型架构

<p align="center">
  <a href="https://raw.githubusercontent.com/IgarashiAkatuki/VanillaSort/feat/inference-package/assets/architecture.jpg"><img src="https://raw.githubusercontent.com/IgarashiAkatuki/VanillaSort/feat/inference-package/assets/architecture.jpg" alt="VanillaSort 模型架构：VanillaDet 尖峰检测、HuiduRep 波形表征与 VanillaCluster 神经元归属" width="1000"></a>
</p>

**VanillaDet → HuiduRep → VanillaCluster。** 图中展示了检测器训练与事件筛选、波形表征学习，以及结合相对振幅特征和模板引导重分配的聚类流程。

## 方法

1. **VanillaDet：检测尖峰。** 卷积前端与六层局部 Transformer 输出逐采样点的检测分数。预训练模型使用 256 维隐状态、四个注意力头、旋转位置编码和 L2 归一化的 query/key。峰值筛选采用 12 个采样点的排除半径。达到直接接受阈值的事件予以保留；分数介于两个阈值之间的事件，需要最强通道的信噪比至少为 3，且另一个通道的信噪比至少为 2。
2. **HuiduRep：编码波形。** 每个事件提取 60 个采样点、四个通道的波形。预处理按通道在事件与时间维度上进行标准化，将时间轴插值到 90 个采样点，并通过通道重复与裁剪形成 11 通道输入。HuiduRep 内部完成去噪，输出 32 维表征，再在当前记录的事件间进行标准化。
3. **VanillaCluster：分配单元。** 根据各通道峰峰值占总峰峰值的比例，投影到 Helmert 对比基并标准化，得到三维相对振幅特征。完整协方差 GMM 对组合后的 35 维向量进行聚类，随后用一轮模板残差修正，在得分最高的三个候选组分间更新归属。

模板修正按交替的一秒时间块将事件分为两折。每个目标折使用另一折中 GMM 后验概率至少为 0.9 的事件构建模板。每个核心事件池最多保留 512 个事件，再按检测分数去除最低的 25%。有效模板需要至少 30 个保留事件且能量非零。残差拟合搜索 ±2 个采样点的位移和 0.5–2.0 的共享振幅缩放，事件时间保持不变。模板支持不足的事件沿用 GMM 归属。

## 实验结果

[论文表 1–2](https://www.biorxiv.org/content/10.64898/2026.09.18.752552)在 **Hybrid Janelia 静态（Static）与漂移（Drift）子集**上进行评估，每个子集包含 9 段记录，纳入具有真实标签且 SNR ≥ 3 的单元。分选准确率为 `TP / (TP + FP + FN)`，事件匹配容差为 ±6 个采样点。下表为**均值 ± 均值标准误（SEM）**，数值越高越好。

| 分选工具 | 静态准确率 | 漂移准确率 |
| --- | ---: | ---: |
| HerdingSpikes2 | 0.35 ± 0.01 | 0.29 ± 0.01 |
| IronClust | 0.57 ± 0.04 | 0.54 ± 0.03 |
| JRClust | 0.47 ± 0.04 | 0.35 ± 0.03 |
| KiloSort | 0.60 ± 0.02 | 0.51 ± 0.02 |
| KiloSort2 | 0.39 ± 0.03 | 0.30 ± 0.02 |
| KiloSort4 | 0.40 ± 0.03 | 0.34 ± 0.02 |
| MountainSort4 | 0.59 ± 0.02 | 0.36 ± 0.02 |
| MountainSort5 | 0.40 ± 0.06 | 0.33 ± 0.04 |
| SpykingCircus | 0.57 ± 0.01 | 0.48 ± 0.02 |
| Tridesclous | 0.54 ± 0.03 | 0.37 ± 0.02 |
| SimSort | 0.62 ± 0.04 | 0.56 ± 0.03 |
| HuiduRep（无 DAE） | 0.69 ± 0.02 | 0.56 ± 0.02 |
| HuiduRep（含 DAE） | 0.70 ± 0.02 | 0.60 ± 0.02 |
| **VanillaSort（无 DAE）** | **0.73 ± 0.02** | 0.61 ± 0.02 |
| **VanillaSort（含 DAE）** | **0.73 ± 0.01** | **0.64 ± 0.02** |

DAE 指去噪自编码器。其他工具的分数采用论文引用的 SpikeForest 或对应原始论文结果。

- **检测：** VanillaDet 在静态/漂移子集上的准确率为 0.74/0.71，SimSort 为 0.72/0.68，振幅阈值检测为 0.61/0.60。
- **分选：** 相比对应的 HuiduRep 基线，VanillaSort 在**静态记录上提高 3–4 个百分点**，在**漂移记录上提高 4–5 个百分点**（配对 Wilcoxon 检验，*p* < 0.05）。

## Package 开发

开发 package：

```bash
git clone --branch feat/inference-package https://github.com/IgarashiAkatuki/VanillaSort.git
cd VanillaSort
python -m pip install -e ".[dev]"
pytest
```

测试使用小型合成记录与本地生成的模型权重。[`requirements.txt`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/requirements.txt) 记录数值计算依赖的历史参考版本；package 的运行依赖由 [`pyproject.toml`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/pyproject.toml) 声明。

维护者可按[发布流程](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/docs/releasing.md)通过 GitHub Actions 发布版本。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| [`src/vanillasort/`](https://github.com/IgarashiAkatuki/VanillaSort/tree/feat/inference-package/src/vanillasort/) | 唯一核心实现：API、pipeline、models、geometry、checkpoint 和数值运算。 |
| [`run.py`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/run.py) | 转发到 package CLI 的兼容入口。 |
| [`pyproject.toml`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/pyproject.toml) / [`tests/`](https://github.com/IgarashiAkatuki/VanillaSort/tree/feat/inference-package/tests/) | 安装配置和小型合成测试。 |
| [`models/detector.py`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/models/detector.py) | VanillaDet 的卷积前端、注意力层与预测头。 |
| [`models/huidurep/`](https://github.com/IgarashiAkatuki/VanillaSort/tree/feat/inference-package/src/vanillasort/models/huidurep/) | HuiduRep 编码器、解码器、投影模块及 `CMAES` 模型类。 |
| [`ops.py`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/ops.py) | 事件筛选、波形预处理、振幅特征和模板修正所需的数值运算。 |
| [`configs/default.json`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/configs/default.json) | 模型结构、预处理配置、阈值及聚类设置。 |
| [`configs/default_model.json`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/src/vanillasort/configs/default_model.json) | Hugging Face 模型仓库、固定 revision 和校验哈希；权重单独缓存。 |
| [`requirements.txt`](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/requirements.txt) | 用于复现原始环境的数值计算依赖参考版本。 |

## 引用

**使用 VanillaSort 必须引用以下论文：**

Zishuo Feng and Feng Cao. **Spike Sorting with VanillaSort.** bioRxiv, 2026. [doi:10.64898/2026.09.18.752552](https://www.biorxiv.org/content/10.64898/2026.09.18.752552)。

```bibtex
@article{feng2026vanillasort,
  title   = {Spike Sorting with {VanillaSort}},
  author  = {Feng, Zishuo and Cao, Feng},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.18.752552},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.18.752552}
}
```

## 许可证

[GNU Affero General Public License v3.0](https://github.com/IgarashiAkatuki/VanillaSort/blob/feat/inference-package/LICENSE)。
