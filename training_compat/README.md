# 轻量 LTE PDSCH 占用检测网络

本目录是用户当前要求的新神经网络工程，交付位置 `D:\PDSCH\deep_learning`。历史仿真工程和读取接口已移除。本工程的依据是当前运行的“规划 LTE 仿真数据集生成”对话及本聊天明确的检测任务，不继承旧数据布局。

**当前边界：另一个对话仍在深入分析实测，最终仿真格式尚未定稿。** 网络、训练/评价/推理代码可以先独立实现；数据读取只接受 [显式接口草案](数据接入规范_草案.md)，不能声称已经接入那个尚未定稿的大规模数据集。软件测试中的随机数组只检查实现，不是 LTE 仿真样本，也不代表检测性能。

## 网络与输出

- 轻量 U-Net，三次下采样，默认通道数 8/16/32/64；可通过 `--base 16` 比较 16/32/64/128 配置。
- 每阶段两个深度可分离卷积单元；GroupNorm 和 ReLU；双线性上采样及跳跃连接。
- 接收复数网格提取 I、Q、对数功率；公共位置图可选。
- 主输出：每 PRB、每子帧的占用分数；辅助输出：每 RE 的占用分数。
- 主输出前按 12 个子载波、一个子帧的符号数汇总学习特征；不平均原始能量后代替网络判断。
- 输入边缘填充到 8 的倍数，输出裁回原坐标。支持不同带宽及正常/扩展 CP，按原尺寸分批。
- 默认损失为 PRB 二元交叉熵 + 0.3 × RE 二元交叉熵；推理再执行 Sigmoid。
- 参数量通过检查程序实际统计。参数少不等于所有硬件都快；最终测量需包括接收前端、数据搬运和神经网络。

## 文件入口

| 文件 | 功能 |
|---|---|
| `pdsch_dl/model.py` | 网络主体及两个输出头 |
| `pdsch_dl/data.py` | 新接口草案读取、连续窗口、训练归一化、同尺寸分批 |
| `pdsch_dl/engine.py` | 训练、早停、检查点、验证门限、独立评价 |
| `pdsch_dl/metrics.py` | 占用指标、误拆误合并统计、区间坐标 |
| `pdsch_dl/cli.py` / `run.py` | 命令行入口 |
| `tests/test_pipeline.py` | 尺寸、轴序、梯度、隔离与端到端软件测试 |
| `check_network.py` | 运行测试，统计真实参数量、卷积 MAC 和本机 CPU 耗时 |
| `运行.ps1` | 本机便捷入口，优先本目录 .venv 或 PDSCH_PYTHON；否则使用本次验收环境 |

## 环境

建议 Python 3.11 或更新的受支持版本。当前本机是 Python 3.14，检查依赖解析结果；未来服务器根据官方支持选择版本。以下命令在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe check_network.py --output software_validation.json
```

`requirements.txt` 的普通安装由 PyTorch 包决定 CUDA 支持；需服务器 GPU 加速时，按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 为实际驱动选择版本。仅做 CPU 运行检查可先通过官方 CPU 索引安装 torch，再安装 NumPy/SciPy。

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2"
```

测试报告描述实际使用的环境，不意味着当前 CUDA 训练已验收。

本机另建的验收环境在当前项目 `.runtime/pdsch-dl-venv`，没有修改全局 Python。将代码放到 D 盘后，可直接执行 `./运行.ps1 check` 复用该环境；它仍位于 C 盘，迁移到服务器时需按上述步骤创建服务器自己的环境。

## 训练与使用

准备满足接口草案的新数据清单后，先检查数据。下面路径是待替换的示例，不存在则应报错，不会回退到历史批次。

```powershell
.\.venv\Scripts\python.exe run.py inspect --manifest D:\LTE_Sim_New\manifest.csv --output inspection.json
.\.venv\Scripts\python.exe run.py train --manifest D:\LTE_Sim_New\manifest.csv --output runs\experiment_001 --epochs 60 --batch-size 2 --device auto
.\.venv\Scripts\python.exe run.py evaluate --manifest D:\LTE_Sim_New\manifest.csv --checkpoint runs\experiment_001\best.pt --split test --output runs\experiment_001\test.json
.\.venv\Scripts\python.exe run.py predict --input D:\LTE_Sim_New\receiver_only.mat --checkpoint runs\experiment_001\best.pt --output predictions\capture_001
```

默认窗口 10 子帧（10 ms），默认 `--base 8`。显存不足先将 batch-size 降至 1；需要提高模型容量时可比较 `--base 16`。`--window` 是训练与部署契约的一部分，保存在检查点。多支路使用 `--rx all`；公共图使用 `--public`；功率对照使用 `--mode power`。

输出目录必须尚不存在，避免覆盖实验。检查点同时保存权重、输入设置、归一化和验证集选出的门限。`best.pt` 根据验证集 PRB F1 选择，测试集不参与训练或门限选择。`last.pt` 保存最后一轮；当前不提供优化器状态续训，不能将它当作完整断点续训文件。

调试时可以设置 `--max-batches 2 --epochs 2`；这种检查点会标记 limited_run，不能把软件通路跑通当作有效训练结论。

## 预测结果和评价边界

`probabilities.npz` 保存 PRB/RE 概率、子帧编号和有效标记；`regions.json` 保存每子帧的连续频率区间、PRB/有效子载波起止、辅助符号边界估计。所有坐标从 0 开始；子载波坐标不包含 FFT DC。不能直接把相对帧序号解释为 SFN。

预测不读取 label 字段。输出缺失标记必须被使用：短于训练窗口或包含缺失编号而无法组成窗口的子帧不会被当作“无占用”。末尾有完整上下文时采用重叠窗口平均覆盖。

模型输出的是连续占用区域，不保证区分相邻用户、同一用户的不连续资源或不同调度实例。RE 符号边界只是辅助估计，可能与 PRB 输出不一致；原始概率一并保留，不能强行填平全部低能量间隙。概率未做可信度校准。

评价提供 PRB Precision/Recall/F1/IoU、RE 指标及误拆/误合并。误拆指同一个真实频率段与多个预测段重叠；误合并指一个预测段与多个真实段重叠。漏检和虚警同时报告，不能只看分段比例。同步拒绝与未使用子帧另计。

正式研究还需在可信仿真与独立实测上验证，与能量门限、间隙合并方法对照，按信噪比、深衰落宽度/持续时间及干扰条件分组评估。测试通过只说明软件行为，不代表实测检测精度。
