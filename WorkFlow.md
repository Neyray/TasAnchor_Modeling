# TasAnchor项目 - 完整工作流程指南

**项目**: 2025 iGEM SCU-China 功能测试模块建模  
**负责**: B组  
**最后更新**: 2025-11-22

---

## 📋 目录

1. [环境准备](#环境准备)
2. [Step 1: 数据提取](#step-1-数据提取)
3. [Step 2: 模块3.1建模](#step-2-模块31建模)
4. [Step 3: 模块3.2建模](#step-3-模块32建模)
5. [Step 4: 综合分析](#step-4-综合分析)
6. [数据更新流程](#数据更新流程)
7. [Jupyter Notebook使用](#jupyter-notebook使用)
8. [常见问题排查](#常见问题排查)

---

## 🔧 环境准备

### 1. 激活虚拟环境

```bash
# Windows
conda activate igem_env

# Mac/Linux
conda activate igem_env
```

### 2. 验证依赖包

```bash
python -c "import numpy, pandas, matplotlib, scipy; print('✓ 所有依赖已安装')"
```

### 3. 确认项目结构

```
TasAnchor_Modeling/
├── code/              # Python脚本
├── data/raw/          # 原始数据
├── figures/           # 生成图表
├── results/           # 模型结果
└── notebooks/         # Jupyter文件
```

---

## 📊 Step 1: 数据提取

### 目标
从SCU-China Wiki页面提取实验数据，转换为CSV格式。

### 操作步骤

**1.1 运行数据提取脚本**

```bash
python code/01_data_extraction.py
```

**预期输出**:
- ✅ 7个CSV文件在 `data/raw/`
- ✅ 1张预览图 `figures/00_data_overview.png`
- ✅ 1份报告 `results/data_extraction_report.txt`

**1.2 验证生成的数据文件**

```bash
# 检查文件是否生成
ls data/raw/

# 应该看到:
# module_3.1_growth_curves.csv
# module_3.1_fluorescence.csv
# module_3.2_biofilm.csv
# module_3.2_adsorption_9mg.csv
# module_3.2_adsorption_5mg.csv
# module_3.3_experimental_conditions.csv
# module_3.4_verification.csv
```

**1.3 检查数据预览图**

打开 `figures/00_data_overview.png`，确认6个子图显示正常。

### 数据提取说明

当前使用的是**示例数据**。如需使用真实数据：

1. 访问 https://automeris.io/WebPlotDigitizer/
2. 上传SCU-China页面的图表截图
3. 校准坐标轴（X1, X2, Y1, Y2）
4. 逐点提取数据
5. 导出为CSV或复制数据
6. 打开 `code/01_data_extraction.py`
7. 找到对应的数据数组（如 `growth_curve_data`）
8. 替换为真实数据
9. 重新运行 `python code/01_data_extraction.py`

---

## 🧬 Step 2: 模块3.1建模

### 目标
建立生长动力学模型和荧光响应模型。

### 涉及的模型

1. **Modified Gompertz模型** - 描述微生物生长曲线
   - 参数: A (最大OD), μ_max (生长速率), lag (滞后期)
   - 拟合每个Cd²⁺浓度下的生长曲线

2. **Hill方程** - 描述荧光剂量-响应关系
   - 参数: F_max (最大荧光), EC₅₀ (半数有效浓度), n (Hill系数)
   - 评估传感器灵敏度

### 操作步骤

**2.1 运行建模脚本**

```bash
python code/02_module_3.1.py
```

**2.2 查看终端输出**

你会看到：
- 每个浓度组的拟合参数（A, μ_max, lag, R²）
- Hill方程参数（F_max, EC₅₀, n, R²）
- 文件保存提示

**2.3 检查生成的文件**

图表：
- `figures/01_module_3.1_gompertz_fitting.png` (6个生长曲线拟合子图)
- `figures/02_module_3.1_hill_dose_response.png` (剂量-响应曲线)

数据：
- `results/module_3.1_gompertz_parameters.csv` (Gompertz参数表)
- `results/module_3.1_hill_parameters.csv` (Hill参数)
- `results/module_3.1_report.txt` (模块报告)

**2.4 解读结果**

打开 `results/module_3.1_report.txt`，关注：
- 生长速率随Cd²⁺浓度的变化趋势
- EC₅₀值（传感器灵敏度指标）
- 拟合优度R²（应>0.95）

### 关键结论示例

> 在1.5 mg/L Cd²⁺下，μ_max从0.162 h⁻¹降至0.117 h⁻¹，下降27.8%，表明工程菌在重金属胁迫下仍能保持生长。

---

## 💧 Step 3: 模块3.2建模

### 目标
建立吸附动力学模型和二级生长模型。

### 涉及的模型

1. **Langmuir吸附等温线** - 描述平衡吸附容量
   - 参数: q_max (最大吸附容量), K_L (亲和常数)

2. **Freundlich吸附等温线** - 备选模型
   - 参数: K_F, 1/n

3. **二级生长模型** - 描述μ_max与Cd²⁺浓度关系
   - 公式: μ_max(Cd) = μ₀ × [1 - (Cd/MIC)^n]

### 操作步骤

**3.1 运行建模脚本**

```bash
python code/03_module_3.2.py
```

**3.2 查看终端输出**

注意：
- Langmuir和Freundlich的R²对比（通常Langmuir更好）
- 吸附-解吸循环效率
- 二级模型参数（μ₀, MIC, n）

**3.3 检查生成的文件**

图表：
- `figures/03_module_3.2_adsorption_isotherm.png` (吸附等温线)
- `figures/04_module_3.2_adsorption_cycles.png` (循环性能)
- `figures/05_module_3.2_secondary_growth_model.png` (二级模型)

数据：
- `results/module_3.2_isotherm_parameters.csv` (等温线参数)
- `results/module_3.2_secondary_model_parameters.csv` (二级模型参数)
- `results/module_3.2_report.txt`

**3.4 解读结果**

关键指标：
- **q_max**: 最大吸附容量（单位：mg Cd²⁺ / g 干重菌体）
- **循环效率**: 第3次循环后的去除率（应>60%）
- **MIC**: 最小抑制浓度（预测菌株耐受上限）

### 关键结论示例

> Langmuir模型拟合优度R²=0.99，最大吸附容量q_max=5.07 mg/g。吸附-解吸循环3次后，9 mg/L组的去除效率仍达68.9%，证明系统可重复使用。

---

## 📈 Step 4: 综合分析

### 目标
统计验证、敏感性分析、实际应用预测。

### 包含的分析

1. **模块3.4验证** - t检验比较对照组和实验组
2. **敏感性分析** - 测试参数扰动±20%对预测的影响
3. **应用场景预测** - 计算不同规模废水处理所需菌量和成本

### 操作步骤

**4.1 运行综合分析脚本**

```bash
python code/04_module_3.3_3.4.py
```

**4.2 查看终端输出**

重点关注：
- t检验p值（<0.05表示显著差异）
- 敏感性分析结果（哪个参数影响最大）
- 各应用场景的菌量需求

**4.3 检查生成的文件**

图表：
- `figures/06_module_3.4_verification.png` (统计验证)
- `figures/07_sensitivity_analysis.png` (敏感性分析)
- `figures/08_application_scenarios.png` (应用预测)

数据：
- `results/application_scenarios.csv` (应用场景表)
- `results/final_summary_report.txt` (⭐最终总结报告)

**4.4 阅读最终报告**

打开 `results/final_summary_report.txt`，这是整个项目的总结，包含：
- 所有建模成果汇总
- 统计验证结果
- 敏感性分析结论
- 实际应用建议

---

## 🔄 数据更新流程

### 何时需要更新数据？

- 从WebPlotDigitizer提取了真实数据
- 发现原始数据有误
- 需要添加新的实验数据

### 更新步骤

**方法1：完全重新生成（推荐）**

```bash
# 1. 修改数据提取脚本
nano code/01_data_extraction.py  # 或用VSCode打开

# 2. 删除旧数据和结果（可选，避免混淆）
rm -rf data/raw/*
rm -rf results/*
rm -rf figures/*

# 3. 重新运行所有脚本
python code/01_data_extraction.py
python code/02_module_3.1.py
python code/03_module_3.2.py
python code/04_module_3.3_3.4.py
```

**方法2：增量更新（快速）**

如果只修改了某个模块的数据（如只改了模块3.1）：

```bash
# 1. 修改数据提取脚本
# 2. 只重新运行数据提取
python code/01_data_extraction.py

# 3. 重新运行受影响的模块
python code/02_module_3.1.py

# 4. 如果后续模块依赖这个模块的结果，也需要重新运行
python code/04_module_3.3_3.4.py  # 因为它读取module_3.1的参数
```

### 是否需要删除旧文件？

**不需要删除**：Python脚本会自动覆盖同名文件。

**建议删除的情况**：
- 文件数量或命名发生变化
- 需要确保没有残留的旧数据
- 发现结果异常，想完全清理重来

### 数据更新检查清单

更新后验证：
- [ ] CSV文件的行数和列数是否正确
- [ ] 预览图 `00_data_overview.png` 是否合理
- [ ] 终端没有报错或警告
- [ ] 拟合优度R²是否>0.90
- [ ] 最终报告的数值是否符合预期

---

## 📓 Jupyter Notebook使用

### Jupyter与Python脚本的关系

| 维度 | Python脚本 (.py) | Jupyter Notebook (.ipynb) |
|------|------------------|---------------------------|
| **用途** | 批量处理，自动化生成报告 | 交互式探索，演示展示 |
| **运行方式** | `python code/XX.py` | 在Jupyter界面逐单元格运行 |
| **输出位置** | 自动保存到文件夹 | 显示在Notebook内 |
| **适合场景** | 最终报告生成 | 数据探索、调试、教学 |
| **依赖关系** | 无依赖 | 依赖Python脚本生成的数据 |

### 使用Jupyter的建议工作流

**情况1：已运行完Python脚本**

```bash
# 1. 确保已运行完所有Python脚本
python code/01_data_extraction.py
python code/02_module_3.1.py
python code/03_module_3.2.py
python code/04_module_3.3_3.4.py

# 2. 启动Jupyter
jupyter notebook

# 3. 在浏览器中打开notebooks文件夹
# 4. 依次打开并运行:
#    - 01_Data_Exploration.ipynb (数据探索)
#    - 02_Primary_Model_Fitting.ipynb (初级建模)
#    - 03_Secondary_Model.ipynb (二级建模)
#    - 04_Sensitivity_Analysis.ipynb (敏感性分析)
```

**情况2：只用Jupyter进行交互式分析**

适合：
- 探索数据分布和异常值
- 调试建模参数
- 生成展示用的可视化
- 向团队演示建模过程

不适合：
- 批量生成最终报告
- 自动化流程

### Jupyter Notebook详细使用流程

**1. 启动Jupyter**

```bash
conda activate igem_env
cd TasAnchor_Modeling
jupyter notebook
```

浏览器会自动打开 `http://localhost:8888`

**2. 打开Notebook**

- 导航到 `notebooks/` 文件夹
- 点击 `01_Data_Exploration.ipynb`

**3. 运行单元格**

方法1：逐个运行
- 点击单元格
- 按 `Shift + Enter`（运行并跳到下一个）
- 或点击工具栏的"▶ Run"按钮

方法2：全部运行
- 菜单栏：`Cell` → `Run All`

**4. 查看结果**

- 图表会直接显示在单元格下方
- 数据表格会以格式化形式显示
- 可以随时修改代码重新运行

**5. 保存和导出**

保存：`Ctrl+S` 或 `File` → `Save and Checkpoint`

导出为HTML（方便分享）：
```bash
jupyter nbconvert --to html notebooks/01_Data_Exploration.ipynb
```

### Jupyter与主文件夹的数据共享

Jupyter Notebook通过相对路径访问主文件夹的数据：

```python
# 在Notebook中
import sys
sys.path.append('../code')  # 访问code文件夹的utils.py
from utils import load_data

# 加载Python脚本生成的数据
df = load_data('module_3.1_growth_curves.csv')  # 从 data/raw/ 读取
```

**数据流向**：

```
Python脚本 → 生成CSV和图表 → Jupyter读取 → 交互式分析
   ↓                              ↓
 data/raw/                    实时可视化
 results/                     参数调试
 figures/
```

### Jupyter常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `Shift + Enter` | 运行当前单元格并跳到下一个 |
| `Ctrl + Enter` | 运行当前单元格 |
| `A` | 在上方插入单元格（命令模式） |
| `B` | 在下方插入单元格（命令模式） |
| `DD` | 删除当前单元格（命令模式） |
| `M` | 转换为Markdown单元格 |
| `Y` | 转换为代码单元格 |
| `Esc` | 进入命令模式 |
| `Enter` | 进入编辑模式 |

---

## ❓ 常见问题排查

### 问题1：ModuleNotFoundError: No module named 'numpy'

**原因**: 虚拟环境未激活或依赖包未安装

**解决**:
```bash
conda activate igem_env
pip install numpy pandas matplotlib scipy seaborn scikit-learn
```

### 问题2：Hill拟合失败 - 数据点不足

**原因**: 数据点数量<参数数量（3个点拟合4个参数）

**解决**: 使用简化的三参数Hill方程（见修复补丁）

### 问题3：中文字符显示为方框

**原因**: Matplotlib缺少中文字体

**解决**: 
```python
# 在utils.py或脚本开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 问题4：Jupyter Notebook无法导入utils

**原因**: 路径问题

**解决**:
```python
import sys
import os
sys.path.append(os.path.abspath('../code'))
from utils import *
```

### 问题5：拟合优度R²为负数

**原因**: 模型不适合数据或数据质量差

**解决**:
1. 检查数据是否正确提取
2. 调整初始猜测值 `p0`
3. 尝试其他模型

### 问题6：运行脚本后没有生成图表

**原因**: 可能报错但被忽略，或文件夹权限问题

**解决**:
```bash
# 检查是否有报错
python code/02_module_3.1.py 2>&1 | tee log.txt

# 检查文件夹权限
ls -la figures/
chmod 755 figures/  # Linux/Mac
```

---

## 📚 参考资源

- **WebPlotDigitizer**: https://automeris.io/WebPlotDigitizer/
- **SCU-China Wiki**: https://2025.igem.wiki/scu-china/results
- **Gompertz模型**: Zwietering et al. (1990) Applied and Environmental Microbiology
- **Hill方程**: Hill, A.V. (1910) Journal of Physiology
- **Langmuir吸附**: Langmuir, I. (1918) Journal of the American Chemical Society

---

## 📞 技术支持

遇到问题？
1. 检查本文档的"常见问题排查"部分
2. 查看 `results/final_summary_report.txt` 了解整体流程
3. 联系团队其他成员或导师

**最后更新**: 2025-11-22  
**维护者**: B组