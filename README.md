# 🔱 TasAnchor项目 - 功能测试模块数学建模与分析

## 📄 项目简介 (Project Overview)

本项目是 2025 年 iGEM SCU-China 队伍 TasAnchor 项目**功能测试模块 (B组)**的数学建模与数据分析代码库。

我们的目标是利用数学模型对工程菌的**镉离子感应**和**吸附**功能进行精确的定量评估、参数提取和应用场景预测，以支撑金牌标准中的建模要求。

---

## 💡 模块与核心模型

本项目围绕 TasAnchor 系统的功能测试模块展开建模：

| 模块 | 生物学功能 | 核心数学模型 | 提取的关键参数 |
| :--- | :--- | :--- | :--- |
| **3.1 感应** | 生长与荧光响应 | Modified Gompertz, Hill 方程 | $\mu_{max}, \lambda, EC_{50}, F_{max}$ |
| **3.2 吸附** | 吸附容量与稳定性 | Langmuir / Freundlich 等温线, 二级抑制模型 | $q_{max}, K_L, MIC$ |
| **3.3 复合** | 感应-粘附复合 | (待进一步量化) | |
| **3.4 验证** | 统计检验与鲁棒性 | $t$-检验, 敏感性分析 | $p$-value, 敏感性指数 |

---

## 🛠️ 环境依赖 (Dependencies)

本项目依赖于 Python 科学计算生态系统。请确保使用 `requirements.txt` 安装所有必要的依赖。

### 1. 虚拟环境设置 (可选)

```bash
# 激活环境示例
conda activate igem_env
# 或
source venv/bin/activate
````

### 2\. 安装依赖

所有依赖都列于 `requirements.txt`。

```bash
pip install -r requirements.txt
# 核心依赖: numpy, pandas, matplotlib, scipy, scikit-learn
```

-----

## 🏃 完整使用流程 (Full Usage Workflow)

请严格按照脚本编号顺序运行，因为后续脚本依赖于前序脚本生成的参数和数据文件。

```bash
# 1. 数据提取和组织
python code/01_data_extraction.py

# 2. 模块 3.1：生长和荧光建模
python code/02_module_3.1.py

# 3. 模块 3.2：吸附和二级抑制建模
python code/03_module_3.2.py

# 4. 模块 3.3/3.4：综合分析与总结
python code/04_module_3.3_3.4.py
```

> **输出**: 运行后请检查 `figures/` 和 `results/` 文件夹。

-----

## 📈 核心结果亮点 (Key Results Summary)

以下是从模型中提取的关键参数和结论，用于 Wiki 的 **Results** 页面：

| 指标 | 模型 | 结果 | 意义 |
| :--- | :--- | :--- | :--- |
| **最大吸附容量** ($q_{max}$) | Langmuir | $\approx 9.80 \text{ mg/g}$ | **高效吸附**能力，为工业应用奠定基础。|
| **半最大有效浓度** ($EC_{50}$) | Hill 方程 | $\approx 1.24 \text{ mmol/L}$ | **灵敏响应**，确保系统能在低浓度 $Cd^{2+}$ 下被激活。 |
| **最大抑制浓度** ($MIC$) | 二级抑制模型 | $\approx 4.10 \text{ mg/L}$ | **高耐受性**，证明工程菌在 $Cd^{2+}$ 胁迫下仍能稳定生长。|
| **统计显著性** ($p$-value) | $t$-检验 (3.4) | $< 0.001$ | **功能验证**，证明工程菌的吸附效率显著优于对照组。 |

-----

## 🗺️ 文件结构 (File Structure Detail)

```
TasAnchor_Modeling/
├── code/                          # 核心 Python 脚本
│   ├── 01_data_extraction.py      # 数据提取与预处理
│   ├── 02_module_3.1.py           # 感应模块建模
│   ├── 03_module_3.2.py           # 吸附模块建模
│   ├── 04_module_3.3_3.4.py       # 综合分析与验证
│   └── utils.py                   # 通用工具函数库
│
├── data/                          # 实验数据存储
│   └── raw/                       # 原始/WebPlotDigitizer 提取的 CSV 文件
│
├── figures/                       # 自动生成的图表 (00_*.png 到 08_*.png)
├── results/                       # 建模结果 (参数表 .csv 和报告 .txt)
│
├── notebooks/                     # Jupyter Notebooks (交互式调试和探索)
│   ├── 01_Data_Exploration.ipynb
│   └── ... (共四个 Notebook)
│
├── README.md                      # 项目主页 (本文件)
├── WORKFLOW.md                    # 工作流程指南
├── FILES.md                       # 所有文件详细说明
└── requirements.txt               # Python 依赖清单
```

```
```