# TasAnchor项目 - 文件功能详细说明

**文档目的**: 详细解释每个文件的功能、输入输出、核心算法和使用方法  
**最后更新**: 2025-11-22

---

## 📁 文件结构总览

```
TasAnchor_Modeling/
├── code/                          # 核心Python脚本
│   ├── 01_data_extraction.py      # 数据提取与预处理
│   ├── 02_module_3.1.py           # 感应模块建模
│   ├── 03_module_3.2.py           # 吸附模块建模
│   ├── 04_module_3.3_3.4.py       # 综合分析与验证
│   └── utils.py                   # 通用工具函数库
│
├── data/                          # 数据文件
│   ├── raw/                       # 原始实验数据（CSV）
│   └── processed/                 # 处理后的数据
│
├── figures/                       # 生成的图表（PNG）
├── results/                       # 建模结果（CSV + TXT）
│
├── notebooks/                     # Jupyter Notebook
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Primary_Model_Fitting.ipynb
│   ├── 03_Secondary_Model.ipynb
│   └── 04_Sensitivity_Analysis.ipynb
│
├── README.md                      # 项目简介
├── WORKFLOW.md                    # 工作流程指南
├── FILES.md                       # 本文档
└── requirements.txt               # 依赖包列表
```

---

## 🔧 核心Python脚本

### 1. `code/utils.py` - 通用工具函数库

**文件类型**: 工具模块  
**行数**: ~400行  
**依赖**: numpy, pandas, matplotlib, scipy, sklearn

#### 功能概述

这是整个项目的**基础设施**，提供所有建模脚本共用的函数。相当于一个工具箱，其他脚本通过 `from utils import *` 调用。

#### 包含的函数分类

##### A. 数据加载与保存

**`load_data(filename, folder='data/raw')`**
- **作用**: 加载CSV数据文件
- **输入**: 
  - `filename`: 文件名（如 'module_3.1_growth_curves.csv'）
  - `folder`: 数据文件夹路径（默认 'data/raw'）
- **输出**: pandas DataFrame
- **特性**: 自动添加.csv后缀，输出加载信息
- **示例**:
  ```python
  df = load_data('module_3.1_growth_curves.csv')
  # ✓ 已加载数据: data/raw/module_3.1_growth_curves.csv (11 行, 6 列)
  ```

**`save_results(data, filename, folder='results')`**
- **作用**: 保存结果数据到CSV
- **输入**: 
  - `data`: pandas DataFrame或字典
  - `filename`: 输出文件名
  - `folder`: 输出文件夹（默认 'results'）
- **输出**: 无（直接保存文件）
- **特性**: 自动创建文件夹，自动将字典转DataFrame
- **示例**:
  ```python
  params = {'A': 1.5, 'mu_max': 0.4}
  save_results(params, 'gompertz_params.csv')
  ```

##### B. 统计分析

**`calculate_statistics(data)`**
- **作用**: 计算描述性统计量
- **输入**: array-like数据
- **输出**: 字典，包含mean, std, min, max, median, cv_%
- **用途**: 快速了解数据分布特征
- **示例**:
  ```python
  data = [1.2, 1.5, 1.3, 1.6, 1.4]
  stats = calculate_statistics(data)
  # {'mean': 1.4, 'std': 0.158, 'cv_%': 11.3, ...}
  ```

**`calculate_model_metrics(y_true, y_pred, model_name='Model')`**
- **作用**: 计算模型评估指标
- **输入**: 
  - `y_true`: 真实值数组
  - `y_pred`: 模型预测值数组
  - `model_name`: 模型名称（用于标注）
- **输出**: 字典，包含R², RMSE, MAE, MAPE
- **核心算法**:
  - R² = 1 - SS_res / SS_tot（决定系数）
  - RMSE = √(Σ(y_true - y_pred)² / n)（均方根误差）
  - MAE = Σ|y_true - y_pred| / n（平均绝对误差）
  - MAPE = Σ|y_true - y_pred| / y_true / n × 100%（平均百分比误差）
- **用途**: 评估模型拟合优度
- **示例**:
  ```python
  y_true = [1, 2, 3, 4, 5]
  y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
  metrics = calculate_model_metrics(y_true, y_pred, 'Gompertz')
  # {'model': 'Gompertz', 'R2': 0.989, 'RMSE': 0.158, ...}
  ```

**`perform_ttest(group1, group2, alpha=0.05)`**
- **作用**: 执行独立样本t检验
- **输入**: 两组数据和显著性水平
- **输出**: 字典，包含t统计量, p值, 是否显著
- **算法**: scipy.stats.ttest_ind（Welch's t-test）
- **用途**: 比较两组数据是否有显著差异
- **解读**:
  - p < 0.05: 有显著差异
  - p >= 0.05: 无显著差异
- **示例**:
  ```python
  control = [15.2, 12.8, 18.1]
  experimental = [42.8, 39.5, 46.2]
  result = perform_ttest(control, experimental)
  # {'t_statistic': -10.73, 'p_value': 0.0004, 'significant': True}
  ```

##### C. 绘图工具

**`set_plot_style(style='seaborn')`**
- **作用**: 设置Matplotlib绘图样式
- **输入**: 样式名称（'seaborn', 'scientific', 'default'）
- **效果**: 
  - 'seaborn': 美观的网格背景
  - 'scientific': 学术论文风格（衬线字体，粗坐标轴）
  - 'default': Matplotlib默认样式
- **示例**:
  ```python
  set_plot_style('scientific')  # 适合发表论文
  ```

**`save_figure(fig, filename, folder='figures', dpi=300)`**
- **作用**: 保存图表到文件
- **输入**: 
  - `fig`: matplotlib figure对象
  - `filename`: 输出文件名
  - `folder`: 输出文件夹
  - `dpi`: 分辨率（默认300，适合出版）
- **特性**: 
  - 自动添加.png后缀
  - 白色背景，紧凑布局
  - 高分辨率
- **示例**:
  ```python
  fig, ax = plt.subplots()
  ax.plot([1,2,3], [1,4,9])
  save_figure(fig, '01_test_plot.png')
  ```

**`plot_residuals(y_true, y_pred, title='Residual Analysis')`**
- **作用**: 绘制残差分析图
- **输入**: 真实值、预测值、标题
- **输出**: 包含2个子图的figure对象
  - 左图: 残差vs预测值（检查是否有模式）
  - 右图: 残差直方图（检查正态性）
- **用途**: 诊断模型问题
  - 残差随机分布 → 模型良好
  - 残差有规律 → 模型欠拟合或过拟合
- **示例**:
  ```python
  fig = plot_residuals(y_true, y_pred, 'Gompertz Model Residuals')
  plt.show()
  ```

##### D. 数据转换

**`concentration_unit_converter(value, from_unit='mg/L', to_unit='mmol/L', molecular_weight=112.41)`**
- **作用**: 镉离子浓度单位转换
- **输入**: 
  - `value`: 浓度值
  - `from_unit`: 原始单位（'mg/L', 'mmol/L', 'g/L'）
  - `to_unit`: 目标单位
  - `molecular_weight`: 分子量（Cd²⁺ = 112.41 g/mol）
- **算法**:
  - mg/L → mmol/L: value / molecular_weight
  - mmol/L → mg/L: value × molecular_weight
- **示例**:
  ```python
  cd_mmol = concentration_unit_converter(9.0, 'mg/L', 'mmol/L')
  # 0.0800 mmol/L
  ```

##### E. 模型诊断

**`residual_analysis(y_true, y_pred)`**
- **作用**: 残差统计分析
- **输入**: 真实值和预测值数组
- **输出**: 字典，包含
  - `residuals`: 残差数组
  - `mean_residual`: 平均残差（应接近0）
  - `std_residual`: 残差标准差
  - `normal_test_p`: Shapiro-Wilk正态性检验p值
  - `is_normal`: 残差是否正态分布（p>0.05）
- **用途**: 检验模型假设是否满足
- **示例**:
  ```python
  res = residual_analysis(y_true, y_pred)
  if res['is_normal']:
      print("残差正态分布，模型假设满足")
  ```

##### F. 格式化输出

**`print_section(title, width=70)`**
- **作用**: 打印醒目的章节标题
- **效果**: 
  ```
  ======================================================================
                              模块 3.1：建模
  ======================================================================
  ```

**`print_subsection(title, width=70)`**
- **作用**: 打印小节标题
- **效果**:
  ```
  第一部分：生长曲线建模
  ----------------------------------------------------------------------
  ```

**`print_dict(data_dict, title=None)`**
- **作用**: 格式化打印字典内容
- **效果**:
  ```
  Hill方程参数:
    F_max: 260.0000
    EC50: 1.2350
    n: 2.4500
  ```

#### 使用场景

**场景1**: 快速加载数据并计算统计量
```python
from utils import *

df = load_data('module_3.1_growth_curves.csv')
stats = calculate_statistics(df['OD600_0mg_L'])
print_dict(stats, "0 mg/L组统计")
```

**场景2**: 评估模型性能
```python
from utils import *

# 拟合模型
y_pred = my_model(x_data, *params)

# 评估
metrics = calculate_model_metrics(y_data, y_pred, 'My Model')
print_dict(metrics)

# 诊断
residuals = residual_analysis(y_data, y_pred)
if not residuals['is_normal']:
    print("警告: 残差不满足正态分布")
```

**场景3**: 生成专业图表
```python
from utils import *

set_plot_style('scientific')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'o-')
ax.set_xlabel('Time (h)')
ax.set_ylabel('OD₆₀₀')
save_figure(fig, '01_growth_curve.png', dpi=600)
```

---

### 2. `code/01_data_extraction.py` - 数据提取脚本

**文件类型**: 数据预处理脚本  
**行数**: ~450行  
**运行时间**: ~5秒  
**输出**: 7个CSV + 1张图 + 1份报告

#### 功能概述

这是整个项目的**起点**。它的作用是将SCU-China Wiki页面上的图表数据转换成计算机可读的CSV格式，为后续建模提供标准化输入。

#### 核心功能

##### A. 数据提取

**提取的数据源**:
1. **Fig 3.1-4**: 不同Cd²⁺浓度下的生长曲线（5条曲线）
2. **Fig 3.1-5**: 荧光强度随时间和浓度变化（3条曲线）
3. **Fig 3.2-4**: 生物膜形成的OD570值（4个浓度点）
4. **Fig 3.2-5**: 吸附-解吸循环数据（9mg/L和5mg/L两组）
5. **Fig 3.4-1**: 吸附效率验证数据（2组对比）

**数据格式**:

所有数据都存储在Python字典中，然后转换为pandas DataFrame：

```python
growth_curve_data = {
    'time_h': [0, 2, 4, 6, ...],  # 时间点
    'OD600_0mg_L': [0.05, 0.08, ...],  # 0 mg/L组的OD600值
    'OD600_0.25mg_L': [0.05, 0.08, ...],  # 0.25 mg/L组
    # ... 其他浓度组
}
df = pd.DataFrame(growth_curve_data)
```

##### B. 数据验证

脚本会输出数据统计信息：
```python
✓ 生长曲线数据: 11 个时间点, 5 个浓度组
✓ 荧光强度数据: 7 个时间点, 3 个浓度组
```

##### C. 数据可视化

生成 `00_data_overview.png`，包含6个子图：
1. 生长曲线总览（5条线）
2. 荧光响应总览（3条线）
3. 生物膜形成（误差棒）
4. 9 mg/L吸附-解吸循环
5. 5 mg/L吸附-解吸循环
6. 吸附效率验证（柱状图）

**图表特点**:
- 高分辨率（300 dpi）
- 中文标签（自动识别操作系统选择字体）
- 颜色编码清晰
- 数值标注

##### D. 报告生成

生成 `data_extraction_report.txt`，内容包括：
- 提取的数据文件列表
- 数据汇总表
- 重要提示（如何替换真实数据）
- 下一步操作指引

#### 输入与输出

**输入**: 无（数据硬编码在脚本中）

**输出**:
```
data/raw/
├── module_3.1_growth_curves.csv           # 11行×6列
├── module_3.1_fluorescence.csv            # 7行×4列
├── module_3.2_biofilm.csv                 # 4行×3列
├── module_3.2_adsorption_9mg.csv          # 6行×6列
├── module_3.2_adsorption_5mg.csv          # 6行×6列
├── module_3.3_experimental_conditions.csv # 1行×7列
└── module_3.4_verification.csv            # 2行×6列

figures/
└── 00_data_overview.png                   # 组合图

results/
└── data_extraction_report.txt             # 文本报告
```

#### 如何修改数据

**步骤1**: 使用WebPlotDigitizer提取真实数据

**步骤2**: 打开 `01_data_extraction.py`

**步骤3**: 找到对应的数据数组（例如，修改生长曲线）：

```python
# 第48行左右
growth_curve_data = {
    'time_h': [0, 2, 4, 6, 8, 10, ...],  # ← 替换为你提取的X值
    'OD600_0mg_L': [0.05, 0.08, 0.15, ...],  # ← 替换为你提取的Y值
    ...
}
```

**步骤4**: 保存并重新运行

```bash
python code/01_data_extraction.py
```

#### 关键代码片段

**数据提取与保存**:
```python
# 定义数据
growth_curve_data = {...}
df_growth = pd.DataFrame(growth_curve_data)

# 保存CSV
df_growth.to_csv('data/raw/module_3.1_growth_curves.csv', index=False)
```

**数据可视化**:
```python
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 子图1: 生长曲线
ax1 = fig.add_subplot(gs[0, :2])
for col in df_growth.columns[1:]:
    ax1.plot(df_growth['time_h'], df_growth[col], marker='o', label=col)
# ...

plt.savefig('figures/00_data_overview.png', dpi=300)
```

#### 常见问题

**Q1**: 数据点数量不匹配怎么办？  
**A**: 确保每列的长度相同。例如，如果 `time_h` 有11个点，所有 `OD600_*` 列也必须有11个点。

**Q2**: 如何添加新的浓度组？  
**A**: 在字典中添加新的键值对：
```python
'OD600_2mg_L': [0.04, 0.06, 0.09, ...]
```

**Q3**: 图表中的中文显示为方框？  
**A**: 检查第24-30行的字体设置，确保系统有对应字体。

---

### 3. `code/02_module_3.1.py` - 感应模块建模

**文件类型**: 建模脚本  
**行数**: ~280行  
**运行时间**: ~10秒  
**输出**: 2张图 + 3个CSV + 1份报告

#### 功能概述

建立**生长动力学模型**和**荧光响应模型**，量化工程菌对镉离子的生长响应和传感器性能。

#### 核心算法

##### A. Modified Gompertz模型

**数学公式**:
```
OD(t) = A × exp(-exp(μ_max × e / A × (lag - t) + 1))
```

**参数含义**:
- **A**: 最大OD值（渐近线），表示菌株能达到的最高细胞密度
- **μ_max**: 最大比生长速率（h⁻¹），斜率最大点的切线斜率
- **lag**: 滞后期（h），菌株适应环境所需的时间

**物理意义**:
- A越大 → 菌株生物量越高
- μ_max越大 → 生长越快
- lag越长 → 适应期越长

**拟合过程**:
```python
from scipy.optimize import curve_fit

def gompertz_model(t, A, mu_max, lag):
    return A * np.exp(-np.exp(mu_max * np.e / A * (lag - t) + 1))

# 初始猜测值
p0 = [1.5, 0.4, 5.0]  # [A, μ_max, lag]

# 非线性最小二乘拟合
popt, pcov = curve_fit(gompertz_model, t_data, od_data, p0=p0, maxfev=10000)
```

**初始猜测值设置**:
- `A`: 观察数据的最大值×1.1
- `μ_max`: 0.3-0.5（微生物典型值）
- `lag`: 数据起始上升的时间点

**拟合优度评估**:
```python
od_pred = gompertz_model(t_data, *popt)
R2 = r2_score(od_data, od_pred)  # 应>0.95
RMSE = np.sqrt(mean_squared_error(od_data, od_pred))  # 越小越好
```

##### B. Hill方程

**数学公式（四参数）**:
```
F(C) = F_min + (F_max - F_min) × C^n / (EC50^n + C^n)
```

**简化版（三参数，假设F_min=0）**:
```
F(C) = F_max × C^n / (EC50^n + C^n)
```

**参数含义**:
- **F_max**: 最大荧光强度，饱和时的荧光值
- **EC50**: 半数有效浓度，达到50%最大响应时的Cd²⁺浓度
- **n**: Hill系数，协同性指标
  - n=1: 无协同性（简单结合）
  - n>1: 正协同性（结合后更容易结合）
  - n<1: 负协同性（结合后更难结合）

**拟合过程**:
```python
def hill_equation_simple(x, F_max, EC50, n):
    return F_max * x**n / (EC50**n + x**n)

p0 = [max(fluorescence), 1.0, 2.0]  # [F_max, EC50, n]
popt, pcov = curve_fit(hill_equation_simple, concentrations, fluorescence, p0=p0)
```

**EC50的意义**:
- EC50越小 → 传感器越灵敏（低浓度就能检测到）
- 检测范围约为 0.1×EC50 ~ 10×EC50

#### 执行流程

**步骤1**: 加载数据
```python
df_growth = load_data('module_3.1_growth_curves.csv')
df_fluo = load_data('module_3.1_fluorescence.csv')
```

**步骤2**: 对每个Cd²⁺浓度组拟合Gompertz
```python
for col in df_growth.columns[1:]:  # 跳过time_h列
    # 提取数据
    t_data = df_growth['time_h'].values
    od_data = df_growth[col].values
    
    # 拟合
    popt, pcov = curve_fit(gompertz_model, t_data, od_data, p0=[1.5, 0.4, 5.0])
    
    # 保存参数
    results.append({'Cd': cd_conc, 'A': popt[0], 'mu_max': popt[1], ...})
```

**步骤3**: 拟合Hill方程
```python
# 使用25h时刻的荧光数据（峰值）
time_point = 25
idx = np.argmin(np.abs(df_fluo['time_h'] - time_point))
fluorescence = [df_fluo[col].iloc[idx] for col in df_fluo.columns[1:]]

# 拟合
popt_hill, _ = curve_fit(hill_equation_simple, concentrations, fluorescence, p0=[260, 1, 2])
```

**步骤4**: 生成图表
- 6个子图的生长曲线拟合图
- 1张荧光剂量-响应曲线图

**步骤5**: 保存结果
- CSV: Gompertz参数表、Hill参数
- TXT: 模块报告

#### 输入与输出

**输入**:
- `data/raw/module_3.1_growth_curves.csv`
- `data/raw/module_3.1_fluorescence.csv`

**输出**:
```
figures/
├── 01_module_3.1_gompertz_fitting.png   # 6个子图，每个浓度组一个
└── 02_module_3.1_hill_dose_response.png # Hill拟合曲线

results/
├── module_3.1_gompertz_parameters.csv   # 5行×6列（每个浓度组的参数）
├── module_3.1_hill_parameters.csv       # 1行×4列
└── module_3.1_report.txt                # 文本报告
```

#### 结果解读

**Gompertz参数表示例**:
```
Cd_concentration_mg_L  A_max_OD  mu_max_h-1  lag_time_h     R2
                 0.00    1.4931      0.1616      3.7784 0.9959
                 0.25    1.4435      0.1547      3.8187 0.9956
                 0.50    1.3921      0.1499      3.9685 0.9955
                 1.00    1.2906      0.1365      3.9957 0.9951
                 1.50    1.1420      0.1169      4.1741 0.9942
```

**关键发现**:
- A随Cd²⁺浓度增加而下降（生物量减少）
- μ_max从0.162降至0.117 h⁻¹（下降27.8%）
- lag从3.78延长至4.17 h（适应期延长）
- R²均>0.99（拟合优秀）

**Hill参数示例**:
```
F_max: 260.00 FU
EC50: 1.235 mmol/L
Hill_coefficient_n: 2.450
R2: 0.9876
```

**关键发现**:
- EC50 = 1.235 mmol/L ≈ 139 mg/L（传感器灵敏度中等）
- n = 2.45 > 1（正协同性，响应陡峭）
- 检测范围约为 0.12-12 mmol/L

#### 参数调整指南

**Gompertz拟合不佳时**:
1. 调整初始猜测值 `p0`
   ```python
   p0 = [max(od_data)*1.1, 0.3, 3.0]  # 更接近实际值
   ```
2. 增加最大迭代次数
   ```python
   curve_fit(..., maxfev=50000)
   ```
3. 检查数据质量（是否有异常点）

**Hill拟合失败时**:
- 使用三参数模型（去掉F_min）
- 确保至少有4个数据点
- 调整初始猜测值

#### 常见问题

**Q1**: 为什么只用25h的荧光数据拟合Hill方程？  
**A**: 因为25h是荧光响应的峰值时刻，此时信号最强。使用峰值数据能更准确评估传感器的最大能力。

**Q2**: R²什么水平算好？  
**A**: 
- R² > 0.95: 优秀
- 0.90 < R² < 0.95: 良好
- R² < 0.90: 需改进（检查数据或模型）

**Q3**: 如何选择初始猜测值？  
**A**: 
- A: 数据最大值×1.1
- μ_max: 0.3-0.5（微生物典型值）
- lag: 目测数据起始上升的时间点



-----

## 📁 文件功能详细说明 (续)

### 3\. `code/03_module_3.2.py` - 吸附模块建模

**文件类型**: 核心建模脚本 (Python)  
**行数**: \~380行  
**运行时间**: \~10秒  
**输出**: 3张图 + 3个CSV + 1份报告

#### 功能概述

此脚本是\*\*B任务（功能测试）\*\*的核心建模文件，针对 **TasA-SmtA** 融合蛋白的镉离子吸附功能进行定量分析。它执行以下任务：

1.  **吸附等温线建模**: 拟合 Langmuir 和 Freundlich 模型，确定最大吸附容量 ($q_{max}$) 和吸附亲和力 ($K_L$ 或 $1/n$)。
2.  **吸附循环分析**: 评估菌株在多轮吸附-解吸过程中的性能衰减，验证其工业应用潜力。
3.  **二级生长模型**: 建立 $\mu_{max}$ 与 Cd$^{2+}$ 浓度的定量关系，连接模块 3.1 和 3.2 的模型结果。

#### 核心算法与模型

##### A. Langmuir & Freundlich 等温线模型

用于描述平衡浓度 ($C_e$) 与平衡吸附量 ($q_e$) 之间的关系。

| 模型 | 数学公式 (LaTeX) | 物理意义 |
| :--- | :--- | :--- |
| **Langmuir** | $$q_e = \frac{q_{\max} \cdot K_L \cdot C_e}{1 + K_L \cdot C_e}$$ | 假设**单层吸附**，位点均匀。|
| **Freundlich** | $$q_e = K_F \cdot C_e^{1/n}$$ | 假设**多层吸附**，表面不均匀。|

**吸附量计算核心代码 (Langmuir 拟合)**

```python
# 1. 计算平衡吸附量 qe
def calculate_qe(C0, removal_percent, V=1.0, m=1.0):
    Ce = C0 * (1 - removal_percent / 100)
    return (C0 - Ce) * V / m

# 2. 定义模型函数
def langmuir_model(Ce, q_max, K_L):
    return q_max * K_L * Ce / (1 + K_L * Ce)

# 3. 拟合
popt_lang, _ = curve_fit(langmuir_model, Ce_values, qe_values, p0=[5, 1])
metrics_lang = calculate_model_metrics(qe_values, langmuir_model(Ce_values, *popt_lang))
```

##### B. 二级生长模型 (幂函数抑制, Power Law Inhibition)

利用模块 3.1 得到的不同 Cd$^{2+}$ 浓度下的 $\mu_{max}$ 离散点进行拟合。

**数学公式**:
$$\mu_{\max}(Cd) = \mu_0 \cdot \left[1 - \left(\frac{Cd}{MIC}\right)^n\right]$$

**模型拟合代码**

```python
# 1. 加载 3.1 模块的参数
df_gompertz = load_data('module_3.1_gompertz_parameters.csv', folder='results')
cd_conc = df_gompertz['Cd_concentration_mg_L'].values
mu_max_values = df_gompertz['mu_max_h-1'].values

# 2. 定义二级模型
def secondary_growth_model(Cd, mu0, MIC, n):
    return mu0 * (1 - (Cd / MIC)**n)

# 3. 拟合
popt_sec, _ = curve_fit(secondary_growth_model, cd_conc, mu_max_values, p0=[0.16, 4.0, 2.0])
```

#### 输入与输出

**输入**:

  - `results/module_3.1_gompertz_parameters.csv` (**核心依赖**，提供 $\mu_{max}$ 数据)
  - `data/raw/3.2-5_9mg_pht01.csv`, `3.2-5_9mg_control.csv` 等 (吸附实验数据)

**输出**:

```
figures/
├── 03_module_3.2_adsorption_isotherm.png      # Langmuir & Freundlich 拟合对比图
├── 04_module_3.2_adsorption_cycles.png        # 菌株吸附-解吸循环性能图
└── 05_module_3.2_secondary_growth_model.png   # μ_max vs Cd²⁺ 抑制曲线

results/
├── module_3.2_isotherm_parameters.csv         # Langmuir/Freundlich 参数 (q_max, K_L/K_F, R²)
├── module_3.2_secondary_model_parameters.csv  # 二级生长模型参数 (μ₀, MIC, n, R²)
└── module_3.2_report.txt                      # 文本总结报告
```

### 4\. `code/04_module_3.3_3.4.py` - 综合分析与验证

**文件类型**: 综合分析与验证脚本 (Python)  
**行数**: \~300行  
**运行时间**: \~5秒  
**输出**: 3张图 + 2个CSV + 1份总结报告

#### 功能概述

此脚本标志着 **B任务** 的完成，用于项目的**收尾和验证**。它整合了所有模块的结果，并执行关键的统计和预测任务：

1.  **模块 3.4 统计验证**: 使用 t-检验验证工程菌（TasA-SmtA）的吸附效率是否**显著**高于对照组。
2.  **敏感性分析**: 测试二级生长模型对关键参数（如 $\mu_0, MIC$）变化的鲁棒性。
3.  **应用场景预测**: 估算处理特定污染所需的菌株用量（基于 $q_{max}$）。
4.  **生成最终报告**: 汇总项目所有建模结果。

#### 核心算法与分析

##### A. 模块 3.4 统计验证 (t-检验)

用于判断两种菌株（实验组 vs 对照组）的吸附效率差异是否具有统计学意义。

**t-检验核心代码**

```python
# 1. 加载验证实验数据
df_verification = load_data('module_3.4_verification.csv')
control_data = df_verification[df_verification['group'] == 'Control']['adsorption_efficiency_%']
exp_data = df_verification[df_verification['group'] == 'Experimental']['adsorption_efficiency_%']

# 2. 执行 t-检验
t_statistic, p_value = stats.ttest_ind(control_data, exp_data, equal_var=False)

# 3. 结果判断
is_significant = p_value < 0.05 
# 绘制柱状图，并在显著差异的柱子上添加 '***' 标记
```

##### B. 敏感性分析 (Sobol 或 OAT 简化)

通过微小扰动核心参数（如 $\mu_0$ 和 $MIC$），观察模型输出（如最大 OD 值）的变化，评估模型对参数输入的依赖程度。此脚本采用**单参数扰动法 (OAT 简化)**。

**敏感性分析核心代码**

```python
# 1. 加载二级模型参数
params = load_data('module_3.2_secondary_model_parameters.csv', folder='results').iloc[0]
mu0_nominal = params['mu0_h-1']
mic_nominal = params['MIC_mg_L']

# 2. 扰动参数 (±10%)
perturbation = 0.1
mu0_high = mu0_nominal * (1 + perturbation)
mu0_low = mu0_nominal * (1 - perturbation)

# 3. 预测并计算差异
# sensitivity_mu0 = (Output(mu0_high) - Output(mu0_low)) / (mu0_high - mu0_low)
# 绘制曲线族图 (figures/07_module_3.4_sensitivity_mu0.png)
```

#### 输入与输出

**输入**:

  - `data/raw/module_3.4_verification.csv` (验证实验结果)
  - `results/module_3.2_secondary_model_parameters.csv` (二级模型参数，用于敏感性分析)

**输出**:

```
figures/
├── 06_module_3.4_verification_ttest.png       # 验证实验（tasA-smtA vs 对照）统计柱状图
├── 07_module_3.4_sensitivity_mu0.png          # μ₀ 敏感性曲线图
└── 08_module_3.4_sensitivity_MIC.png          # MIC 敏感性曲线图

results/
├── module_3.4_verification_summary.csv        # 统计检验结果 (t-stat, p-value, 结论)
├── module_3.4_sensitivity_analysis.csv        # 敏感性指标 (对 μ₀, MIC, n 的影响度)
└── module_3.4_final_report.txt                # 项目总报告 (汇总 3.1, 3.2, 3.4 结果)
```

### 5\. `code/utils.py` - 通用工具函数库

**文件类型**: 工具函数库 (Python)  
**行数**: \~150行  
**依赖**: NumPy, Pandas, Matplotlib, SciPy, Scikit-learn

#### 功能概述

这是一个可重用的工具箱，包含项目所有脚本都会用到的通用函数，确保核心建模脚本 (`01` 到 `04`) 的简洁性和可维护性。

#### 核心函数

| 函数名 | 功能描述 | 核心算法 / 依赖 |
| :--- | :--- | :--- |
| `load_data(filename, folder)` | 安全地从指定文件夹加载 CSV 文件。 | `os.path.join`, `pd.read_csv` |
| `save_results(df, filename)` | 保存 Pandas DataFrame 到 `results/` 文件夹。 | `df.to_csv` |
| `save_figure(fig, filename)` | 保存 Matplotlib 图表到 `figures/` 文件夹。 | `fig.savefig` |
| `calculate_statistics(data)` | 计算描述性统计量（均值、标准差、变异系数）。 | `np.mean`, `np.std`, `scipy.stats` |
| `calculate_model_metrics(y_true, y_pred)` | **模型评估函数**，计算拟合优度指标。 | `r2_score`, `mean_squared_error` |
| `perform_ttest(data1, data2)` | 执行双样本 t-检验，用于比较两组数据的差异。 | `scipy.stats.ttest_ind` |
| `print_section/print_subsection` | 格式化控制台输出，增强可读性。 | Python `print` |

**模型评估指标 ($\text{R}^2$) 代码**

```python
def calculate_model_metrics(y_true, y_pred, model_name='Model'):
    r"""
    计算模型评估指标 R², RMSE, MAE
    
    R² (R-squared): $$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
    """
    R2 = r2_score(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MAE = mean_absolute_error(y_true, y_pred)
    
    # ... (返回字典)
```

### 6\. `README.md` - 项目概览

**文件类型**: Markdown (项目主页)  
**功能概述**: 提供项目的**高层概览**，包括项目背景、团队分工、模块结构、文件结构和快速启动指南。这是项目公开展示的首要文档。

#### 核心内容

  - **项目名称**: TasAnchor项目 - 功能测试模块建模 (2025 iGEM SCU-China 复现)
  - **团队分工**: 明确 A, B (本项目), C 组的任务范围。
  - **模块结构**: 简要介绍模块 3.1 到 3.4 的生物学目标。
  - **快速启动**: 提供一行命令运行数据提取和模型的指南。

### 7\. `WORKFLOW.md` - 工作流程指南

**文件类型**: Markdown (操作指南)  
**功能概述**: 详细的**操作手册**，指导用户或新成员从环境设置到最终结果生成的完整步骤。

#### 核心内容

  - **环境准备**: 如何安装 `requirements.txt` 中的依赖。
  - **运行步骤**: 强调必须按照 `01` -\> `02` -\> `03` -\> `04` 的顺序运行脚本。
  - **数据更新流程**: 如果 iGEM Wiki 数据更新，如何仅修改 `01_data_extraction.py`。
  - **常见问题排查**: 针对 `ModuleNotFoundError`, `curve_fit` 失败等常见问题提供解决方案。

### 8\. `requirements.txt` - 依赖清单

**文件类型**: 纯文本文件  
**功能概述**: 列出运行所有 Python 脚本所需的全部第三方库及其最低版本。用于环境重现。

#### 核心依赖

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=1.0.0
# ... (其他可选依赖如 jupyter, seaborn)
```

-----

---

## 8. 💻 notebooks/ (Jupyter Notebooks - 交互式分析)

Notebooks 文件夹用于快速验证模型、调整初始参数 $p0$ 和进行交互式数据探索 (EDA)。所有核心逻辑最终由 `code/` 文件夹中的 Python 脚本运行。

| 文件名 | 对应脚本 | 功能描述 | 关键用途 |
| :--- | :--- | :--- | :--- |
| `01_Data_Exploration.ipynb` | `01_data_extraction.py` | 交互式加载、清洗并可视化原始生长、荧光和吸附数据，进行初步的数据趋势和质量检查。 | 确认数据质量，确定初步 $p0$ 范围。 |
| `02_Primary_Model_Fitting.ipynb` | `02_module_3.1.py` | 集中测试 Modified Gompertz 和 Hill 方程的拟合效果。用于参数精调和 $R^2$ 验证。 | 快速生成拟合图，确定最优 Gompertz 和 Hill 参数。 |
| `03_Secondary_Model.ipynb` | `03_module_3.2.py` | 专注于 Langmuir/Freundlich 等温线和二级生长抑制模型的拟合，用于模型选择。 | 确定 $q_{max}$ 和 $MIC$ 参数的精确值。 |
| `04_Sensitivity_Analysis.ipynb` | `04_module_3.3_3.4.py` | 交互式地运行参数扰动分析，并计算实际应用所需的菌体干重，验证模型的鲁棒性。 | 敏感性曲线，实际应用场景所需生物量（g/L）。 |

---

## 9. 📊 生成的输出文件 (figures/, results/)

运行所有脚本 (`01_` 至 `04_`) 后，项目会生成以下图表和结果数据文件。

### 🖼️ figures/ (图表文件)

| 文件名 | 来源脚本 | 描述 |
| :--- | :--- | :--- |
| `00_data_overview.png` | `01_data_extraction.py` | 原始数据的总览，例如生物膜形成的OD570柱状图。 |
| `01_module_3.1_growth_fit.png` | `02_module_3.1.py` | Modified Gompertz 模型拟合曲线，对比不同 $Cd^{2+}$ 浓度。 |
| `02_module_3.1_hill_fit.png` | `02_module_3.1.py` | Hill 方程拟合曲线，展示荧光响应的剂量-效应关系。 |
| `03_module_3.2_adsorption_isotherm.png` | `03_module_3.2.py` | Langmuir/Freundlich 模型拟合对比图，显示最优拟合曲线。 |
| `04_module_3.2_adsorption_cycles.png` | `03_module_3.2.py` | 菌株在多次吸附-解吸循环中的性能图。 |
| `05_module_3.2_secondary_growth_model.png` | `03_module_3.2.py` | $\mu_{max}$ vs $Cd^{2+}$ 浓度（二级抑制模型）拟合曲线。 |
| `06_module_3.4_verification_ttest.png` | `04_module_3.3_3.4.py` | 模块 3.4 验证实验结果，柱状图及显著性标记。 |
| `07_module_3.4_sensitivity_mu0.png` | `04_module_3.3_3.4.py` | $\mu_0$ 扰动下的模型敏感性分析图。 |
| `08_module_3.4_sensitivity_MIC.png` | `04_module_3.3_3.4.py` | $MIC$ 扰动下的模型敏感性分析图。 |

### 📝 results/ (建模结果文件)

| 文件名 | 来源脚本 | 描述 |
| :--- | :--- | :--- |
| `data_extraction_report.txt` | `01_data_extraction.py` | 原始数据提取的统计摘要和文件校验信息。 |
| `module_3.1_gompertz_parameters.csv` | `02_module_3.1.py` | 所有 Modified Gompertz 模型的拟合参数 ($\mu_{max}, \lambda, A$ 等) 及 $R^2$。|
| `module_3.1_hill_parameters.csv` | `02_module_3.1.py` | Hill 方程的关键参数 ($EC_{50}, F_{max}, n$) 及 $R^2$。 |
| `module_3.2_isotherm_parameters.csv` | `03_module_3.2.py` | Langmuir 和 Freundlich 模型的拟合参数 ($q_{max}, K_L, K_F, 1/n$)。 |
| `module_3.2_secondary_model_parameters.csv` | `03_module_3.2.py` | 二级生长抑制模型的 $\mu_0, MIC, n$ 参数。 |
| `module_3.4_verification_summary.csv` | `04_module_3.3_3.4.py` | 模块 3.4 $t$-检验结果、吸附效率平均值和标准差。 |
| `module_3.4_application_scenarios.csv` | `04_module_3.3_3.4.py` | 基于 $q_{max}$，预测不同废水体积和浓度下所需的菌体干重。 |
| `final_summary_report.txt` | `04_module_3.3_3.4.py` | **项目最终总结报告**，整合 3.1-3.4 的关键结果、结论和应用建议。|