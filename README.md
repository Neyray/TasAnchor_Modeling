# TasAnchor项目 - 功能测试模块建模

## 项目简介
2025年iGEM SCU-China项目复现 - 功能测试模块的数学建模与数据分析

## 团队分工
- **A组**: 蛋白质建模和粘附仿真
- **B组**: 功能测试（本项目）
- **C组**: 安全测试

## 模块说明
- **3.1**: 镉离子感应模块 (epcadR-pcadR-mcherry)
- **3.2**: 镉离子吸附模块 (tasA-smtA)
- **3.3**: 感应-粘附复合模块
- **3.4**: 吸附-粘附功能验证

## 文件结构
```
TasAnchor_Modeling/
├── data/                 # 数据文件
├── code/                 # Python脚本
├── figures/              # 生成的图表
├── results/              # 模型结果
└── README.md             # 本文件
```

## 使用方法
1. 激活虚拟环境: `venv\Scripts\activate` (Windows) 或 `source venv/bin/activate` (Mac/Linux)
2. 运行数据提取: `python code/01_data_extraction.py`
3. 依次运行建模脚本: `02_module_3.1.py`, `03_module_3.2.py` 等

## 数据来源
https://2025.igem.wiki/scu-china/results
```

#### **.gitignore**
打开 `.gitignore`，粘贴：
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Data files (optional)
# data/raw/*.csv
# data/processed/*.csv

# OS
.DS_Store
Thumbs.db