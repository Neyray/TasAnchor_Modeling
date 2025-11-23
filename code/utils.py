"""
TasAnchor Project - Utility Functions
通用工具函数库

包含的功能:
- 数据加载和预处理
- 统计分析
- 绘图样式设置
- 模型评估指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 数据加载函数
# ============================================================================

def load_data(filename, folder='data/raw'):
    """
    加载CSV数据文件
    
    Parameters:
    -----------
    filename : str
        文件名（可以包含或不包含.csv后缀）
    folder : str
        数据文件夹路径
    
    Returns:
    --------
    df : pandas.DataFrame
        加载的数据
    """
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ 已加载数据: {filepath} ({len(df)} 行, {len(df.columns)} 列)")
    return df


def save_results(data, filename, folder='results'):
    """
    保存结果数据
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        要保存的数据
    filename : str
        输出文件名
    folder : str
        输出文件夹
    """
    os.makedirs(folder, exist_ok=True)
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = os.path.join(folder, filename)
    
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    data.to_csv(filepath, index=False)
    print(f"✓ 已保存结果: {filepath}")


# ============================================================================
# 统计分析函数
# ============================================================================

def calculate_statistics(data):
    """
    计算描述性统计量
    
    Parameters:
    -----------
    data : array-like
        输入数据
    
    Returns:
    --------
    stats_dict : dict
        统计量字典
    """
    data = np.array(data)
    
    stats_dict = {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'cv_%': (np.std(data, ddof=1) / np.mean(data)) * 100,  # 变异系数
    }
    
    return stats_dict


def calculate_model_metrics(y_true, y_pred, model_name='Model'):
    """
    计算模型评估指标
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    model_name : str
        模型名称
    
    Returns:
    --------
    metrics : dict
        评估指标字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 去除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {
        'model': model_name,
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE_%': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }
    
    return metrics


def perform_ttest(group1, group2, alpha=0.05):
    """
    执行独立样本t检验
    
    Parameters:
    -----------
    group1, group2 : array-like
        两组数据
    alpha : float
        显著性水平
    
    Returns:
    --------
    result : dict
        t检验结果
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
    }
    
    return result


# ============================================================================
# 绘图样式设置
# ============================================================================

def set_plot_style(style='seaborn'):
    """
    设置Matplotlib绘图样式
    
    Parameters:
    -----------
    style : str
        样式名称 ('seaborn', 'default', 'scientific')
    """
    if style == 'seaborn':
        plt.style.use('seaborn-v0_8-darkgrid')
    elif style == 'scientific':
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.8
    else:
        plt.style.use('default')
    
    print(f"✓ 已设置绘图样式: {style}")


def save_figure(fig, filename, folder='figures', dpi=300):
    """
    保存图表
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        图表对象
    filename : str
        输出文件名
    folder : str
        输出文件夹
    dpi : int
        分辨率
    """
    os.makedirs(folder, exist_ok=True)
    
    if not (filename.endswith('.png') or filename.endswith('.pdf')):
        filename += '.png'
    
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ 已保存图表: {filepath}")


# ============================================================================
# 数据转换函数
# ============================================================================

def concentration_unit_converter(value, from_unit='mg/L', to_unit='mmol/L', molecular_weight=112.41):
    """
    浓度单位转换（针对Cd²⁺）
    
    Parameters:
    -----------
    value : float or array
        浓度值
    from_unit : str
        原始单位 ('mg/L', 'mmol/L', 'g/L')
    to_unit : str
        目标单位
    molecular_weight : float
        分子量 (Cd²⁺ = 112.41 g/mol)
    
    Returns:
    --------
    converted : float or array
        转换后的浓度
    """
    # 首先转换为mg/L
    if from_unit == 'mg/L':
        mg_per_L = value
    elif from_unit == 'mmol/L':
        mg_per_L = value * molecular_weight
    elif from_unit == 'g/L':
        mg_per_L = value * 1000
    else:
        raise ValueError(f"不支持的单位: {from_unit}")
    
    # 从mg/L转换为目标单位
    if to_unit == 'mg/L':
        return mg_per_L
    elif to_unit == 'mmol/L':
        return mg_per_L / molecular_weight
    elif to_unit == 'g/L':
        return mg_per_L / 1000
    else:
        raise ValueError(f"不支持的单位: {to_unit}")


# ============================================================================
# 模型诊断函数
# ============================================================================

def residual_analysis(y_true, y_pred):
    """
    残差分析
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    residuals : dict
        残差统计信息
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals_array = y_true - y_pred
    
    # Shapiro-Wilk正态性检验
    _, p_normal = stats.shapiro(residuals_array)
    
    residuals = {
        'residuals': residuals_array,
        'mean_residual': np.mean(residuals_array),
        'std_residual': np.std(residuals_array),
        'normal_test_p': p_normal,
        'is_normal': p_normal > 0.05,
    }
    
    return residuals


def plot_residuals(y_true, y_pred, title='Residual Analysis'):
    """
    绘制残差图
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    title : str
        图表标题
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图表对象
    """
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 残差vs预测值
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # 残差直方图
    axes[1].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# 打印工具
# ============================================================================

def print_section(title, width=70):
    """打印章节标题"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title, width=70):
    """打印小节标题"""
    print("\n" + title)
    print("-" * width)


def print_dict(data_dict, title=None):
    """格式化打印字典"""
    if title:
        print(f"\n{title}:")
    for key, value in data_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


# ============================================================================
# 测试函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("工具函数模块测试")
    print("=" * 70)
    
    # 测试统计函数
    test_data = [1.2, 1.5, 1.3, 1.6, 1.4]
    stats = calculate_statistics(test_data)
    print_dict(stats, "描述性统计")
    
    # 测试单位转换
    cd_mg_L = 9.0
    cd_mmol_L = concentration_unit_converter(cd_mg_L, 'mg/L', 'mmol/L')
    print(f"\n单位转换: {cd_mg_L} mg/L = {cd_mmol_L:.4f} mmol/L")
    
    # 测试模型评估
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = calculate_model_metrics(y_true, y_pred, 'Test Model')
    print_dict(metrics, "模型评估指标")
    
    print("\n✓ 所有测试通过！")
    print("=" * 70)