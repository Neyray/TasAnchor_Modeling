"""
TasAnchor Project - Module 3.2 Modeling
模块3.2：镉离子吸附模块建模

目标：
1. 建立吸附等温线模型 (Langmuir vs Freundlich)
2. 分析吸附-解吸循环性能
3. 建立二级生长模型（μ_max vs Cd²⁺浓度）

作者: B组成员
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print_section("模块 3.2：镉离子吸附模块建模", 70)

# ============================================================================
# 第一部分：吸附等温线建模
# ============================================================================
print_subsection("第一部分：吸附等温线模型", 70)

# 从吸附-解吸数据计算平衡吸附量
print("\n[1] 加载吸附-解吸数据...")
df_ads_9mg = load_data('module_3.2_adsorption_9mg.csv')
df_ads_5mg = load_data('module_3.2_adsorption_5mg.csv')

# 计算吸附量 q_e (mg/g)
# 假设使用 1g 干重菌体处理 1L溶液
def calculate_qe(C0, Ce, V=1.0, m=1.0):
    """计算吸附量 q_e = (C0 - Ce) * V / m"""
    return (C0 - Ce) * V / m

# 提取吸附步骤的数据
ads_data_9mg = df_ads_9mg[df_ads_9mg['process_type'] == 'Adsorption']
ads_data_5mg = df_ads_5mg[df_ads_5mg['process_type'] == 'Adsorption']

# 平衡浓度和吸附量
Ce_values = []
qe_values = []

for _, row in ads_data_9mg.iterrows():
    Ce = row['Cd_remaining_mg_L']
    qe = calculate_qe(9.0, Ce)
    Ce_values.append(Ce)
    qe_values.append(qe)

for _, row in ads_data_5mg.iterrows():
    Ce = row['Cd_remaining_mg_L']
    qe = calculate_qe(5.0, Ce)
    Ce_values.append(Ce)
    qe_values.append(qe)

Ce_values = np.array(Ce_values)
qe_values = np.array(qe_values)

print(f"\n平衡吸附数据:")
for ce, qe in zip(Ce_values, qe_values):
    print(f"  Ce = {ce:.2f} mg/L → qe = {qe:.2f} mg/g")

# Langmuir模型
def langmuir_model(Ce, q_max, K_L):
    """
    Langmuir等温吸附模型
    
    参数：
    - q_max: 最大吸附容量 (mg/g)
    - K_L: Langmuir常数 (L/mg)，越大吸附亲和力越强
    
    假设：单层吸附，吸附位点均匀
    """
    return q_max * K_L * Ce / (1 + K_L * Ce)

# Freundlich模型
def freundlich_model(Ce, K_F, n):
    """
    Freundlich等温吸附模型
    
    参数：
    - K_F: Freundlich常数
    - 1/n: 吸附强度指数（0.1-1表示易吸附）
    
    假设：多层吸附，表面不均匀
    """
    return K_F * Ce**(1/n)

# 拟合Langmuir
print("\n[2] 拟合Langmuir模型...")
p0_lang = [10, 1]
popt_lang, _ = curve_fit(langmuir_model, Ce_values, qe_values, p0=p0_lang)
qe_pred_lang = langmuir_model(Ce_values, *popt_lang)
metrics_lang = calculate_model_metrics(qe_values, qe_pred_lang, 'Langmuir')

langmuir_params = {
    'q_max_mg_g': round(popt_lang[0], 3),
    'K_L_L_mg': round(popt_lang[1], 4),
    'R2': round(metrics_lang['R2'], 4),
}
print_dict(langmuir_params, "Langmuir参数")

# 拟合Freundlich
print("\n[3] 拟合Freundlich模型...")
p0_freu = [5, 2]
popt_freu, _ = curve_fit(freundlich_model, Ce_values, qe_values, p0=p0_freu)
qe_pred_freu = freundlich_model(Ce_values, *popt_freu)
metrics_freu = calculate_model_metrics(qe_values, qe_pred_freu, 'Freundlich')

freundlich_params = {
    'K_F': round(popt_freu[0], 3),
    '1_over_n': round(1/popt_freu[1], 3),
    'R2': round(metrics_freu['R2'], 4),
}
print_dict(freundlich_params, "Freundlich参数")

# 绘图对比
fig1, ax = plt.subplots(figsize=(10, 7))

ax.plot(Ce_values, qe_values, 'o', markersize=12, label='实验数据', 
        color='#D62828', zorder=3)

Ce_fine = np.linspace(0.01, max(Ce_values)*1.5, 200)
ax.plot(Ce_fine, langmuir_model(Ce_fine, *popt_lang), '-', linewidth=2.5,
        label=f'Langmuir (R²={metrics_lang["R2"]:.3f})', color='#003049')
ax.plot(Ce_fine, freundlich_model(Ce_fine, *popt_freu), '--', linewidth=2.5,
        label=f'Freundlich (R²={metrics_freu["R2"]:.3f})', color='#F77F00')

ax.set_xlabel('平衡浓度 Ce (mg/L)', fontsize=13, fontweight='bold')
ax.set_ylabel('吸附量 qe (mg/g)', fontsize=13, fontweight='bold')
ax.set_title('吸附等温线模型对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 标注最大吸附容量
ax.axhline(popt_lang[0], color='gray', linestyle=':', alpha=0.7)
ax.text(max(Ce_values)*0.7, popt_lang[0]*1.05, 
        f'q_max = {popt_lang[0]:.2f} mg/g', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
save_figure(fig1, '03_module_3.2_adsorption_isotherm.png')
plt.close()

# 保存参数
isotherm_params = {
    'Model': ['Langmuir', 'Freundlich'],
    'q_max_or_KF': [langmuir_params['q_max_mg_g'], freundlich_params['K_F']],
    'K_L_or_1_over_n': [langmuir_params['K_L_L_mg'], freundlich_params['1_over_n']],
    'R2': [langmuir_params['R2'], freundlich_params['R2']],
}
df_isotherm = pd.DataFrame(isotherm_params)
save_results(df_isotherm, 'module_3.2_isotherm_parameters.csv')

# ============================================================================
# 第二部分：吸附-解吸循环分析
# ============================================================================
print_subsection("第二部分：吸附-解吸循环性能", 70)

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# 9 mg/L循环
ax1 = axes[0]
ads_9 = df_ads_9mg[df_ads_9mg['process_type'] == 'Adsorption']
des_9 = df_ads_9mg[df_ads_9mg['process_type'] == 'Desorption']

ax1.plot(ads_9['cycle_number'], ads_9['Cd_remaining_mg_L'], 
         'o-', markersize=10, linewidth=2, label='吸附后', color='#2E86AB')
ax1.plot(des_9['cycle_number'], des_9['Cd_remaining_mg_L'], 
         's-', markersize=10, linewidth=2, label='解吸后', color='#D62828')
ax1.axhline(9.0, linestyle='--', color='gray', alpha=0.7, label='初始浓度')

ax1.set_xlabel('循环次数', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cd²⁺浓度 (mg/L)', fontsize=12, fontweight='bold')
ax1.set_title('9 mg/L 吸附-解吸循环', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([1, 2, 3])

# 5 mg/L循环
ax2 = axes[1]
ads_5 = df_ads_5mg[df_ads_5mg['process_type'] == 'Adsorption']
des_5 = df_ads_5mg[df_ads_5mg['process_type'] == 'Desorption']

ax2.plot(ads_5['cycle_number'], ads_5['Cd_remaining_mg_L'], 
         'o-', markersize=10, linewidth=2, label='吸附后', color='#2E86AB')
ax2.plot(des_5['cycle_number'], des_5['Cd_remaining_mg_L'], 
         's-', markersize=10, linewidth=2, label='解吸后', color='#D62828')
ax2.axhline(5.0, linestyle='--', color='gray', alpha=0.7, label='初始浓度')

ax2.set_xlabel('循环次数', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cd²⁺浓度 (mg/L)', fontsize=12, fontweight='bold')
ax2.set_title('5 mg/L 吸附-解吸循环', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks([1, 2, 3])

plt.tight_layout()
save_figure(fig2, '04_module_3.2_adsorption_cycles.png')
plt.close()

# 计算循环效率
print("\n[4] 计算循环效率...")
cycle_efficiencies_9mg = ads_9['removal_efficiency_%'].values
cycle_efficiencies_5mg = ads_5['removal_efficiency_%'].values

print(f"\n9 mg/L 循环去除效率:")
for i, eff in enumerate(cycle_efficiencies_9mg, 1):
    print(f"  循环{i}: {eff:.1f}%")

print(f"\n5 mg/L 循环去除效率:")
for i, eff in enumerate(cycle_efficiencies_5mg, 1):
    print(f"  循环{i}: {eff:.1f}%")

# ============================================================================
# 第三部分：二级生长模型
# ============================================================================
print_subsection("第三部分：二级生长模型 (μ_max vs Cd²⁺)", 70)

# 从模块3.1读取Gompertz参数
print("\n[5] 加载Gompertz参数...")
df_gompertz = pd.read_csv('results/module_3.1_gompertz_parameters.csv')

cd_conc = df_gompertz['Cd_concentration_mg_L'].values
mu_max_values = df_gompertz['mu_max_h-1'].values

# 二级模型：幂函数抑制
def secondary_growth_model(Cd, mu0, MIC, n):
    """
    二级生长模型
    
    μ_max(Cd) = μ₀ × [1 - (Cd/MIC)^n]
    
    参数：
    - μ₀: 无Cd²⁺时的最大生长速率
    - MIC: 最小抑制浓度 (mg/L)
    - n: 抑制指数
    """
    return mu0 * (1 - (Cd / MIC)**n)

print("\n[6] 拟合二级生长模型...")
p0_sec = [max(mu_max_values), 50, 2]
popt_sec, _ = curve_fit(secondary_growth_model, cd_conc, mu_max_values, p0=p0_sec)

mu_pred = secondary_growth_model(cd_conc, *popt_sec)
metrics_sec = calculate_model_metrics(mu_max_values, mu_pred, 'Secondary')

secondary_params = {
    'mu0_h-1': round(popt_sec[0], 4),
    'MIC_mg_L': round(popt_sec[1], 2),
    'n': round(popt_sec[2], 3),
    'R2': round(metrics_sec['R2'], 4),
}
print_dict(secondary_params, "二级模型参数")

# 绘图
fig3, ax = plt.subplots(figsize=(10, 7))

ax.plot(cd_conc, mu_max_values, 'o', markersize=12, label='实测数据', 
        color='#D62828', zorder=3)

cd_fine = np.linspace(0, 50, 200)
mu_fit = secondary_growth_model(cd_fine, *popt_sec)
ax.plot(cd_fine, mu_fit, '-', linewidth=3, 
        label=f'二级模型 (R²={metrics_sec["R2"]:.3f})', color='#003049')

# 标注典型废水浓度
ax.axvline(30, color='orange', linestyle='--', linewidth=2, label='典型废水浓度')
mu_at_30 = secondary_growth_model(30, *popt_sec)
ax.plot(30, mu_at_30, 's', markersize=12, color='orange', zorder=4)
ax.text(32, mu_at_30, f'{mu_at_30:.3f} h⁻¹', fontsize=10)

ax.set_xlabel('Cd²⁺浓度 (mg/L)', fontsize=13, fontweight='bold')
ax.set_ylabel('最大比生长速率 μ_max (h⁻¹)', fontsize=13, fontweight='bold')
ax.set_title('Cd²⁺对生长速率的二级抑制模型', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig3, '05_module_3.2_secondary_growth_model.png')
plt.close()

save_results(secondary_params, 'module_3.2_secondary_model_parameters.csv')

# ============================================================================
# 生成报告
# ============================================================================
print_section("生成模块3.2总结报告", 70)

report_path = 'results/module_3.2_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("TasAnchor 项目 - 模块3.2建模报告\n")
    f.write("镉离子吸附模块 (tasA-smtA)\n")
    f.write("=" * 70 + "\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("一、吸附等温线模型\n")
    f.write("-" * 70 + "\n")
    f.write(df_isotherm.to_string(index=False))
    f.write(f"\n\n推荐模型: Langmuir (R² = {langmuir_params['R2']:.4f})\n")
    f.write(f"最大吸附容量 q_max = {langmuir_params['q_max_mg_g']:.2f} mg/g\n")
    
    f.write("\n\n二、吸附-解吸循环性能\n")
    f.write("-" * 70 + "\n")
    f.write(f"9 mg/L 初始浓度:\n")
    f.write(f"  第1次循环去除率: {cycle_efficiencies_9mg[0]:.1f}%\n")
    f.write(f"  第3次循环去除率: {cycle_efficiencies_9mg[2]:.1f}%\n")
    f.write(f"  循环稳定性: {'良好' if cycle_efficiencies_9mg[2] > 60 else '需改进'}\n")
    
    f.write("\n\n三、二级生长模型\n")
    f.write("-" * 70 + "\n")
    f.write(f"μ_max(Cd) = {secondary_params['mu0_h-1']:.3f} × [1 - (Cd/{secondary_params['MIC_mg_L']:.1f})^{secondary_params['n']:.2f}]\n")
    f.write(f"拟合优度 R² = {secondary_params['R2']:.4f}\n")
    f.write(f"\n预测: 在30 mg/L Cd²⁺下，μ_max ≈ {mu_at_30:.3f} h⁻¹\n")
    f.write(f"相比无Cd²⁺条件，生长速率下降约 {(1-mu_at_30/popt_sec[0])*100:.1f}%\n")

print(f"✓ 已保存报告: {report_path}")

print_section("模块3.2建模完成", 70)
print("\n下一步: 运行 python code/04_module_3.3_3.4.py 进行综合分析")