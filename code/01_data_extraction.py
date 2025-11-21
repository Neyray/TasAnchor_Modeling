"""
TasAnchor Project - Data Extraction and Organization
功能测试模块数据提取脚本 (VSCode版本)

作者: B组成员
日期: 2024
项目: 2025 iGEM SCU-China 复现

使用说明:
1. 确保已激活虚拟环境
2. 从 https://2025.igem.wiki/scu-china/results 提取图表数据
3. 替换下方的示例数据为真实数据
4. 运行: python code/01_data_extraction.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置工作目录为项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# 设置中文字体（根据操作系统自动选择）
if sys.platform.startswith('win'):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif sys.platform.startswith('darwin'):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 创建必要的文件夹
folders = ['data/raw', 'data/processed', 'figures', 'results']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("=" * 70)
print("TasAnchor 项目 - 数据提取与整理")
print("=" * 70)
print(f"工作目录: {os.getcwd()}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================================
# 模块 3.1: 镉离子感应模块 (epcadR-pcadR-mcherry)
# ============================================================================
print("\n[模块 3.1] 镉离子感应模块")
print("-" * 70)

# ------------------------------------------------------------------------
# Fig 3.1-4: 不同Cd²⁺浓度下的生长曲线
# 数据来源: 从SCU-China Results页面提取
# 注意: 这些是示例数据，需要根据实际图表替换！
# ------------------------------------------------------------------------
print("\n提取 Fig 3.1-4: 生长曲线数据...")

growth_curve_data = {
    'time_h': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # 时间(小时)
    
    # 不同Cd²⁺浓度下的OD600值
    'OD600_0mg_L': [0.05, 0.08, 0.15, 0.35, 0.65, 0.95, 1.20, 1.35, 1.40, 1.42, 1.42],
    'OD600_0.25mg_L': [0.05, 0.08, 0.14, 0.33, 0.62, 0.90, 1.15, 1.30, 1.35, 1.37, 1.37],
    'OD600_0.5mg_L': [0.05, 0.07, 0.13, 0.30, 0.58, 0.85, 1.10, 1.25, 1.30, 1.32, 1.32],
    'OD600_1mg_L': [0.05, 0.07, 0.12, 0.27, 0.52, 0.78, 1.00, 1.15, 1.20, 1.22, 1.22],
    'OD600_1.5mg_L': [0.05, 0.06, 0.10, 0.22, 0.42, 0.65, 0.85, 1.00, 1.05, 1.07, 1.07],
}

df_growth = pd.DataFrame(growth_curve_data)
print(f"✓ 生长曲线数据: {len(df_growth)} 个时间点, {len(df_growth.columns)-1} 个浓度组")
print(df_growth.head(3))

# ------------------------------------------------------------------------
# Fig 3.1-5: 荧光强度随时间和浓度的变化
# 注意: 原文中浓度单位是 mmol/L
# ------------------------------------------------------------------------
print("\n提取 Fig 3.1-5: 荧光强度数据...")

fluorescence_data = {
    'time_h': [0, 5, 10, 15, 20, 25, 30],  # 时间(小时)
    
    # 不同Cd²⁺浓度下的荧光强度 (FU - Fluorescence Units)
    'FL_0mM': [10, 12, 15, 18, 20, 22, 23],        # 0 mmol/L (对照组)
    'FL_1mM': [10, 25, 55, 95, 140, 180, 175],     # 1 mmol/L
    'FL_2mM': [10, 35, 80, 145, 210, 260, 250],    # 2 mmol/L (最强响应)
}

df_fluorescence = pd.DataFrame(fluorescence_data)
print(f"✓ 荧光强度数据: {len(df_fluorescence)} 个时间点, {len(df_fluorescence.columns)-1} 个浓度组")
print(df_fluorescence.head(3))

# 保存原始数据
df_growth.to_csv('data/raw/module_3.1_growth_curves.csv', index=False)
df_fluorescence.to_csv('data/raw/module_3.1_fluorescence.csv', index=False)
print("\n✓ 已保存: data/raw/module_3.1_growth_curves.csv")
print("✓ 已保存: data/raw/module_3.1_fluorescence.csv")

# ============================================================================
# 模块 3.2: 镉离子吸附模块 (tasA-smtA)
# ============================================================================
print("\n[模块 3.2] 镉离子吸附模块")
print("-" * 70)

# ------------------------------------------------------------------------
# Fig 3.2-4: 生物膜形成 (结晶紫染色 OD570)
# ------------------------------------------------------------------------
print("\n提取 Fig 3.2-4: 生物膜形成数据...")

biofilm_data = {
    'Cd_concentration_mg_L': [0, 3, 6, 9],  # Cd²⁺浓度 (mg/L)
    'OD570_mean': [0.85, 0.82, 0.80, 0.78],  # 平均OD570值
    'OD570_std': [0.05, 0.06, 0.05, 0.07],   # 标准差
}

df_biofilm = pd.DataFrame(biofilm_data)
print(f"✓ 生物膜形成数据: {len(df_biofilm)} 个浓度点")
print(df_biofilm)

# ------------------------------------------------------------------------
# Fig 3.2-5: 吸附-解吸循环曲线
# ------------------------------------------------------------------------
print("\n提取 Fig 3.2-5: 吸附-解吸循环数据...")

# 9 mg/L Cd²⁺溶液的吸附-解吸数据
adsorption_9mg = {
    'cycle_step': ['Ads1', 'Des1', 'Ads2', 'Des2', 'Ads3', 'Des3'],
    'cycle_number': [1, 1, 2, 2, 3, 3],
    'process_type': ['Adsorption', 'Desorption', 'Adsorption', 'Desorption', 'Adsorption', 'Desorption'],
    'Cd_remaining_mg_L': [0.1, 6.5, 1.2, 7.0, 2.8, 7.5],  # 剩余Cd²⁺浓度
    'Cd_initial_mg_L': 9.0,  # 初始浓度
}

df_ads_9mg = pd.DataFrame(adsorption_9mg)
df_ads_9mg['removal_efficiency_%'] = ((df_ads_9mg['Cd_initial_mg_L'] - df_ads_9mg['Cd_remaining_mg_L']) / 
                                       df_ads_9mg['Cd_initial_mg_L'] * 100)
print(f"✓ 9 mg/L 吸附-解吸数据: {len(df_ads_9mg)} 个步骤")
print(df_ads_9mg[['cycle_step', 'Cd_remaining_mg_L', 'removal_efficiency_%']])

# 5 mg/L Cd²⁺溶液的吸附-解吸数据
adsorption_5mg = {
    'cycle_step': ['Ads1', 'Des1', 'Ads2', 'Des2', 'Ads3', 'Des3'],
    'cycle_number': [1, 1, 2, 2, 3, 3],
    'process_type': ['Adsorption', 'Desorption', 'Adsorption', 'Desorption', 'Adsorption', 'Desorption'],
    'Cd_remaining_mg_L': [3.2, 0.0, 2.5, 0.0, 1.8, 0.0],
    'Cd_initial_mg_L': 5.0,
}

df_ads_5mg = pd.DataFrame(adsorption_5mg)
df_ads_5mg['removal_efficiency_%'] = ((df_ads_5mg['Cd_initial_mg_L'] - df_ads_5mg['Cd_remaining_mg_L']) / 
                                       df_ads_5mg['Cd_initial_mg_L'] * 100)
print(f"\n✓ 5 mg/L 吸附-解吸数据: {len(df_ads_5mg)} 个步骤")
print(df_ads_5mg[['cycle_step', 'Cd_remaining_mg_L', 'removal_efficiency_%']])

# 保存数据
df_biofilm.to_csv('data/raw/module_3.2_biofilm.csv', index=False)
df_ads_9mg.to_csv('data/raw/module_3.2_adsorption_9mg.csv', index=False)
df_ads_5mg.to_csv('data/raw/module_3.2_adsorption_5mg.csv', index=False)
print("\n✓ 已保存: data/raw/module_3.2_biofilm.csv")
print("✓ 已保存: data/raw/module_3.2_adsorption_9mg.csv")
print("✓ 已保存: data/raw/module_3.2_adsorption_5mg.csv")

# ============================================================================
# 模块 3.3: 感应-粘附复合模块
# ============================================================================
print("\n[模块 3.3] 感应-粘附复合模块")
print("-" * 70)
print("注意: 该模块主要是定性验证（荧光显微镜图像）")
print("量化数据将从模块3.1和3.2综合分析得出")

# 记录实验条件
module_3_3_info = {
    'experiment': 'Sensing-Adhesion Module',
    'plasmid': 'pHT01-p43-epcadR-mcherry-p43-tasA-mfp5',
    'substrate': 'Polystyrene microspheres',
    'incubation_time_h': 30,
    'Cd_concentration_mM': 2,
    'induction_time_h': 16,
    'observation': 'Red fluorescence under microscope',
}

df_module_3_3 = pd.DataFrame([module_3_3_info])
df_module_3_3.to_csv('data/raw/module_3.3_experimental_conditions.csv', index=False)
print("✓ 已保存: data/raw/module_3.3_experimental_conditions.csv")

# ============================================================================
# 模块 3.4: 吸附-粘附功能验证
# ============================================================================
print("\n[模块 3.4] 吸附-粘附功能验证")
print("-" * 70)

# ------------------------------------------------------------------------
# Fig 3.4-1: 吸附效率对比
# ------------------------------------------------------------------------
print("\n提取 Fig 3.4-1: 吸附效率验证数据...")

verification_data = {
    'group': ['Control (ΔtasAΔsinR)', 'Experimental (tasA-smtA)'],
    'strain_type': ['Knockout only', 'With adsorption module'],
    'adsorption_efficiency_%': [15.2, 42.8],  # 从柱状图估算
    'std_%': [3.5, 4.2],  # 误差棒估算
    'Cd_initial_mg_L': 9.0,
    'incubation_time_h': 15,
}

df_verification = pd.DataFrame(verification_data)
print(f"✓ 吸附效率验证数据: {len(df_verification)} 组对比")
print(df_verification[['group', 'adsorption_efficiency_%', 'std_%']])

df_verification.to_csv('data/raw/module_3.4_verification.csv', index=False)
print("\n✓ 已保存: data/raw/module_3.4_verification.csv")

# ============================================================================
# 数据汇总统计
# ============================================================================
print("\n" + "=" * 70)
print("数据提取汇总")
print("=" * 70)

data_summary = {
    'Module': ['3.1 Sensing', '3.1 Sensing', '3.2 Adsorption', '3.2 Adsorption', 
               '3.2 Adsorption', '3.3 Complex', '3.4 Verification'],
    'Data_Type': ['Growth curves', 'Fluorescence', 'Biofilm', 'Ads-Des (9mg)', 
                  'Ads-Des (5mg)', 'Experimental conditions', 'Efficiency comparison'],
    'Data_Points': [len(df_growth), len(df_fluorescence), len(df_biofilm), 
                    len(df_ads_9mg), len(df_ads_5mg), 1, len(df_verification)],
    'File_Saved': ['module_3.1_growth_curves.csv', 'module_3.1_fluorescence.csv',
                   'module_3.2_biofilm.csv', 'module_3.2_adsorption_9mg.csv',
                   'module_3.2_adsorption_5mg.csv', 'module_3.3_experimental_conditions.csv',
                   'module_3.4_verification.csv'],
}

df_summary = pd.DataFrame(data_summary)
print(df_summary.to_string(index=False))

# ============================================================================
# 数据可视化 - 快速预览
# ============================================================================
print("\n[生成数据预览图] 正在绘制...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 生长曲线
ax1 = fig.add_subplot(gs[0, :2])
colors = plt.cm.viridis(np.linspace(0, 1, 5))
for i, col in enumerate(df_growth.columns[1:]):
    label = col.replace('OD600_', '').replace('mg_L', ' mg/L')
    ax1.plot(df_growth['time_h'], df_growth[col], marker='o', 
             color=colors[i], linewidth=2, markersize=6, label=label)
ax1.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
ax1.set_ylabel('OD₆₀₀', fontsize=11, fontweight='bold')
ax1.set_title('3.1 Growth Curves under Different Cd²⁺ Concentrations', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 20)

# 2. 荧光强度
ax2 = fig.add_subplot(gs[0, 2])
colors_fl = ['#2E86AB', '#A23B72', '#F18F01']
for i, col in enumerate(df_fluorescence.columns[1:]):
    label = col.replace('FL_', '').replace('mM', ' mM')
    ax2.plot(df_fluorescence['time_h'], df_fluorescence[col], 
             marker='s', color=colors_fl[i], linewidth=2, markersize=6, label=label)
ax2.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Fluorescence Intensity (FU)', fontsize=11, fontweight='bold')
ax2.set_title('3.1 Fluorescence Response', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# 3. 生物膜形成
ax3 = fig.add_subplot(gs[1, 0])
ax3.errorbar(df_biofilm['Cd_concentration_mg_L'], df_biofilm['OD570_mean'], 
             yerr=df_biofilm['OD570_std'], marker='o', capsize=5, 
             color='#06A77D', linewidth=2, markersize=8, elinewidth=2)
ax3.set_xlabel('Cd²⁺ Concentration (mg/L)', fontsize=11, fontweight='bold')
ax3.set_ylabel('OD₅₇₀', fontsize=11, fontweight='bold')
ax3.set_title('3.2 Biofilm Formation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0.7, 0.9)

# 4. 吸附-解吸 (9mg/L)
ax4 = fig.add_subplot(gs[1, 1])
ads_steps = df_ads_9mg[df_ads_9mg['process_type'] == 'Adsorption']
des_steps = df_ads_9mg[df_ads_9mg['process_type'] == 'Desorption']
ax4.plot(ads_steps['cycle_number'], ads_steps['Cd_remaining_mg_L'], 
         marker='o', color='#D62828', linewidth=2, markersize=8, label='Adsorption')
ax4.plot(des_steps['cycle_number'], des_steps['Cd_remaining_mg_L'], 
         marker='s', color='#003049', linewidth=2, markersize=8, label='Desorption')
ax4.set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cd²⁺ Remaining (mg/L)', fontsize=11, fontweight='bold')
ax4.set_title('3.2 Ads-Des Cycles (9 mg/L)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xticks([1, 2, 3])

# 5. 吸附-解吸 (5mg/L)
ax5 = fig.add_subplot(gs[1, 2])
ads_steps_5 = df_ads_5mg[df_ads_5mg['process_type'] == 'Adsorption']
des_steps_5 = df_ads_5mg[df_ads_5mg['process_type'] == 'Desorption']
ax5.plot(ads_steps_5['cycle_number'], ads_steps_5['Cd_remaining_mg_L'], 
         marker='o', color='#D62828', linewidth=2, markersize=8, label='Adsorption')
ax5.plot(des_steps_5['cycle_number'], des_steps_5['Cd_remaining_mg_L'], 
         marker='s', color='#003049', linewidth=2, markersize=8, label='Desorption')
ax5.set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cd²⁺ Remaining (mg/L)', fontsize=11, fontweight='bold')
ax5.set_title('3.2 Ads-Des Cycles (5 mg/L)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xticks([1, 2, 3])

# 6. 吸附效率验证
ax6 = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(df_verification))
bars = ax6.bar(x_pos, df_verification['adsorption_efficiency_%'], 
               yerr=df_verification['std_%'], capsize=8, 
               color=['#EF476F', '#06A77D'], edgecolor='black', linewidth=1.5, width=0.6)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(df_verification['group'], fontsize=10, rotation=0)
ax6.set_ylabel('Adsorption Efficiency (%)', fontsize=11, fontweight='bold')
ax6.set_title('3.4 Adsorption Efficiency Verification', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 60)
ax6.grid(True, alpha=0.3, linestyle='--', axis='y')

# 在柱状图上添加数值标签
for i, (bar, val, std) in enumerate(zip(bars, df_verification['adsorption_efficiency_%'], 
                                         df_verification['std_%'])):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + std + 2,
             f'{val:.1f}%±{std:.1f}%', ha='center', va='bottom', 
             fontsize=10, fontweight='bold')

plt.suptitle('TasAnchor Project - Data Overview (功能测试模块)', 
             fontsize=14, fontweight='bold', y=0.995)

# 保存图片
output_path = 'figures/00_data_overview.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ 已保存数据预览图: {output_path}")
plt.close()

# ============================================================================
# 生成数据提取报告
# ============================================================================
report_path = 'results/data_extraction_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("TasAnchor 项目 - 数据提取报告\n")
    f.write("=" * 70 + "\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"项目目录: {os.getcwd()}\n\n")
    
    f.write("数据文件列表:\n")
    f.write("-" * 70 + "\n")
    for file in df_summary['File_Saved']:
        f.write(f"  ✓ data/raw/{file}\n")
    
    f.write("\n数据汇总:\n")
    f.write("-" * 70 + "\n")
    f.write(df_summary.to_string(index=False))
    
    f.write("\n\n重要提示:\n")
    f.write("-" * 70 + "\n")
    f.write("1. 当前数据为示例值，需要从 SCU-China Results 页面提取真实数据\n")
    f.write("2. 推荐使用 WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/)\n")
    f.write("3. 提取数据时注意单位换算和误差范围\n")
    f.write("4. 完成数据替换后，重新运行本脚本验证\n")
    
    f.write("\n下一步操作:\n")
    f.write("-" * 70 + "\n")
    f.write("1. 运行建模脚本: python code/02_module_3.1.py\n")
    f.write("2. 依次完成其他模块的建模分析\n")
    f.write("3. 生成最终报告和可视化结果\n")

print(f"✓ 已保存数据提取报告: {report_path}")

# ============================================================================
# 完成
# ============================================================================
print("\n" + "=" * 70)
print("数据提取完成！")
print("=" * 70)
print(f"\n生成的文件:")
print(f"  - 数据文件: {len(df_summary)} 个 CSV 文件 (保存在 data/raw/)")
print(f"  - 预览图表: figures/00_data_overview.png")
print(f"  - 提取报告: results/data_extraction_report.txt")

print(f"\n下一步操作:")
print(f"  1. 检查 figures/00_data_overview.png 确认数据正确")
print(f"  2. 如需替换真实数据，修改本脚本中的数据数组")
print(f"  3. 运行下一个脚本: python code/02_module_3.1.py")

print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)