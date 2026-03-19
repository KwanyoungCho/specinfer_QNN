#!/usr/bin/env python3
"""
SnapKV Overhead Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np

# Data from analysis
standard_total = 984.389
snapkv_total = 21.405

# SnapKV breakdown
importance_total = 12.554
k_pack_total = 0.718
v_pack_total = 8.133

# Detailed SnapKV operations
snapkv_detailed = {
    'Full KQV': 10.397,
    'KQ Copy': 1.033,
    'Importance Sum': 0.293,
    'Importance Reshape': 0.287,
    'GQA Reduce': 0.064,
    'Argsort': 0.480,
    'K Get Rows': 0.192,
    'K Win Copy': 0.098,
    'K Concat': 0.159,
    'K Pack Write': 0.268,
    'V Get Rows': 0.257,
    'V Concat': 3.747,
    'V Permute': 0.227,
    'V Pack Write': 3.902,
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Color schemes
colors_main = ['#2E86AB', '#A23B72']
colors_snapkv = ['#E63946', '#457B9D', '#1D3557']
colors_detailed = plt.cm.Set3(np.linspace(0, 1, len(snapkv_detailed)))

# ===== Plot 1: Overall breakdown (pie chart) =====
ax1 = fig.add_subplot(gs[0, 0])
total_time = standard_total + snapkv_total
sizes = [standard_total, snapkv_total]
labels = ['Standard\nOperations', 'SnapKV\nOverhead']
explode = (0, 0.1)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, 
                                     autopct=lambda pct: f'{pct:.1f}%\n({pct*total_time/100:.1f}ms)',
                                     colors=colors_main, startangle=90,
                                     textprops={'fontsize': 11, 'weight': 'bold'})

ax1.set_title('Total Prefill Time Breakdown\n(125 tokens, 32 layers)', 
              fontsize=13, weight='bold', pad=20)

# Add total time annotation
ax1.text(0, -1.4, f'Total: {total_time:.1f} ms', 
         ha='center', fontsize=12, weight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

# ===== Plot 2: SnapKV overhead breakdown (pie chart) =====
ax2 = fig.add_subplot(gs[0, 1])
sizes_snapkv = [importance_total, k_pack_total, v_pack_total]
labels_snapkv = ['Importance\nScoring', 'K Packing', 'V Packing']
explode_snapkv = (0.05, 0, 0.05)

wedges2, texts2, autotexts2 = ax2.pie(sizes_snapkv, explode=explode_snapkv, 
                                       labels=labels_snapkv,
                                       autopct=lambda pct: f'{pct:.1f}%\n({pct*snapkv_total/100:.2f}ms)',
                                       colors=colors_snapkv, startangle=45,
                                       textprops={'fontsize': 11, 'weight': 'bold'})

ax2.set_title('SnapKV Overhead Breakdown\n(21.4 ms total)', 
              fontsize=13, weight='bold', pad=20)

# ===== Plot 3: SnapKV stages comparison (bar chart) =====
ax3 = fig.add_subplot(gs[1, 0])
stages = ['Importance\nScoring', 'K\nPacking', 'V\nPacking']
times = [importance_total, k_pack_total, v_pack_total]
bars = ax3.bar(stages, times, color=colors_snapkv, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, time in zip(bars, times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.2f}ms\n({time/snapkv_total*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, weight='bold')

ax3.set_ylabel('Time (ms)', fontsize=12, weight='bold')
ax3.set_title('SnapKV Overhead by Stage', fontsize=13, weight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(times) * 1.2)

# ===== Plot 4: Detailed operations (horizontal bar chart) =====
ax4 = fig.add_subplot(gs[1, 1])

# Group operations by stage
importance_ops = ['Full KQV', 'KQ Copy', 'Importance Sum', 'Importance Reshape', 'GQA Reduce', 'Argsort']
k_ops = ['K Get Rows', 'K Win Copy', 'K Concat', 'K Pack Write']
v_ops = ['V Get Rows', 'V Concat', 'V Permute', 'V Pack Write']

all_ops = importance_ops + k_ops + v_ops
all_times = [snapkv_detailed[op] for op in all_ops]

# Color by stage
colors_bars = (['#E63946'] * len(importance_ops) + 
               ['#457B9D'] * len(k_ops) + 
               ['#1D3557'] * len(v_ops))

y_pos = np.arange(len(all_ops))
bars = ax4.barh(y_pos, all_times, color=colors_bars, edgecolor='black', linewidth=0.8)

# Add value labels
for i, (bar, time) in enumerate(zip(bars, all_times)):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f' {time:.2f}ms',
            ha='left', va='center', fontsize=9, weight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(all_ops, fontsize=9)
ax4.set_xlabel('Time (ms)', fontsize=11, weight='bold')
ax4.set_title('Detailed SnapKV Operations', fontsize=13, weight='bold', pad=15)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.invert_yaxis()

# Add legend for stages
legend_patches = [
    mpatches.Patch(color='#E63946', label='Importance Scoring'),
    mpatches.Patch(color='#457B9D', label='K Packing'),
    mpatches.Patch(color='#1D3557', label='V Packing')
]
ax4.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)

# Overall title
fig.suptitle('SnapKV Prefill Overhead Analysis', 
             fontsize=16, weight='bold', y=0.98)

# Save figure
plt.savefig('/home/chokwans99/dev/llm/specinfer.cpp/tmp_result/snapkv_overhead_plots.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: snapkv_overhead_plots.png")

# Also create a focused view on just SnapKV breakdown
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart
wedges, texts, autotexts = ax1.pie(sizes_snapkv, explode=explode_snapkv, 
                                    labels=labels_snapkv,
                                    autopct=lambda pct: f'{pct:.1f}%\n({pct*snapkv_total/100:.2f}ms)',
                                    colors=colors_snapkv, startangle=45,
                                    textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('SnapKV Overhead Distribution', fontsize=14, weight='bold', pad=20)

# Right: Bar chart with detailed breakdown
y_pos = np.arange(len(all_ops))
bars = ax2.barh(y_pos, all_times, color=colors_bars, edgecolor='black', linewidth=1)

for i, (bar, time) in enumerate(zip(bars, all_times)):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f' {time:.2f}ms ({time/snapkv_total*100:.1f}%)',
            ha='left', va='center', fontsize=9, weight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(all_ops, fontsize=10)
ax2.set_xlabel('Time (ms)', fontsize=12, weight='bold')
ax2.set_title('Per-Operation Breakdown', fontsize=14, weight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.invert_yaxis()

legend_patches = [
    mpatches.Patch(color='#E63946', label=f'Importance ({importance_total:.2f}ms)'),
    mpatches.Patch(color='#457B9D', label=f'K Packing ({k_pack_total:.2f}ms)'),
    mpatches.Patch(color='#1D3557', label=f'V Packing ({v_pack_total:.2f}ms)')
]
ax2.legend(handles=legend_patches, loc='lower right', fontsize=10, framealpha=0.9)

fig2.suptitle(f'SnapKV Overhead Details (Total: {snapkv_total:.2f}ms)', 
              fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig('/home/chokwans99/dev/llm/specinfer.cpp/tmp_result/snapkv_overhead_detailed.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: snapkv_overhead_detailed.png")

plt.show()
print("\nGraphs generated successfully!")
print(f"Total prefill time: {total_time:.1f}ms")
print(f"SnapKV overhead: {snapkv_total:.2f}ms ({snapkv_total/total_time*100:.1f}%)")
