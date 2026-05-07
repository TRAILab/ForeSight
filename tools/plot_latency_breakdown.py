"""Paper figure: stacked horizontal bar of inference-latency breakdown.

Run:
    python tools/plot_latency_breakdown.py
Outputs reports/latency_breakdown.{pdf,png}.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- data (ms) ---
components = ['Encoder', 'Det', 'Map', 'Pred', 'Plan', 'Collision Rescore']
data = {
    'Baseline': [12.22, 26.67, 20.00, 21.11,  4.44, 26.67],
    'Ours':     [12.56,  0.00,  0.00,  0.00, 35.75,  0.14],
}
# Optional totals override (set to None to use sum(data[v])).
totals_override = {}
# Visually distinct, colorblind-safe palette (Okabe-Ito-ish).
colors = {
    'Encoder': '#4A4A4A',  # dark grey — shared compute
    'Det':     '#D55E00',  # orange-red — perception
    'Map':     '#E69F00',  # gold — perception
    'Pred':    '#CC79A7',  # pink — perception
    'Plan':        '#0072B2',  # blue — planner (the part we keep)
    'Collision Rescore': '#009E73',  # green — collision rescorer
}

# --- figure ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,    # editable text in PDF
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(6.0, 2.0))

variants = list(data.keys())
y_positions = np.arange(len(variants))
bar_height = 0.55

totals = {v: totals_override.get(v, sum(data[v])) for v in variants}
baseline_total = totals['Baseline']

for i, variant in enumerate(variants):
    left = 0.0
    for comp in components:
        width = data[variant][components.index(comp)]
        if width <= 0:
            continue
        ax.barh(
            y_positions[i], width, left=left,
            height=bar_height, color=colors[comp],
            edgecolor='white', linewidth=0.6,
            label=comp if i == 0 else None,
        )
        # Inline label only if segment is wide enough.
        if width / baseline_total > 0.05:
            ax.text(
                left + width / 2, y_positions[i],
                f'{width:.1f}', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
            )
        left += width

    # Total + speedup at bar end.
    total = totals[variant]
    speedup = baseline_total / total
    label = f'  {total:.1f} ms'
    if variant != 'Baseline':
        label += f'  ({speedup:.1f}×)'
    ax.text(total, y_positions[i], label,
            ha='left', va='center', fontsize=9)

ax.set_yticks(y_positions)
ax.set_yticklabels(variants)
ax.invert_yaxis()  # Baseline on top
ax.set_xlabel('Latency (ms)')
ax.set_xlim(0, baseline_total * 1.18)  # leave room for labels
ax.tick_params(axis='y', length=0)

# Legend ordered by component list.
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(c) for c in components if c in labels]
ax.legend(
    [handles[k] for k in order], [labels[k] for k in order],
    loc='upper center', bbox_to_anchor=(0.5, -0.45),
    ncol=len(components), frameon=False, fontsize=9, handlelength=1.0,
    columnspacing=1.2, handletextpad=0.4,
)

fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.30)

out_dir = Path(__file__).resolve().parent.parent / 'reports'
out_dir.mkdir(exist_ok=True)
for ext in ('pdf', 'png'):
    out = out_dir / f'latency_breakdown.{ext}'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'wrote {out}')
