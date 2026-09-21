import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path(sys.argv[1])
data = json.loads((root / 'data_audit_val.json').read_text())
example = max(data['examples'], key=lambda x: x['bridge_length_m'])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 7.0), constrained_layout=True)
for ax in axes:
    source = np.asarray(example['source_xy'])
    ax.plot(source[:, 0], source[:, 1], color='#a7b0b9', lw=3, label='Original road geometry', zorder=1)
    ax.add_patch(Rectangle((-15, -30), 30, 60, fill=False, edgecolor='#52606d', ls='--', lw=1.2))
    ax.set(xlim=(-20, 20), ylim=(-34, 34), xlabel='Lateral x (m)', ylabel='Forward y (m)', aspect='equal')
    ax.grid(alpha=.15)
old = np.asarray(example['old_xy'])
axes[0].plot(old[:, 0], old[:, 1], color='#ca3e47', lw=1.7, label='Training target before fix')
for i, part in enumerate(example['clipped_parts']):
    pts = np.asarray(part)
    axes[1].plot(pts[:, 0], pts[:, 1], color='#008577', lw=2,
                 label='Separate clipped targets' if i == 0 else None)
axes[0].set_title('Before: outside vertices deleted')
axes[1].set_title('After: full segments clipped to ROI')
for ax in axes:
    ax.legend(loc='upper left', fontsize=8)
fig.suptitle('A real map-label error: the model was supervised with a false link\n'
             + example['town'] + ' | ' + example['kind'] + ' | frame ' + str(example['frame']), fontsize=12)
fig.savefig(root / 'real_label_clipping.png', dpi=170)
plt.close(fig)

res = json.loads((root / 'decode_audit_v2/result.json').read_text())
thresholds = [0., .1, .3, .5]
fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
for metric, label, color in [('macro_iou', 'All-class mean IoU (includes absent classes)', '#566c88'),
                              ('gt_present_iou', 'IoU on classes actually present', '#ca3e47')]:
    vals = [100*res['summary']['flat50@%.1f' % t][metric] for t in thresholds]
    ax.plot(thresholds, vals, 'o-', lw=2, label=label, color=color)
    for t, v in zip(thresholds, vals):
        ax.annotate('%.1f' % v, (t, v), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9)
ax.set(xlabel='Score threshold', ylabel='IoU (%)', xlim=(-.02, .52), ylim=(0, 55),
       title='A higher score cutoff can hide worse map coverage')
ax.grid(alpha=.15)
ax.legend(loc='center right', fontsize=8)
fig.text(.02, -.045, 'Same frozen res480 model; 50 clip midpoints; cold-start diagnostic, not the full benchmark.', fontsize=8)
fig.savefig(root / 'threshold_tradeoff.png', dpi=170, bbox_inches='tight')
plt.close(fig)
