"""Compact manuscript panels from the completed independent result table.

No experiment execution, scientific model import, or repeated membership audit.
"""
from pathlib import Path
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v4_residual_online/independent_analysis_01/residual_online_table.csv'
OUT = ROOT / 'paper/v4_figures/residual_results'
STYLES = {
    'ResidualCapacity': ('Capacity', '#0072B2', 'o'),
    'ResidualCheapSummary': ('CheapSummary', '#D55E00', 's'),
    'P1': ('P1', '#008A69', '^'),
    'AnytimeGreedyPortfolio': ('Greedy', '#6950A1', 'D'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5,
                         'axes.labelsize': 7.5, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'text.color': '#233342', 'axes.labelcolor': '#233342'})
    metrics = [
        ('online_quality', 'strict_return_relative_gain', '(a) Complete decision quality', 'Returned gain (%)'),
        ('online_extra', 'returned_extra_above_prefix_relative_gain', '(b) Recovery beyond the prefix', 'Additional returned gain (%)'),
        ('online_lateness', 'recorded_deadline_miss_rate', '(c) Deadline sensitivity', 'Late caller returns (%)'),
    ]
    for filename, field, title, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(2.22, 1.72))
        fig.subplots_adjust(left=.25, right=.97, bottom=.27, top=.83)
        for method, (label, color, marker) in STYLES.items():
            data = sorted([r for r in rows if r['method'] == method], key=lambda r: float(r['deadline_seconds']))
            if len(data) != 3:
                raise ValueError(f'Expected all three completed deadlines: {method}')
            ax.plot([float(r['deadline_seconds']) for r in data],
                    [100*float(r[field]) for r in data], color=color, marker=marker,
                    markersize=3.3, linewidth=1.35, label=label,
                    linestyle='--' if method == 'AnytimeGreedyPortfolio' else '-')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator([.139, .556, 2.221]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.minorticks_off()
        ax.set_xlabel('Total deadline (s)', labelpad=2)
        direction = r' $\downarrow$' if filename == 'online_lateness' else r' $\uparrow$'
        ax.set_ylabel(ylabel + direction, labelpad=2)
        ax.set_title(title, pad=6, loc='left', fontweight='bold')
        ax.grid(axis='y', color='#DDE3E8', linewidth=.55)
        ax.set_axisbelow(True)
        if filename == 'online_lateness':
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
        elif filename == 'online_extra':
            ax.set_ylim(-.15, 4.5)
            ax.set_yticks([0, 1, 2, 3, 4])
        else:
            ax.set_ylim(-5, 265)
            ax.set_yticks([0, 75, 150, 225])
        ax.axvline(.556, color='#667585', linewidth=.65, linestyle=':', zorder=0)
        for fmt in ['pdf', 'svg', 'png']:
            fig.savefig(OUT / f'{filename}.{fmt}', dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, .32))
    ax.axis('off')
    handles = [Line2D([0], [0], color=color, marker=marker, markersize=3.5,
                      linewidth=1.4, label=label,
                      linestyle='--' if method == 'AnytimeGreedyPortfolio' else '-')
               for method, (label, color, marker) in STYLES.items()]
    ax.legend(handles=handles, loc='center', ncol=4, frameon=False,
              handlelength=1.8, columnspacing=2.5)
    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'online_legend.{fmt}', dpi=220)
    plt.close(fig)
    (OUT / 'online_panels_provenance.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'completed_units': 1512, 'aggregation': 'fit seeds within graph, equal menu/resource domain mass',
        'new_experiments': 0, 'panels': [r[0] for r in metrics],
    }, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
