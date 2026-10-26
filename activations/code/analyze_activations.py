"""Step 3: compare the six activations side by side. Reads results/runs/.

Writes
  results/activation_summary.csv   one row per activation, mean (and std) over seeds
  results/activation_table.md      the same, rounded, ready to paste onto a slide
  results/plots/learning_curves.png     train and test loss by epoch
  results/plots/final_error.png         final train vs test error
  results/plots/gradient_flow.png       vanishing-gradient check
  results/plots/activation_health.png   average slope f'(z) and stuck units

    python analyze_activations.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from diagnostics import ACT_LAYERS, DENSE_LAYERS
from train import run_dir

# One fixed colour per activation, in this order everywhere (colour follows
# the activation, never its rank).
COLORS = dict(zip(ACTIVATIONS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
TRAIN_COLOR, TEST_COLOR = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
    'legend.frameon': False, 'font.size': 10, 'axes.titlesize': 11,
})


def _number_if_possible(text):
    try:
        return float(text)
    except ValueError:
        return text      # 'relu', 'True', ...


def load_runs():
    """Every finished run -> (metrics, history, diagnostics) DataFrames."""
    metrics, history, diags = [], [], []
    for activation in ACTIVATIONS:
        for seed in RUN_SEEDS:
            d = run_dir(activation, seed)
            if not os.path.exists(os.path.join(d, 'metrics.csv')):
                continue
            m = pd.read_csv(os.path.join(d, 'metrics.csv'), index_col=0, dtype=str)['value']
            metrics.append(m.map(_number_if_possible))
            for name, store in (('history', history), ('diagnostics', diags)):
                df = pd.read_csv(os.path.join(d, f'{name}.csv'))
                df['activation'], df['seed'] = activation, seed
                store.append(df)
    if not metrics:
        raise SystemExit('No finished runs yet - run run_activations.py first.')
    metrics = pd.DataFrame(metrics)
    metrics['seed'] = metrics['seed'].astype(int)
    return metrics, pd.concat(history), pd.concat(diags)


def convergence_epochs(history):
    """First epoch where test loss <= CONVERGENCE_TOLERANCE x the best test loss
    ANY run reached. One finish line for everybody, so the epochs compare."""
    target = CONVERGENCE_TOLERANCE * history['test_loss'].min()
    out = {}
    for (act, seed), h in history.groupby(['activation', 'seed']):
        hit = h.loc[h['test_loss'] <= target, 'epoch']
        out[(act, seed)] = hit.iloc[0] if len(hit) else np.nan   # nan = never got there
    return pd.Series(out, name='epochs_to_converge'), target


def summarise(metrics, history, diags):
    conv, target = convergence_epochs(history)
    best = history.sort_values('test_loss').groupby(['activation', 'seed']).head(1)
    per_run = metrics.set_index(['activation', 'seed'])
    per_run['epochs_to_converge'] = conv
    per_run['best_test_epoch'] = best.set_index(['activation', 'seed'])['epoch']
    per_run['train_test_gap'] = per_run['test_mse'] - per_run['train_mse']

    # Vanishing-gradient signal: first hidden layer's gradient relative to the
    # output layer's. Near 1 = healthy flow; far below 1 = signal fades.
    d = diags.set_index(['activation', 'seed'])
    ratio = d['grad_norm_hidden1'] / d['grad_norm_output']
    start, end = d['epoch'] == 0, d['epoch'] == d.groupby(level=[0, 1])['epoch'].transform('max')
    per_run['grad_ratio_start'] = ratio[start]
    per_run['grad_ratio_end'] = ratio[end]
    per_run['max_grad_norm'] = d[[f'grad_norm_{n}' for n in DENSE_LAYERS]].max(axis=1) \
        .groupby(level=[0, 1]).max()
    per_run['stuck_frac_end'] = d.loc[end, [f'stuck_frac_{a}' for a in ACT_LAYERS]].mean(axis=1)
    per_run['mean_slope_end'] = d.loc[end, [f'mean_slope_{a}' for a in ACT_LAYERS]].mean(axis=1)

    cols = ['train_mse', 'test_mse', 'train_test_gap', 'test_r2', 'test_mae_units',
            'test_rmse_units', 'epochs_to_converge', 'best_test_epoch', 'sec_per_epoch',
            'grad_ratio_start', 'grad_ratio_end', 'max_grad_norm', 'mean_slope_end',
            'stuck_frac_end']
    cols += [c for c in per_run.columns if c.startswith('prelu_alpha')]
    grouped = per_run[cols].astype(float).groupby(level='activation')
    summary = grouped.mean()
    summary = summary.join(grouped.std()[['train_mse', 'test_mse']].add_suffix('_sd'))
    summary['n_seeds'] = grouped.size()
    summary['n_diverged'] = per_run['diverged'].astype(str).eq('True').groupby(level='activation').sum()
    summary = summary.reindex([a for a in ACTIVATIONS if a in summary.index])
    summary.to_csv(SUMMARY_PATH)
    return summary, target


def write_table(summary, target):
    t = pd.DataFrame({
        'Train MSE': summary['train_mse'].map('{:.4f}'.format),
        'Test MSE (± sd)': [f'{m:.4f} ± {s:.4f}' for m, s in
                            zip(summary['test_mse'], summary['test_mse_sd'].fillna(0))],
        'Gap': summary['train_test_gap'].map('{:.4f}'.format),
        'Test R²': summary['test_r2'].map('{:.3f}'.format),
        'Test MAE (units)': summary['test_mae_units'].map('{:.2f}'.format),
        'Epochs to converge': summary['epochs_to_converge'].map(
            lambda e: f'>{EPOCHS}' if np.isnan(e) else f'{e:.1f}'),
        'Best epoch': summary['best_test_epoch'].map('{:.1f}'.format),
        'Sec / epoch': summary['sec_per_epoch'].map('{:.1f}'.format),
        'Grad ratio h1/out (start → end)': [f'{a:.3f} → {b:.3f}' for a, b in
                                            zip(summary['grad_ratio_start'], summary['grad_ratio_end'])],
        'Stuck units': summary['stuck_frac_end'].map('{:.1%}'.format),
    }, index=summary.index)
    note = (f'Mean over seeds {RUN_SEEDS}. MSE is on standardised log(1+quantity), the '
            f'training loss. "Epochs to converge" = first epoch with test MSE <= '
            f'{target:.4f} ({CONVERGENCE_TOLERANCE:.0%} of the best test MSE of any run).')
    lines = ['| Activation | ' + ' | '.join(t.columns) + ' |',
             '|' + '---|' * (len(t.columns) + 1)]
    lines += [f'| {act} | ' + ' | '.join(row) + ' |' for act, row in t.iterrows()]
    with open(SUMMARY_TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n\n' + note + '\n')
    return t


def plot_learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in zip(axes, ['train_loss', 'test_loss'], ['Train loss', 'Test loss']):
        for act in ACTIVATIONS:
            h = history[history['activation'] == act]
            if h.empty:
                continue
            g = h.groupby('epoch')[col]
            ax.fill_between(g.mean().index, g.min(), g.max(), color=COLORS[act], alpha=0.15, lw=0)
            ax.plot(g.mean().index, g.mean(), color=COLORS[act], label=act)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xlabel('epoch')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Zoom in: sigmoid's first epoch is far above everything else and would
    # squash the differences we care about into one line.
    top = history.groupby(['activation', 'epoch'])['test_loss'].mean().quantile(0.97)
    axes[0].set_ylim(history['test_loss'].min() * 0.985, top)
    axes[0].set_ylabel('MSE (standardised log quantity)')
    axes[0].annotate('sigmoid starts off the chart\n(epoch 1 train loss ≈ 0.44)',
                     (1, top), xytext=(3, top * 0.995), fontsize=8, color=MUTED, va='top')
    axes[1].legend(title='activation', loc='upper right')
    fig.suptitle('Learning curves by hidden activation (line = mean over seeds, band = min–max)',
                 x=0.01, ha='left', color=INK)
    _save(fig, 'learning_curves.png')


def plot_final_error(summary):
    """Dot plot, not bars: the differences are in the third decimal, and a bar
    has to start at zero, which would hide them."""
    y = np.arange(len(summary))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.hlines(y, summary['train_mse'], summary['test_mse'], color=GRID, lw=3)
    for col, label, color in (('train_mse', 'train', TRAIN_COLOR), ('test_mse', 'test', TEST_COLOR)):
        ax.errorbar(summary[col], y, xerr=summary[f'{col}_sd'].fillna(0), fmt='o', ms=8,
                    color=color, ecolor=MUTED, elinewidth=1, capsize=3, label=label)
    for yi, v in zip(y, summary['test_mse']):
        ax.annotate(f'{v:.4f}', (v, yi), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=8, color=MUTED)
    ax.set_yticks(y, summary.index)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('MSE after training (standardised log quantity)  ← better')
    ax.set_title('Final train vs test error (whiskers = sd over seeds)', loc='left', color=INK)
    ax.legend(loc='lower right')
    _save(fig, 'final_error.png')


def plot_gradient_flow(diags):
    """Left: gradient size at each layer before training (log scale).
    Right: hidden1 / output gradient ratio over training."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    start = diags[diags['epoch'] == 0]
    for act in ACTIVATIONS:
        d = diags[diags['activation'] == act]
        if d.empty:
            continue
        s = start[start['activation'] == act][[f'grad_norm_{n}' for n in DENSE_LAYERS]].mean()
        axes[0].plot(DENSE_LAYERS, s.values, marker='o', ms=6, color=COLORS[act], label=act)
        r = (d['grad_norm_hidden1'] / d['grad_norm_output']).groupby(d['epoch']).mean()
        axes[1].plot(r.index, r.values, color=COLORS[act], label=act)
    # At the start every z is near 0, where tanh and elu both have slope 1 and
    # the three relus are identical, so those lines sit exactly on top of each other.
    hidden1 = start.groupby('activation')['grad_norm_hidden1'].mean()
    for acts, text in ((['tanh', 'elu'], 'tanh = elu'),
                       (['relu', 'leaky_relu', 'prelu'], 'relu = leaky = prelu'),
                       (['sigmoid'], 'sigmoid')):
        if acts[0] in hidden1:
            axes[0].annotate(text, (0, hidden1[acts[0]]), xytext=(8, -12),
                             textcoords='offset points', fontsize=8, color=MUTED)
    axes[0].set_yscale('log')
    axes[0].set_title('Gradient size by layer, before training', loc='left', color=INK)
    axes[0].set_ylabel('‖dLoss/dW‖ (log scale)')
    axes[0].set_xlabel('layer (input side → output side)')
    axes[1].set_yscale('log')
    axes[1].set_title('Gradient reaching hidden1, relative to output layer', loc='left', color=INK)
    axes[1].set_ylabel('‖grad hidden1‖ / ‖grad output‖ (log)')
    axes[1].set_xlabel('epoch')
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend(title='activation')
    _save(fig, 'gradient_flow.png')


def plot_activation_health(diags):
    """Average slope f'(z) and share of stuck units, per hidden layer, at the end."""
    end = diags[diags['epoch'] == diags.groupby(['activation', 'seed'])['epoch'].transform('max')]
    end = end.groupby('activation').mean(numeric_only=True)
    end = end.reindex([a for a in ACTIVATIONS if a in end.index])
    x = np.arange(len(ACT_LAYERS))
    w = 0.8 / len(end)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, act in enumerate(end.index):
        off = (i - (len(end) - 1) / 2) * w
        for ax, stat in zip(axes, ['mean_slope', 'stuck_frac']):
            ax.bar(x + off, end.loc[act, [f'{stat}_{a}' for a in ACT_LAYERS]].values, w - 0.02,
                   color=COLORS[act], label=act)
    for ax, title, ylabel in zip(
            axes, ["Average slope f'(z) after training", 'Stuck units after training'],
            ["mean |f'(z)|  (signal let through)", "share of units with f'(z) ≈ 0 on every row"]):
        ax.set_xticks(x, ACT_LAYERS)
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylabel(ylabel)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    axes[0].legend(title='activation', ncol=2)
    _save(fig, 'activation_health.png')


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=PLOT_DPI)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')   # Windows console chokes on ± and →
    metrics, history, diags = load_runs()
    summary, target = summarise(metrics, history, diags)
    print(write_table(summary, target).to_string())
    plot_learning_curves(history)
    plot_final_error(summary)
    plot_gradient_flow(diags)
    plot_activation_health(diags)
    print(f'\nSaved {SUMMARY_PATH}, {SUMMARY_TABLE_PATH} and plots in {PLOT_DIR}')
