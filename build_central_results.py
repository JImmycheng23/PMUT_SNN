"""Build central results master CSV/JSON from all experiment outputs."""
import csv, json, glob, os
from pathlib import Path

rows = []

def add_row(**kw):
    rows.append(kw)

# ── Neural models (train.py) — Point 5/6 ──
for f in sorted(glob.glob('outputs/runs/high-glucose_grouped_*_w128_seed7/metrics.json')):
    d = json.load(open(f))
    model = d.get('model', '?')
    enc = d.get('dataset', {}).get('encode_channels', '')
    label = f"{model} ({enc})" if enc and 'encoded' in model else model
    exp = 'Point 5 Encoder→SNN' if 'encoded' in model else 'Point 6 Neural models'
    add_row(
        experiment=exp, task='high-glucose', split='grouped', window=128,
        model=label, seed=str(d['seed']),
        final_test_accuracy=f"{d['final_test_accuracy']:.4f}",
        balanced_accuracy=f"{d.get('balanced_accuracy', 'N/A')}",
        macro_f1=f"{d.get('macro_f1', 'N/A')}",
        best_val_accuracy=f"{d['best'].get('val_accuracy', 'N/A')}",
        best_epoch=str(d['best']['epoch']),
        output_file=str(Path(f).relative_to('outputs')),
    )

# ── Reservoir ESN + LSM — Points 10/14 ──
for model_type, subdir in [('ESN (tanh)', ''), ('LSM (membrane)', 'lsm')]:
    for f in sorted(glob.glob(f'outputs/reservoir/*/grouped_w128/{subdir}metrics_summary.json'.replace('//','/'))):
        task = Path(f).parts[2]  # outputs / reservoir / {task} / ...
        d = json.load(open(f))
        acc_key = 'test_accuracy_mean' if 'test_accuracy_mean' in d else 'test_accuracy_mean'
        add_row(
            experiment='Point 10 Reservoir', task=task, split='grouped', window=128,
            model=model_type, seed=f"{d.get('num_seeds','?')} seeds",
            final_test_accuracy=f"{d.get('test_accuracy_mean', d.get('test_accuracy','?')):.4f} ± {d.get('test_accuracy_std', 0):.4f}",
            balanced_accuracy='—', macro_f1='—',
            best_val_accuracy='—', best_epoch='—',
            output_file=str(Path(f).relative_to('outputs')),
        )
    for f in sorted(glob.glob(f'outputs/reservoir/*/window_w128/{subdir}metrics_summary.json'.replace('//','/'))):
        task = Path(f).parts[2]  # outputs / reservoir / {task} / ...
        d = json.load(open(f))
        add_row(
            experiment='Point 10 Reservoir', task=task, split='window', window=128,
            model=model_type, seed=f"{d.get('num_seeds','?')} seeds",
            final_test_accuracy=f"{d.get('test_accuracy_mean', d.get('test_accuracy','?')):.4f} ± {d.get('test_accuracy_std', 0):.4f}",
            balanced_accuracy='—', macro_f1='—',
            best_val_accuracy='—', best_epoch='—',
            output_file=str(Path(f).relative_to('outputs')),
        )

# ── Modality ablation — Point 11 ──
for model_type, subdir in [('ESN', ''), ('LSM', 'lsm')]:
    pattern = f'outputs/reservoir/*/grouped_w128/{subdir}modality_comparison.json'.replace('//','/')
    for f in sorted(glob.glob(pattern)):
        task = Path(f).parts[2]  # outputs / reservoir / {task} / ...
        data = json.load(open(f))
        for mr in data:
            if 'error' in mr: continue
            add_row(
                experiment='Point 11 Modality', task=task, split='grouped', window=128,
                model=f"{model_type} {mr['modality']}", seed='1',
                final_test_accuracy=f"{mr['test_accuracy']:.4f}",
                balanced_accuracy='—', macro_f1='—',
                best_val_accuracy='—', best_epoch='—',
                output_file=str(Path(f).relative_to('outputs')),
            )

# ── NB5 Feature ML — Point 3/12 ──
for task_name in ['high-glucose']:  # glucose-ladder removed from primary study (Point 1)
    csv_path = f'outputs/feature_ml/{task_name}_baseline_summary.csv'
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        for _, row_df in df.iterrows():
            add_row(
                experiment='Point 3/12 Feature ML', task=task_name, split='CV', window='—',
                model=row_df['model'], seed='CV',
                final_test_accuracy=f"{row_df['accuracy']:.4f}",
                balanced_accuracy='—',
                macro_f1=f"{row_df['f1_macro']:.4f}" if 'f1_macro' in row_df else '—',
                best_val_accuracy='—', best_epoch='—',
                output_file=csv_path,
            )

# ── Write ──
os.makedirs('outputs/central_results', exist_ok=True)
fieldnames = [
    'experiment', 'task', 'split', 'window', 'model', 'seed',
    'final_test_accuracy', 'balanced_accuracy', 'macro_f1',
    'best_val_accuracy', 'best_epoch', 'output_file',
]

out_csv = 'outputs/central_results/master_results.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open('outputs/central_results/master_results.json', 'w') as f:
    json.dump(rows, f, indent=2)

print(f'Wrote {len(rows)} rows to {out_csv}')
