"""Summarize native CLI baselines; estimates and board samples stay separately named."""
import csv
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for case in sorted(root.iterdir()):
    if not (case / 'exit.txt').exists():
        continue
    row = {'case': case.name, 'exit': int((case / 'exit.txt').read_text())}
    stderr = (case / 'stderr.log').read_text(errors='replace')
    row['nonfinite_skips'] = stderr.count('non-finite gradient norm')
    match = re.search(r'memory: (.*?), ([\d.]+) of ([\d.]+) GB free per GPU', stderr)
    if match:
        row.update(mode=match[1], planned_GB=float(match[2]), free_GB=float(match[3]))
    no_fit = re.search(r'memory: no mode fits ([\d.]+) GB free \(smallest needs ([\d.]+) GB\); trying (.*)', stderr)
    if no_fit:
        row.update(mode=no_fit[3], planned_GB=float(no_fit[2]), free_GB=float(no_fit[1]))
    row['fits'] = 'no mode fits' not in stderr if 'planned_GB' in row else None
    timing = (case / 'time.txt').read_text(errors='replace')
    match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', timing)
    row['host_VmHWM_KiB'] = int(match[1]) if match else None
    sampled = case / 'gpu-sampled.csv'
    values = []
    if sampled.exists():
        for line in sampled.read_text().splitlines():
            try:
                values.append(float(line.rsplit(',', 1)[1]))
            except (ValueError, IndexError):
                pass
    row['board_sampled_peak_MiB'] = max(values, default=None)
    log = case / 'run/log.csv'
    if log.exists():
        data = list(csv.DictReader(log.open()))
        if not data:
            row['error_tail'] = 'no completed training steps: ' + '\n'.join(stderr.splitlines()[-3:])
        # Exclude initial warm-up / JIT interval; each remaining row covers five steps.
        rates = [float(x['samples_per_s']) for x in data[1:] if float(x['samples_per_s']) > 0]
        if rates and not row['nonfinite_skips']:
            z, xy = map(int, case.name.split('-')[-2].split('x'))
            rate = statistics.median(rates)
            row['end_to_end_ms_step'] = 1000 / rate
            row['end_to_end_Mvox_s'] = rate * z * xy * xy / 1e6
            row['median_wait_s_per_interval'] = statistics.median(float(x['wait_s']) for x in data[1:])
    if case.name.startswith('infer-'):
        match = re.search(r'gpu\(fwd\+d2h\) ([\d.]+).*?wall ([\d.]+), (\d+) tiles', stderr)
        if match:
            z, xy = map(int, case.name.split('-')[-1].split('x'))
            gpu_s, wall_s, tiles = float(match[1]), float(match[2]), int(match[3])
            row.update(tiles=tiles, inference_window_Mvox_s=tiles * z * xy * xy / 1e6 / wall_s,
                       inference_gpu_window_Mvox_s=tiles * z * xy * xy / 1e6 / gpu_s if gpu_s else None)
    if row['exit']:
        row['error_tail'] = '\n'.join(stderr.splitlines()[-5:])
    rows.append(row)
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
keys = ['case', 'exit', 'fits', 'mode', 'planned_GB', 'free_GB', 'host_VmHWM_KiB',
        'board_sampled_peak_MiB', 'end_to_end_ms_step', 'end_to_end_Mvox_s',
        'median_wait_s_per_interval', 'tiles', 'inference_window_Mvox_s',
        'inference_gpu_window_Mvox_s', 'nonfinite_skips', 'error_tail']
with (root / 'summary.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
