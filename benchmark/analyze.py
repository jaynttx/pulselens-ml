"""Evaluate exported paired benchmark observations; Python standard library only."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

def summarize(errors):
    n = len(errors)
    return {'n': n, 'mae_bpm': sum(map(abs, errors)) / n,
            'rmse_bpm': math.sqrt(sum(e * e for e in errors) / n),
            'bias_bpm': sum(errors) / n}

def analyze(path):
    groups = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        expected = {'condition', 'method', 'app_bpm', 'reference_bpm'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError('Expected condition,method,app_bpm,reference_bpm columns')
        for number, row in enumerate(reader, 2):
            if not row['app_bpm'] and not row['reference_bpm']:
                continue
            try:
                app, reference = float(row['app_bpm']), float(row['reference_bpm'])
                if not all(math.isfinite(v) and 30 <= v <= 240 for v in (app, reference)):
                    raise ValueError('BPM must be finite and between 30 and 240')
                if row['method'] not in {'pos', 'deep'}:
                    raise ValueError('method must be pos or deep')
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid observation at line {number}: {error}') from error
            error = app - reference
            groups[(row['method'], row['condition'])].append(error)
            groups[(row['method'], 'ALL')].append(error)
    return [{'method': method, 'condition': condition, **summarize(errors)}
            for (method, condition), errors in sorted(groups.items())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='CSV exported from Benchmark lab')
    parser.add_argument('--output', default='benchmark/output/summary.csv')
    args = parser.parse_args()
    result = analyze(args.csv)
    if not result:
        parser.error('No complete observations found')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    print(f'Wrote {len(result)} summaries to {output}')
    for row in result:
        if row['condition'] == 'ALL':
            print(f"{row['method']}: n={row['n']}, MAE={row['mae_bpm']:.2f}, RMSE={row['rmse_bpm']:.2f}, bias={row['bias_bpm']:.2f} BPM")

if __name__ == '__main__':
    main()
