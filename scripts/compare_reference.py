"""Compare controlled inputs with the pinned reference algorithms.

Uses an external checkout read-only; does not copy its source into this project.
The classical check executes the reference TypeScript via Node's type stripper.
The neural check independently binds recurrent states by their tensor names.
This verifies implementation agreement, NOT accuracy on a person's pulse.
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulselens.assets import REVISION, ensure
from pulselens.classical import estimate_rgb
from pulselens.facephys import FacePhys, interpreter, PROJECTION_KEYS


def classical_comparison(checkout):
    files = {
        'signal': 'apps/pulselens-classical/src/lib/detection/signal.ts',
        'spatial-beamformer': 'apps/pulselens-classical/src/lib/detection/spatial-beamformer.ts',
        'rppg': 'apps/pulselens-classical/src/lib/detection/rppg.ts',
        'constants': 'apps/pulselens-classical/src/lib/utils/constants.ts',
    }
    fixtures = []
    for count in [150, 256, 400]:
        for rate in [48, 60, 72, 90, 120, 150, 180]:
            t = np.arange(count) / 30
            pulse = np.sin(2 * np.pi * rate / 60 * t + .2)
            light = 1 + .03 * np.sin(2 * np.pi * .11 * t)
            rgb = np.column_stack([130 * light + .1 * pulse, 100 * light + .8 * pulse, 75 * light - .1 * pulse])
            fixtures.append(dict(rate=rate, rgb=rgb.tolist()))
    with tempfile.TemporaryDirectory(prefix='pulselens-reference-') as folder:
        folder = Path(folder)
        for name, path in files.items():
            source = subprocess.check_output(['git', '-C', str(checkout), 'show', f'{REVISION}:{path}'], text=True)
            (folder / f'{name}.ts').write_text(source)
        (folder / 'fixtures.json').write_text(json.dumps(fixtures))
        runner = r'''
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { pathToFileURL } from 'node:url';
const folder = process.argv[1];
for (const name of ['signal','spatial-beamformer','rppg','constants']) {
  let source = fs.readFileSync(`${folder}/${name}.ts`, 'utf8');
  source = source.replaceAll("'./signal'", "'./signal.mjs'")
    .replaceAll("'./spatial-beamformer'", "'./spatial-beamformer.mjs'")
    .replaceAll("'../utils/constants'", "'./constants.mjs'");
  fs.writeFileSync(`${folder}/${name}.mjs`, stripTypeScriptTypes(source, {mode:'transform'}));
}
globalThis.OffscreenCanvas = class { getContext() { return {}; } };
const {RppgDetector} = await import(pathToFileURL(`${folder}/rppg.mjs`).href);
const fixtures = JSON.parse(fs.readFileSync(`${folder}/fixtures.json`, 'utf8'));
const output = fixtures.map(fixture => {
  const detector = new RppgDetector();
  for (const rgb of fixture.rgb) {
    detector.bufferR.push(rgb[0]); detector.bufferG.push(rgb[1]); detector.bufferB.push(rgb[2]);
  }
  const result = detector.computeBpm(30);
  return result ? {bpm:result.rawBpm, quality:result.confidence} : null;
});
process.stdout.write(JSON.stringify(output));
'''
        upstream = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', runner, str(folder)], text=True))
    differences = []; quality_errors = []; decisions = []
    for fixture, expected in zip(fixtures, upstream):
        rows = [(i * 1000 / 30, rgb) for i, rgb in enumerate(fixture['rgb'])]
        result = estimate_rgb(rows)
        accepted = result if result and result['valid'] else None
        decisions.append(bool(accepted) == bool(expected))
        if accepted and expected:
            differences.append(abs(accepted['bpm'] - expected['bpm']))
            quality_errors.append(abs(accepted['quality'] - expected['quality']))
    report = dict(cases=len(fixtures), decisions_match=all(decisions), max_bpm_difference=max(differences, default=0),
                  max_confidence_difference=max(quality_errors, default=0))
    if not all(decisions) or report['max_bpm_difference'] > 1e-6 or report['max_confidence_difference'] > 1e-6:
        raise AssertionError(report)
    return report


def neural_comparison():
    adapter = FacePhys(); native = interpreter(ensure('model.tflite'))
    input_specs, output_specs = native.get_input_details(), native.get_output_details()
    initial = json.load(gzip.open(ensure('state.gz'), 'rt'))
    states = {spec['name']: np.asarray(initial[spec['name']], np.float32).reshape(spec['shape'])
              for spec in input_specs if spec['name'].startswith('state_in_')}
    pulse, reference_pulse = [], []; max_error = 0.; feature_error = 0.
    projection = interpreter(ensure('proj.tflite'))
    rng = np.random.default_rng(123)
    for frame in range(480):
        crop = rng.integers(90, 160, size=(36, 36, 3), dtype=np.uint8)
        value, features = adapter.step(crop, 1 / 30)
        for spec in input_specs:
            name = spec['name']
            data = np.array([1 / 30], np.float32) if name == 'dt' else crop.astype(np.float32).reshape(1, 1, 36, 36, 3) / 255 if name == 'input' else states[name]
            native.set_tensor(spec['index'], data)
        native.invoke()
        outputs = {spec['name']: native.get_tensor(spec['index']) for spec in output_specs}
        reference = float(outputs['Identity'].flat[0]); max_error = max(max_error, abs(value - reference))
        pulse.append(value); reference_pulse.append(reference)
        for name in states:
            number = int(name.removeprefix('state_in_'))
            states[name] = outputs[f'Identity_{number + 1}']
        if features:
            for spec in projection.get_input_details():
                state_name = spec['name'].removeprefix('serving_default_').split(':')[0]
                projection.set_tensor(spec['index'], states[state_name])
            projection.invoke()
            projected = {int(spec['name'].split(':')[-1]): projection.get_tensor(spec['index']) for spec in projection.get_output_details()}
            for index, key in enumerate(PROJECTION_KEYS):
                feature_error = max(feature_error, float(np.max(np.abs(projected[index].ravel() - features[key]))))
    learned = adapter.analyze(pulse[-450:], 1 / 30)
    def run_model(name):
        runtime = interpreter(ensure(name))
        runtime.set_tensor(runtime.get_input_details()[0]['index'], np.asarray(reference_pulse[-450:], np.float32).reshape(1, 450))
        runtime.invoke()
        return {int(spec['name'].split(':')[-1]): runtime.get_tensor(spec['index']) for spec in runtime.get_output_details()}
    quality = float(run_model('sqi_model.tflite')[0].flat[0]); rate = float(run_model('psd_model.tflite')[0].flat[0])
    report = dict(frames=480, max_pulse_difference=max_error, max_projection_difference=feature_error,
                  sqi_difference=abs(learned['learnedSqi'] - quality), bpm_difference=abs(learned['bpm'] - rate))
    if max(report[key] for key in report if key != 'frames') > 1e-6:
        raise AssertionError(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True, help='External checkout of xan-antx/pulseLens')
    args = parser.parse_args()
    report = dict(scope='Controlled algorithm/model-contract comparison; not live human accuracy.', reference_revision=REVISION,
                  classical=classical_comparison(args.upstream), neural=neural_comparison())
    print(json.dumps(report, indent=2))
