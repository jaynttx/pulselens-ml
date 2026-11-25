"""Python adapter for pretrained FacePhys; weights are attributed in the notices."""
import gzip
import json
import numpy as np
from .assets import ensure

OUTPUT_FOR_INPUT = [None, None, 1, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 23,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 34, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    35, 42, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 2]
PROJECTION_INPUTS = [4, 5, 12, 15, 16, 23, 26, 27, 34, 37, 38, 40, 41, 46]
PROJECTION_KEYS = ['fm1', 'fm2', 'fm3', 'fm4', 'proj', 'ssm1', 'ssm2', 'ssm3', 'ssm4']


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError('Python neural runtime missing. Install requirements.txt in your virtual environment.') from error
    options = {'model_content': bytes(path)} if isinstance(path, (bytes, bytearray)) else {'model_path': str(path)}
    instance = Interpreter(num_threads=2, **options)
    instance.allocate_tensors()
    return instance


class Model:
    def __init__(self, runtime):
        self.runtime = runtime
        # LiteRT.js exposes tensors in lexical-name order; Python returns the
        # FlatBuffer order. Normalize explicitly before using the pinned contract.
        self.inputs = sorted(runtime.get_input_details(), key=lambda spec: spec['name'])
        self.outputs = sorted(runtime.get_output_details(), key=lambda spec: spec['name'])

    def run(self, values):
        if len(values) != len(self.inputs):
            raise ValueError('Model input count mismatch.')
        for spec, value in zip(self.inputs, values):
            array = np.asarray(value, dtype=spec['dtype'])
            if tuple(array.shape) != tuple(spec['shape']):
                raise ValueError(f"Tensor shape mismatch for {spec['name']}: {array.shape} != {spec['shape']}")
            self.runtime.set_tensor(spec['index'], np.ascontiguousarray(array))
        self.runtime.invoke()
        return [self.runtime.get_tensor(spec['index']).copy() for spec in self.outputs]


class FacePhys:
    def __init__(self, factory=interpreter):
        self.model = Model(factory(ensure('model.tflite')))
        self.projection = Model(factory(ensure('proj.tflite')))
        self.sqi = Model(factory(ensure('sqi_model.tflite')))
        self.psd = Model(factory(ensure('psd_model.tflite')))
        with gzip.open(ensure('state.gz'), 'rt') as stream:
            self.initial = json.load(stream)
        if len(self.model.inputs) != 48 or tuple(self.model.inputs[0]['shape']) != (1,) or tuple(self.model.inputs[1]['shape']) != (1, 1, 36, 36, 3):
            raise ValueError('Unexpected FacePhys tensor order; this runtime/model combination is not compatible.')
        for index in range(2, 48):
            output = self.model.outputs[OUTPUT_FOR_INPUT[index]]
            if tuple(output['shape']) != tuple(self.model.inputs[index]['shape']):
                raise ValueError('Recurrent state mapping does not match model tensor shapes.')
        self.reset()

    def reset(self):
        self.frames = 0
        self.values = []
        for index, spec in enumerate(self.model.inputs):
            value = self.initial.get(spec['name']) if index > 1 else None
            array = np.asarray(value, dtype=np.float32).reshape(spec['shape']) if value is not None else np.zeros(spec['shape'], np.float32)
            if index == 0:
                array[:] = 1 / 30
            self.values.append(array)

    def step(self, rgb, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid neural frame interval.')
        pixels = np.asarray(rgb, dtype=np.float32)
        if pixels.shape != (36, 36, 3) or not np.isfinite(pixels).all():
            raise ValueError('Expected a finite 36 by 36 RGB crop.')
        self.values[0] = np.array([dt], np.float32)
        self.values[1] = (pixels / 255).reshape(1, 1, 36, 36, 3)
        outputs = self.model.run(self.values)
        value = float(outputs[0].flat[0])
        if not np.isfinite(value):
            raise ValueError('Neural model returned a non-finite pulse.')
        for index in range(2, 48):
            self.values[index] = outputs[OUTPUT_FOR_INPUT[index]]
        self.frames += 1
        visualization = None
        if self.frames % 5 == 0:
            projected = self.projection.run([self.values[index] for index in PROJECTION_INPUTS])
            visualization = {key: value.ravel().tolist() for key, value in zip(PROJECTION_KEYS, projected)}
        return value, visualization

    def analyze(self, pulse, dt):
        samples = np.asarray(pulse, np.float32)
        if samples.shape != (450,) or not np.isfinite(samples).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Learned analysis requires 450 finite samples and valid timing.')
        quality = float(self.sqi.run([samples.reshape(1, 450)])[0].flat[0])
        spectral = self.psd.run([samples.reshape(1, 450)])
        bpm = float(spectral[0].flat[0] / (30 * dt))
        if not np.isfinite([quality, bpm]).all():
            raise ValueError('Learned postprocessing returned non-finite values.')
        frequency = spectral[1].ravel() / (30 * dt)
        powers = spectral[2].ravel()
        mask = (frequency >= 40 / 60) & (frequency <= 200 / 60)
        powers = powers[mask]
        return dict(bpm=bpm, frequency=bpm / 60, quality=float(np.clip(quality, 0, 1)), learnedSqi=quality,
                    valid=bool(quality > .38 and 40 <= bpm <= 200),
                    frequencies=frequency[mask].tolist(), powers=(powers / max(float(powers.max()) if len(powers) else 0, 1e-12)).tolist())
