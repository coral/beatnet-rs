#!/usr/bin/env python3
"""Maintainer-only export. Run with the pinned reference environment, not at build time."""
import argparse
import collections
import collections.abc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
# Compatibility shims confined to this tool for madmom 0.16.1.
collections.MutableSequence = collections.abc.MutableSequence
np.float = float
np.int = int
np.complex = complex
import torch
import onnx
import onnxruntime as ort
from madmom.audio.signal import Signal, FramedSignal
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.audio.spectrogram import FilteredSpectrogram, LogarithmicSpectrogram, SpectrogramDifference
from madmom.features.beats_hmm import BarStateSpace, exponential_transition

ROOT = Path(__file__).resolve().parents[1]

def save(path, data, dtype='<f4'):
    np.asarray(data, dtype=dtype).tofile(ROOT / path)

def features(audio):
    frames = FramedSignal(Signal(audio, sample_rate=22050), frame_size=1411, hop_size=441)
    stft = ShortTimeFourierTransform(frames)
    filtered = FilteredSpectrogram(stft, num_bands=24, fmin=30, fmax=17000, norm_filters=True)
    log = LogarithmicSpectrogram(filtered, mul=1, add=1)
    diff = SpectrogramDifference(log, diff_ratio=.5, positive_diffs=True)
    return np.hstack([log, diff]).astype(np.float32), stft, filtered

class Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, features, hidden, cell):
        m = self.model
        x = torch.nn.functional.max_pool1d(torch.relu(m.conv1(features.reshape(1, 1, 272))), 2)
        x = m.linear0(x.flatten(1)).reshape(1, 1, 150)
        x, (hidden, cell) = m.lstm(x, (hidden, cell))
        return torch.softmax(m.linear(x).reshape(3), dim=0), hidden, cell

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    n = 441 * 200
    t = np.arange(n) / 22050
    signals = {'silence': np.zeros(n), 'impulse': np.eye(1, n, 2205).ravel(),
               'tone': .5 * np.sin(2*np.pi*440*t), 'noise': rng.normal(0,.15,n)}
    music = .15*np.sin(2*np.pi*110*t) + .08*np.sin(2*np.pi*277*t)
    for start in range(0,n,11025):
        end = min(start+2205,n)
        music[start:end] += .7*rng.normal(size=end-start)*np.exp(-np.arange(end-start)/300)
    signals['rhythm'] = music
    fixture_meta = {}
    for name, audio in signals.items():
        audio = audio.astype(np.float32)
        feat, stft, filt = features(audio)
        assert feat.shape[1] == 272
        save(f'tests/fixtures/{name}.pcm', audio)
        save(f'tests/fixtures/{name}.features', feat)
        fixture_meta[name] = {'samples':len(audio), 'frames':len(feat)}
    save('assets/window.f64', stft.fft_window, '<f8')
    save('assets/filterbank.f32', filt.filterbank)
    # Reference tables for the Rust state-space/transition implementation.
    state = BarStateSpace(1, 3000/215, 3000/55, 300)
    save('tests/fixtures/state_intervals.f32', state.state_intervals)
    intervals = state.state_intervals[state.first_states[0]]
    save('tests/fixtures/transitions.f64', exponential_transition(intervals, intervals, 60), '<f8')
    weights = np.array([0., .1, .2, .4, 0., .3], dtype=np.float64)
    draws = rng.random(17)
    selected = np.searchsorted(np.cumsum(weights) / weights.sum(), (np.arange(17) + draws) / 17)
    save('tests/fixtures/resample_weights.f64', weights, '<f8')
    save('tests/fixtures/resample_draws.f64', draws, '<f8')
    save('tests/fixtures/resample_indices.u32', selected, '<u4')
    spec = importlib.util.spec_from_file_location('beatnet_model', args.upstream/'src/BeatNet/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
                'sample_rate':22050,'hop':441,'window':1411,'filterbank_shape':list(filt.filterbank.shape),
                'fixtures':fixture_meta,'models':[]}
    for number in range(1,4):
        weights = args.upstream/f'src/BeatNet/models/model_{number}_weights.pt'
        model = module.BDA(272,150,2,'cpu').eval()
        model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        wrapper = Export(model).eval()
        path = ROOT/f'models/beatnet-{number}.onnx'
        with torch.no_grad():
            torch.onnx.export(wrapper,(torch.zeros(1,1,272),torch.zeros(2,1,150),torch.zeros(2,1,150)),str(path),
                              input_names=['features','hidden','cell'],output_names=['probabilities','hidden_out','cell_out'],
                              opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        ort.disable_telemetry_events()
        session = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        h = c = np.zeros((2,1,150),np.float32)
        ph = pc = torch.zeros(2,1,150)
        reference = []
        max_error = 0.
        with torch.no_grad():
            for frame_index in range(len(feat) * 10):
                f = feat[frame_index % len(feat)]
                p, ph, pc = wrapper(torch.from_numpy(f.reshape(1,1,272)),ph,pc)
                op,h,c = session.run(None,{'features':f.reshape(1,1,272),'hidden':h,'cell':c})
                for actual,expected in [(op,p.numpy()),(h,ph.numpy()),(c,pc.numpy())]:
                    np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-4)
                    max_error = max(max_error,float(np.max(np.abs(actual-expected))))
                if frame_index < len(feat):
                    reference.append(np.concatenate([p.numpy().ravel(),ph.numpy().ravel(),pc.numpy().ravel()]))
            # Also verify the wrapper against the unmodified upstream forward.
            model.hidden.zero_(); model.cell.zero_()
            original = model.final_pred(model(torch.from_numpy(feat[None]))[0]).T.numpy()
            np.testing.assert_allclose(original,np.asarray(reference)[:,:3],atol=5e-5,rtol=5e-4)
        save(f'tests/fixtures/model-{number}.reference', reference)
        manifest['models'].append({'number':number,'source_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                                  'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'max_export_error':max_error})
        print(f'model {number}: max export error {max_error:.3g}')
    (ROOT/'assets/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__':
    main()
