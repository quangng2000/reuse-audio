"""Independent NumPy FFT/WOLA with sealed PyTorch Hann; no model weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import re

DESTINATION = Path(__file__).parent / "fixtures"


def write(name, values):
    values = np.asarray(values, dtype="<f4")
    path = DESTINATION / name
    path.write_bytes(values.tobytes())
    return {"shape": list(values.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def make_frames():
    time = np.arange(1600) / 16000
    signal = (.17 * np.sin(2 * np.pi * 173 * time) + .08 * np.sin(2 * np.pi * 917 * time)
              + .013 * np.random.default_rng(421).standard_normal(len(time))).astype(np.float32)
    padded = np.concatenate((signal[1:321][::-1], signal, signal[:-1][::-1]))
    frames = np.stack([padded[i*320:i*320+640] for i in range(6)])
    return signal, frames


def synthesize(spectra, window):
    overlap, envelope = np.zeros(640, np.float32), np.zeros(640, np.float32)
    output = []
    for index, spectrum in enumerate(spectra):
        magnitude, phase = np.expm1(np.maximum(spectrum[:321], 0)), spectrum[321:]
        if (magnitude == 0).sum() > 160:
            magnitude[:] = 0
        audio = np.fft.irfft(magnitude * (np.cos(phase) + 1j * np.sin(phase)), n=640).astype(np.float32)
        overlap += audio * window
        envelope += window * window
        if index:
            output.append((overlap[:320] / np.maximum(envelope[:320], 1e-12)).copy())
        overlap, envelope = np.concatenate((overlap[320:], np.zeros(320, np.float32))), np.concatenate((envelope[320:], np.zeros(320, np.float32)))
    return np.concatenate(output)


def main():
    DESTINATION.mkdir(exist_ok=True)
    signal, frames = make_frames()
    source = (DESTINATION.parent.parent / "src/hann.rs").read_text()
    window = np.array([int(value, 16) for value in re.findall(r"0x[0-9a-f]{8}", source)], dtype="<u4").view("<f4")
    complex_spectrum = np.fft.rfft(frames * window)
    spectra = np.concatenate((np.log1p(np.abs(complex_spectrum)), np.angle(complex_spectrum)), axis=1).astype(np.float32)
    files = {"frames.f32": write("frames.f32", frames),
             "spectra.f32": write("spectra.f32", spectra),
             "output.f32": write("output.f32", synthesize(spectra, window)),
             "input.f32": write("input.f32", signal)}
    report = {"source": "independent NumPy CPU FFT/WOLA; sealed PyTorch Hann", "numpy": np.__version__,
              "seed": 421, "frames": 6, "files": files, "scope": "DSP only; no model inference"}
    (DESTINATION / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
