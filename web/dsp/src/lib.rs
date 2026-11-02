//! Research-only RE-USE streaming DSP. NVIDIA NSCLv1; see LICENSE and NOTICE.
mod fourier;
mod hann;
mod overlap;
mod spectrum;

use fourier::{Fourier, BINS, SIZE};
use overlap::Overlap;
use rustfft::num_complex::Complex32;
use wasm_bindgen::prelude::*;

/// Mono 16 kHz, FFT 640, hop 320. Input framing and resampling belong to the host.
#[wasm_bindgen]
pub struct StreamingDsp {
    fourier: Fourier,
    window: [f32; SIZE],
    overlap: Overlap,
}

#[wasm_bindgen]
impl StreamingDsp {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            fourier: Fourier::new(),
            window: hann::window(),
            overlap: Overlap::new(),
        }
    }

    /// Exactly 640 PCM samples; returns log-magnitude then phase, 321 values each.
    pub fn analyze(&mut self, frame: &[f32]) -> Result<Vec<f32>, String> {
        spectrum::validate(frame, SIZE)?;
        let mut buffer: Vec<_> = frame
            .iter()
            .zip(self.window)
            .map(|(value, window)| Complex32::new(value * window, 0.0))
            .collect();
        self.fourier.forward(&mut buffer);
        spectrum::compress(&buffer)
    }

    /// Empty on the first frame, then 320 PCM samples per frame. No peak scaling.
    pub fn synthesize(&mut self, spectrum: &[f32]) -> Result<Vec<f32>, String> {
        spectrum::validate(spectrum, 2 * BINS)?;
        let mut buffer = spectrum::expand(spectrum)?;
        self.fourier.inverse(&mut buffer);
        if buffer.iter().any(|value| !value.re.is_finite()) {
            return Err("Nonfinite inverse FFT; reset the audio stream".into());
        }
        let output = self.overlap.render(&buffer, &self.window);
        if output.iter().any(|value| !value.is_finite()) {
            return Err("Nonfinite overlap-add; reset the audio stream".into());
        }
        Ok(output)
    }

    /// Reset at a new recording or after a discontinuity; also reset model caches.
    pub fn reset(&mut self) {
        self.overlap = Overlap::new();
    }
}

impl Default for StreamingDsp {
    fn default() -> Self {
        Self::new()
    }
}
