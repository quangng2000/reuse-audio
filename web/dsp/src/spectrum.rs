use crate::fourier::{BINS, SIZE};
use rustfft::num_complex::Complex32;

pub fn validate(values: &[f32], length: usize) -> Result<(), String> {
    if values.len() != length {
        return Err(format!("Expected {length} FP32 samples"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Audio and spectra must contain only finite values".into());
    }
    Ok(())
}

pub fn compress(buffer: &[Complex32]) -> Result<Vec<f32>, String> {
    let mut spectrum = vec![0.0; 2 * BINS];
    for index in 0..BINS {
        let value = buffer[index];
        let imaginary = if index == 0 || index == BINS - 1 {
            0.0
        } else {
            value.im
        };
        spectrum[index] = (value.re * value.re + value.im * value.im).sqrt().ln_1p();
        spectrum[BINS + index] = imaginary.atan2(value.re);
    }
    validate(&spectrum, 2 * BINS)?;
    Ok(spectrum)
}

pub fn expand(spectrum: &[f32]) -> Result<Vec<Complex32>, String> {
    let magnitudes: Vec<_> = spectrum[..BINS]
        .iter()
        .map(|value| value.max(0.0).exp_m1())
        .collect();
    validate(&magnitudes, BINS)?;
    let mut buffer = vec![Complex32::default(); SIZE];
    if magnitudes.iter().filter(|&&value| value == 0.0).count() > 160 {
        return Ok(buffer);
    }
    for index in 0..BINS {
        let phase = spectrum[BINS + index];
        buffer[index] = Complex32::new(
            magnitudes[index] * phase.cos(),
            magnitudes[index] * phase.sin(),
        );
        if index == 0 || index == BINS - 1 {
            buffer[index].im = 0.0;
        } else {
            buffer[SIZE - index] = buffer[index].conj();
        }
    }
    Ok(buffer)
}
