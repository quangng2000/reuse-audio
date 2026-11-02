use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

pub const SIZE: usize = 640;
pub const BINS: usize = 321;

pub struct Fourier {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex32>,
}

impl Fourier {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(SIZE);
        let inverse = planner.plan_fft_inverse(SIZE);
        let length = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        Self {
            forward,
            inverse,
            scratch: vec![Complex32::default(); length],
        }
    }

    pub fn forward(&mut self, buffer: &mut [Complex32]) {
        self.forward.process_with_scratch(buffer, &mut self.scratch);
    }

    pub fn inverse(&mut self, buffer: &mut [Complex32]) {
        self.inverse.process_with_scratch(buffer, &mut self.scratch);
        for value in buffer {
            *value /= SIZE as f32;
        }
    }
}
