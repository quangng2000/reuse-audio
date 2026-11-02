use crate::fourier::SIZE;
use rustfft::num_complex::Complex32;

const HOP: usize = 320;

pub struct Overlap {
    audio: [f32; SIZE],
    envelope: [f32; SIZE],
    first: bool,
}

impl Overlap {
    pub fn new() -> Self {
        Self {
            audio: [0.0; SIZE],
            envelope: [0.0; SIZE],
            first: true,
        }
    }

    pub fn render(&mut self, buffer: &[Complex32], window: &[f32; SIZE]) -> Vec<f32> {
        for index in 0..SIZE {
            self.audio[index] += buffer[index].re * window[index];
            self.envelope[index] += window[index] * window[index];
        }
        let result = if self.first {
            vec![]
        } else {
            (0..HOP)
                .map(|index| self.audio[index] / self.envelope[index].max(1e-12))
                .collect()
        };
        self.shift();
        result
    }

    fn shift(&mut self) {
        self.audio.copy_within(HOP.., 0);
        self.envelope.copy_within(HOP.., 0);
        self.audio[HOP..].fill(0.0);
        self.envelope[HOP..].fill(0.0);
        self.first = false;
    }
}
