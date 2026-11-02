use reuse_streaming_dsp::StreamingDsp;

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn transforms_match_independent_numpy_fft_and_wola() {
    let frames = floats(include_bytes!("fixtures/frames.f32"));
    let spectra = floats(include_bytes!("fixtures/spectra.f32"));
    let expected_audio = floats(include_bytes!("fixtures/output.f32"));
    let mut dsp = StreamingDsp::new();
    let mut output = Vec::new();
    for (frame, expected) in frames.chunks_exact(640).zip(spectra.chunks_exact(642)) {
        let actual = dsp.analyze(frame).unwrap();
        for index in 0..321 {
            assert!((actual[index] - expected[index]).abs() < 1e-5);
            let delta = actual[index + 321] - expected[index + 321];
            assert!(delta.sin().atan2(delta.cos()).abs() < 0.003);
        }
        output.extend(dsp.synthesize(expected).unwrap());
    }
    assert_eq!(output.len(), expected_audio.len());
    let error = output
        .iter()
        .zip(expected_audio)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(error < 1e-6, "NumPy/WOLA maximum error {error}");
}
