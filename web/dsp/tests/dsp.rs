use reuse_streaming_dsp::StreamingDsp;

fn signal(length: usize) -> Vec<f32> {
    let mut seed = 421u32;
    (0..length)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / 16777216.0 - 0.5) * 0.3
        })
        .collect()
}

#[test]
fn analyze_impulse_matches_analytical_fourier_transform() {
    let mut dsp = StreamingDsp::new();
    let mut frame = vec![0.0; 640];
    frame[320] = 1.0;
    let spectrum = dsp.analyze(&frame).unwrap();
    for index in 0..321 {
        assert!((spectrum[index] - 2.0f32.ln()).abs() < 1e-6);
        let expected = if index % 2 == 0 { 1.0 } else { -1.0 };
        assert!((spectrum[321 + index].cos() - expected).abs() < 1e-6);
    }
}

#[test]
fn centered_wola_round_trip_preserves_broadband_audio() {
    let input = signal(1600);
    let padded: Vec<_> = input[1..321]
        .iter()
        .rev()
        .chain(&input)
        .chain(input[..input.len() - 1].iter().rev())
        .copied()
        .collect();
    let mut dsp = StreamingDsp::new();
    let mut output = Vec::new();
    for index in 0..6 {
        let spectrum = dsp
            .analyze(&padded[index * 320..index * 320 + 640])
            .unwrap();
        output.extend(dsp.synthesize(&spectrum).unwrap());
    }
    assert_eq!(output.len(), input.len());
    let error = input
        .iter()
        .zip(output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(error < 1e-6, "round-trip maximum error {error}");
}

#[test]
fn reset_restores_first_hop_suppression_and_state() {
    let mut dsp = StreamingDsp::new();
    let spectrum = dsp.analyze(&signal(640)).unwrap();
    assert!(dsp.synthesize(&spectrum).unwrap().is_empty());
    let before = dsp.synthesize(&spectrum).unwrap();
    dsp.reset();
    assert!(dsp.synthesize(&spectrum).unwrap().is_empty());
    assert_eq!(before, dsp.synthesize(&spectrum).unwrap());
}

#[test]
fn majority_empty_bins_suppress_sweep_artifacts() {
    let mut dsp = StreamingDsp::new();
    let mut spectrum = vec![0.0; 642];
    spectrum[..160].fill(0.3);
    assert!(dsp.synthesize(&spectrum).unwrap().is_empty());
    assert!(dsp
        .synthesize(&spectrum)
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
}

#[test]
fn invalid_and_nonfinite_inputs_are_rejected() {
    let mut dsp = StreamingDsp::new();
    assert!(dsp.analyze(&[0.0; 639]).is_err());
    assert!(dsp.analyze(&[f32::NAN; 640]).is_err());
    assert!(dsp.synthesize(&[0.0; 641]).is_err());
    assert!(dsp.synthesize(&[f32::INFINITY; 642]).is_err());
    assert!(dsp.synthesize(&[f32::MAX; 642]).is_err());
    assert!(dsp.synthesize(&[0.0; 642]).unwrap().is_empty());
}
