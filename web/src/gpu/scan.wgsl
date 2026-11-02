// NVIDIA Real-time RE-USE selective scan, NSCLv1; Mamba recurrence Apache-2.0.
// One channel per invocation; no assumed hardware subgroup width or FP16.
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read> dt: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> z: array<f32>;
@group(0) @binding(5) var<storage, read> a: array<f32>;
@group(0) @binding(6) var<storage, read> d: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channel = id.x;
  if (channel >= 256u) { return; }
  var state: array<f32, 16>;
  for (var bin = 0u; bin < 161u; bin++) {
    let index = bin * 256u + channel;
    var products: array<f32, 16>;
    for (var component = 0u; component < 16u; component++) {
      let decay = exp(dt[index] * a[channel * 16u + component]);
      let update = (dt[index] * b[bin * 16u + component]) * u[index];
      state[component] = decay * state[component] + update;
      products[component] = state[component] * c[bin * 16u + component];
    }
    for (var stride = 8u; stride > 0u; stride = stride / 2u) {
      for (var component = 0u; component < stride; component++) {
        products[component] = products[component] + products[component + stride];
      }
    }
    let gate = z[index] / (1.0 + exp(-z[index]));
    output[index] = (products[0] + d[channel] * u[index]) * gate;
  }
}
