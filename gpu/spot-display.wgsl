// GPU display of the resident linear RGB image.
// The 99th percentile is the exact order statistic of positive f32
// luminances produced by luminance(); it is not bit-identical to JS f64.
struct DisplayParams {
  width: u32,
  shift: u32,
  round: u32,
  pad: u32,
}

// Four u32 words: positive count, rank within selected prefix, selected
// prefix bits, and final threshold bits. Positive IEEE-754 f32 bits sort
// in the same order as their values, including +infinity.
@group(0) @binding(0) var<storage, read> bucketRgb: array<f32>;
@group(0) @binding(1) var<storage, read_write> radixState: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> radixBins: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> bucketParams: DisplayParams;

fn luminance(pixel: u32) -> f32 {
  let index = pixel * 3u;
  return (bucketRgb[index] + bucketRgb[index + 1u] + bucketRgb[index + 2u]) / 3.0;
}

// Aggregate each tile before touching global counters. All invocations reach
// both barriers, including image padding and pixels outside the chosen prefix.
var<workgroup> tileBins: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn bucketLuminance(@builtin(global_invocation_id) gid: vec3u,
                   @builtin(local_invocation_index) lane: u32) {
  // Small images and later sparse radix passes use direct global atomics.
  // Tile clearing/barriers cost more than the contention they remove there.
  if (bucketParams.round != 0u || bucketParams.width < 512u) {
    let pixel = gid.x;
    if (pixel < bucketParams.width * bucketParams.width) {
      let lum = luminance(pixel);
      if (lum > 0.0) {
        let bits = bitcast<u32>(lum);
        if (bucketParams.round == 0u) {
          atomicAdd(&radixState[0], 1u);
          atomicAdd(&radixBins[(bits >> bucketParams.shift) & 255u], 1u);
        } else {
          let prefixBits = bucketParams.shift + 8u;
          if ((bits >> prefixBits) == (atomicLoad(&radixState[2]) >> prefixBits)) {
            atomicAdd(&radixBins[(bits >> bucketParams.shift) & 255u], 1u);
          }
        }
      }
    }
    return;
  }
  atomicStore(&tileBins[lane], 0u);
  workgroupBarrier();
  let pixel = gid.x;
  if (pixel < bucketParams.width * bucketParams.width) {
    let lum = luminance(pixel);
    if (lum > 0.0) {
      let bits = bitcast<u32>(lum);
      atomicAdd(&tileBins[(bits >> bucketParams.shift) & 255u], 1u);
    }
  }
  workgroupBarrier();
  let amount = atomicLoad(&tileBins[lane]);
  if (amount != 0u) {
    atomicAdd(&radixBins[lane], amount);
    atomicAdd(&radixState[0], amount);
  }
}

@compute @workgroup_size(1)
fn chooseLuminanceBucket() {
  let count = atomicLoad(&radixState[0]);
  if (count == 0u) {
    atomicStore(&radixState[3], bitcast<u32>(1.0));
    return;
  }
  // Equivalent to floor(N * 0.99) for supported image sizes, without
  // losing integer precision to f32. The index is zero based.
  var rank = atomicLoad(&radixState[1]);
  if (bucketParams.round == 0u) { rank = (count * 99u) / 100u; }
  var skipped = 0u;
  var chosen = 255u;
  for (var bin = 0u; bin < 256u; bin++) {
    let amount = atomicLoad(&radixBins[bin]);
    if (skipped + amount > rank) {
      chosen = bin;
      break;
    }
    skipped += amount;
  }
  let prefix = atomicLoad(&radixState[2]) | (chosen << bucketParams.shift);
  atomicStore(&radixState[1], rank - skipped);
  atomicStore(&radixState[2], prefix);
  if (bucketParams.round == 3u) {
    atomicStore(&radixState[3], bitcast<u32>(max(bitcast<f32>(prefix), 1e-30)));
  }
}

@group(1) @binding(0) var<storage, read> displayRgb: array<f32>;
@group(1) @binding(1) var<storage, read> displayState: array<u32>;
@group(1) @binding(2) var<uniform> displayParams: DisplayParams;

@vertex
fn displayVertex(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
  let x = select(-1.0, 3.0, vertex == 1u);
  let y = select(-1.0, 3.0, vertex == 2u);
  return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn displayFragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let x = u32(position.x);
  let y = u32(position.y);
  let pixel = y * displayParams.width + x;
  let index = pixel * 3u;
  let raw = vec3f(displayRgb[index], displayRgb[index + 1u], displayRgb[index + 2u]);
  let threshold = bitcast<f32>(displayState[3]);
  let mapped = min(vec3f(1.0), pow(max(vec3f(0.0), raw / threshold), vec3f(0.6)));
  // floor(x + 0.5) matches Math.round for the nonnegative display channels.
  let bytes = floor(255.0 * mapped + vec3f(0.5));
  return vec4f(bytes / 255.0, 1.0);
}
