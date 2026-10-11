// Immutable CPU-prepared task descriptors select coherent field/wavelength
// inputs along the y dispatch axis. The packed host ABI is 176 bytes per task.
struct OpticalTask {
  config: Params,
  padding: vec2u,
  generation: GenParams,
  buffers: vec4u
}
@group(1) @binding(4) var<storage,read> opticalTasks: array<OpticalTask>;
@group(1) @binding(5) var<storage,read_write> taskAimJobs: array<atomic<u32>>;

fn selectOpticalTask(index:u32) {
  let task = opticalTasks[index];
  taskConfiguration = task.config;
  taskConfigurationEnabled = true;
  generationOverride = task.generation;
  generationOverrideEnabled = true;
  taskOffsetsStart = task.buffers.x;
  taskAimStart = task.buffers.y;
}

@compute @workgroup_size(64)
fn generateOpticalTasks(@builtin(global_invocation_id) gid:vec3u) {
  let task = gid.y;
  if(task >= arrayLength(&opticalTasks)) { return; }
  selectOpticalTask(task);
  let cell = gid.x;
  if(cell >= configuration().count) { return; }
  let out = opticalTasks[task].buffers.z + cell;
  let sample = sampleCell(cell);
  if(sample.status < 0.0) {
    generatedRays[out] = GeneratedRay(vec4f(0.0), vec4f(0.0, 0.0, 0.0, sample.status));
    return;
  }
  var q = sample.uv * (generation().geometry.w * 0.5 * 0.9995);
  if((generation().grid.z & 2u) != 0u) {
    let guess = pupilGuess(sample.uv);
    if(guess.ok) {
      q = guess.q;
    } else {
      generatedRays[out] = GeneratedRay(vec4f(sample.uv, 0.0, 0.0), vec4f(0.0, 0.0, 0.0, 2.0));
      let job = atomicAdd(&taskAimJobs[task], 1u);
      let base = arrayLength(&opticalTasks) + task * configuration().count;
      atomicStore(&taskAimJobs[base + job], cell);
      return;
    }
  }
  let ray = launchGenerated(q);
  generatedRays[out] = GeneratedRay(vec4f(ray.p, ray.t0), vec4f(ray.v, 1.0));
}

// Completed queue headers determine bounded indirect dispatches entirely on
// GPU. Every task has fixed slice slots; empty tails dispatch zero workgroups.
struct UniformQueueSchedule {
  tasks: u32,
  slices: u32,
  chunk: u32,
  padding: u32
}
@group(1) @binding(8) var<storage,read_write> taskUniformDispatch: array<atomic<u32>>;
@group(1) @binding(10) var<uniform> uniformQueueSchedule: UniformQueueSchedule;

@compute @workgroup_size(64)
fn prepareUniformTaskSliceDispatch(@builtin(global_invocation_id) gid:vec3u) {
  let slot = gid.x;
  if(slot >= uniformQueueSchedule.tasks * uniformQueueSchedule.slices) { return; }
  let task = slot / uniformQueueSchedule.slices;
  let slice = slot % uniformQueueSchedule.slices;
  let first = slice * uniformQueueSchedule.chunk;
  let jobs = min(atomicLoad(&taskAimJobs[task]), opticalTasks[task].config.count);
  var length = 0u;
  if(first < jobs) { length = min(uniformQueueSchedule.chunk, jobs - first); }
  atomicStore(&taskUniformDispatch[slot * 3u], (length + 63u) / 64u);
  atomicStore(&taskUniformDispatch[slot * 3u + 1u], 1u);
  atomicStore(&taskUniformDispatch[slot * 3u + 2u], 1u);
}

// Nonlinear aiming binds immutable uniforms for one task and one queue slice.
// The indirect buffer is not bound as storage during this pass.
struct UniformQueueControl {
  task: u32,
  tasks: u32,
  first: u32,
  extent: u32
}
@group(1) @binding(9) var<uniform> uniformQueueControl: UniformQueueControl;

@compute @workgroup_size(64)
fn aimOpticalUniformQueue(@builtin(global_invocation_id) gid:vec3u) {
  let task = uniformQueueControl.task;
  let job = uniformQueueControl.first + gid.x;
  if(gid.x >= uniformQueueControl.extent) { return; }
  if(job >= atomicLoad(&taskAimJobs[task])) { return; }
  let cell = atomicLoad(&taskAimJobs[uniformQueueControl.tasks + cfg.base + job]);
  if(cell >= cfg.count) { return; }
  let out = cfg.base + cell;
  if(generatedRays[out].v.w != 2.0) { return; }
  let uv = generatedRays[out].p.xy;
  let solution = aimGenerated(uv);
  if(solution.ok) {
    let ray = launchGenerated(solution.q);
    generatedRays[out] = GeneratedRay(vec4f(ray.p, ray.t0), vec4f(ray.v, 1.0));
  } else {
    generatedRays[out].v.w = 0.0;
  }
}

@compute @workgroup_size(64)
fn traceOpticalTasks(@builtin(global_invocation_id) gid:vec3u) {
  let task = gid.y;
  if(task >= arrayLength(&opticalTasks)) { return; }
  selectOpticalTask(task);
  let cell = configuration().base + gid.x;
  if(cell >= configuration().count) { return; }
  let out = opticalTasks[task].buffers.z + cell;
  let ray = generatedRays[out];
  let status = ray.v.w;
  if(status == -1.0 || status == -2.0) {
    results[out] = vec4f(0.0, 0.0, status, 0.0);
    return;
  }
  if(status != 1.0) {
    results[out] = vec4f(0.0);
    return;
  }
  let traced = traceOptics(ray.p.xyz, ray.v.xyz, ray.p.w, configuration().ns, true);
  if(traced.ok && abs(traced.v.z) >= 1e-12) {
    let xy = fma(vec2f((configuration().zImg - traced.p.z) / traced.v.z), traced.v.xy, traced.p.xy);
    results[out] = vec4f(xy, select(0.0, 1.0, finite(xy.x) && finite(xy.y)), 0.0);
  } else {
    results[out] = vec4f(0.0);
  }
}
