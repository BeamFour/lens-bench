/* The single Spot GPU backend: GPU depth search, CPU f64 aim tables,
 * GPU bounded ray tracing/statistics/display. Optical work never falls back. */
var SPOTGPU = (function () {
  'use strict';

  async function create(opt) {
    if (!navigator.gpu) throw new Error('当前浏览器 / Worker 没有 WebGPU');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('没有可用的 WebGPU adapter');
    const device = await adapter.requestDevice();
    let lost = null, validationFailure = null, disposed = false;
    let post, preparation, tasks, cache;
    device.addEventListener('uncapturederror', event => {
      validationFailure ||= event.error.message;
    });
    device.lost.then(info => {
      lost = info.message || info.reason;
      if (cache) cache.clear();
    });
    function assertAvailable() {
      if (disposed) throw new Error('GPU Spot 已关闭');
      if (lost) throw new Error('WebGPU device lost: ' + lost);
      if (validationFailure) throw new Error(validationFailure);
    }
    function dispose() {
      if (disposed) return;
      disposed = true;
      if (cache) cache.clear();
      device.destroy();
      if (tasks) tasks.dispose({ abandon: true });
      if (post) post.dispose();
      if (preparation) preparation.dispose();
    }
    try {
      const shader = device.createShaderModule({ label: 'Spot optical kernels', code: SPOT_GPU_SOURCE });
      // Keep each difficult aiming dispatch on immutable uniforms. This avoids
      // carrying task descriptor state through the Newton loop.
      const uniformCode = SPOT_GPU_SOURCE
        .replace(/fn generation\(\)->GenParams \{[\s\S]*?return gen;\s*\}/, 'fn generation()->GenParams {return gen;}')
        .replace('fn configuration()->Params { if(taskConfigurationEnabled){return taskConfiguration;}return cfg;}', 'fn configuration()->Params {return cfg;}')
        .replace('fn surfaceOffset(i:u32)->u32{return offsets[i+select(0u,taskOffsetsStart,taskConfigurationEnabled)];}', 'fn surfaceOffset(i:u32)->u32{return offsets[i];}');
      const uniformShader = device.createShaderModule({ label: 'Spot immutable aiming', code: uniformCode });
      // Every compute/render pipeline is compiled here, including display.
      // The page calls create while idle, before the first Generate action.
      post = await SPOTPOST.create(device, shader);
      preparation = await SPOTPREP.create(opt, { device, shader });
      tasks = await SPOTTASK.create(opt, { device, shader, uniformShader, post });
      cache = SPOTCACHE.create(opt, preparation);
      assertAvailable();
      async function readStats(ctx, di) {
        assertAvailable();
        try {
          const result = await post.readStats(ctx, di);
          tasks.markStatsReady(ctx, di);
          assertAvailable();
          return result;
        } catch (error) {
          await tasks.settle().catch(() => {});
          throw error;
        }
      }
      return {
        buildSystem: preparation.buildSystem,
        defocusDepths: async (...args) => { assertAvailable(); return cache.defocusDepths(...args); },
        spotPrepare: async (...args) => { assertAvailable(); return cache.spotPrepare(...args); },
        spotTraceRow: async (...args) => { assertAvailable(); await tasks.traceRow(...args); },
        submitStatsRow: (...args) => { assertAvailable(); return post.submitStats(...args); },
        readStatsRow: readStats,
        spotRaster: async (...args) => {
          assertAvailable();
          const image = await post.raster(...args);
          try { assertAvailable(); return image; }
          catch (error) { image.close(); throw error; }
        },
        dispose
      };
    } catch (error) {
      dispose();
      throw error;
    }
  }
  return { create };
})();
