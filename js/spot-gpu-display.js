/* Resident RGB -> exact f32 GPU tone map -> ImageBitmap. */
var SPOTDISPLAY = (function () {
  'use strict';

  async function create(device) {
    if (typeof OffscreenCanvas === 'undefined') throw Error('GPU display requires OffscreenCanvas');
    if (typeof navigator === 'undefined' || !navigator.gpu || typeof navigator.gpu.getPreferredCanvasFormat !== 'function') {
      throw Error('GPU display requires a WebGPU canvas');
    }
    if (typeof SPOT_DISPLAY_SOURCE === 'undefined' || !SPOT_DISPLAY_SOURCE) throw Error('GPU display WGSL source is required');
    var module = device.createShaderModule({ label: 'Spot exact f32 display', code: SPOT_DISPLAY_SOURCE });
    if (module.getCompilationInfo) {
      var info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw Error(errors.map(m => m.lineNum + ':' + m.message).join('\n'));
    }
    var format = navigator.gpu.getPreferredCanvasFormat();
    var [bucket, choose, draw] = await Promise.all([
      device.createComputePipelineAsync({ label: 'Spot luminance radix bucket', layout: 'auto', compute: { module, entryPoint: 'bucketLuminance' } }),
      device.createComputePipelineAsync({ label: 'Spot luminance radix choose', layout: 'auto', compute: { module, entryPoint: 'chooseLuminanceBucket' } }),
      device.createRenderPipelineAsync({
        label: 'Spot direct GPU display', layout: 'auto', vertex: { module, entryPoint: 'displayVertex' },
        fragment: { module, entryPoint: 'displayFragment', targets: [{ format }] }, primitive: { topology: 'triangle-list' }
      })
    ]);
    var canvas = new OffscreenCanvas(1, 1), context = canvas.getContext('webgpu');
    if (!context) throw Error('OffscreenCanvas WebGPU context is unavailable');
    var U = GPUBufferUsage, state, bins, params = [], parameterWidth = 0;
    try {
      context.configure({ device, format, alphaMode: 'opaque', colorSpace: 'srgb' });
      state = device.createBuffer({ label: 'Spot display radix state', size: 16, usage: U.STORAGE | U.COPY_DST });
      bins = device.createBuffer({ label: 'Spot display radix bins', size: 256 * 4, usage: U.STORAGE | U.COPY_DST });
      for (var round = 0; round < 4; round++) {
        params.push(device.createBuffer({ label: 'Spot display radix parameters', size: 16, usage: U.UNIFORM | U.COPY_DST }));
      }
    } catch (error) {
      if (state) state.destroy();
      if (bins) bins.destroy();
      for (var parameter of params) parameter.destroy();
      if (context.unconfigure) context.unconfigure();
      throw error;
    }
    var disposed = false, disposePromise, lost = null, tail = Promise.resolve();
    if (device.lost && typeof device.lost.then === 'function') {
      device.lost.then(info => { lost = info.reason + ': ' + info.message; });
    }

    function assertAvailable() {
      if (disposed) throw Error('GPU display has been disposed');
      if (lost) throw Error('GPU display device lost: ' + lost);
    }

    function bind(pipe, group, entries) {
      return device.createBindGroup({
        layout: pipe.getBindGroupLayout(group),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } }))
      });
    }

    async function renderNow(rgbBuffer, width) {
      assertAvailable();
      if (!Number.isSafeInteger(width) || width < 1) throw Error('GPU display width must be a positive integer');
      var pixels = width * width, groups = Math.ceil(pixels / 256);
      var maxGroups = device.limits && device.limits.maxComputeWorkgroupsPerDimension || 65535;
      if (pixels > Math.floor(0xffffffff / 99) || groups > maxGroups || device.limits && width > device.limits.maxTextureDimension2D) {
        throw Error('GPU display size exceeds radix or device limits');
      }
      if (!rgbBuffer || rgbBuffer.size < pixels * 12 || !(rgbBuffer.usage & U.STORAGE)) {
        throw Error('GPU display requires a resident RGB storage buffer');
      }
      if (canvas.width !== width || canvas.height !== width) { canvas.width = width; canvas.height = width; }
      // Four immutable stage buffers survive renders. Only a size change
      // needs uploads; queue order protects their previously submitted uses.
      if (parameterWidth !== width) {
        for (var round = 0; round < 4; round++) {
          device.queue.writeBuffer(params[round], 0, new Uint32Array([width, 24 - round * 8, round, 0]));
        }
        parameterWidth = width;
      }
      var encoder = device.createCommandEncoder({ label: 'Spot direct display' });
      encoder.clearBuffer(state);
      for (var stage = 0; stage < 4; stage++) {
        encoder.clearBuffer(bins);
        var histogram = encoder.beginComputePass({ label: 'Spot radix bucket ' + stage });
        histogram.setPipeline(bucket);
        histogram.setBindGroup(0, bind(bucket, 0, [[0, rgbBuffer], [1, state], [2, bins], [3, params[stage]]]));
        histogram.dispatchWorkgroups(groups);
        histogram.end();
        var selected = encoder.beginComputePass({ label: 'Spot radix choose ' + stage });
        selected.setPipeline(choose);
        selected.setBindGroup(0, bind(choose, 0, [[1, state], [2, bins], [3, params[stage]]]));
        selected.dispatchWorkgroups(1);
        selected.end();
      }
      var target = context.getCurrentTexture();
      var pass = encoder.beginRenderPass({
        label: 'Spot exact f32 tone',
        colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }]
      });
      pass.setPipeline(draw);
      pass.setBindGroup(1, bind(draw, 1, [[0, rgbBuffer], [1, state], [2, params[3]]]));
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (lost) throw Error('GPU display device lost: ' + lost);
      var bitmap = canvas.transferToImageBitmap();
      return bitmap;
    }

    function render(rgbBuffer, width) {
      var task = tail.then(() => renderNow(rgbBuffer, width));
      tail = task.catch(() => {});
      return task;
    }

    function dispose() {
      if (disposePromise) return disposePromise;
      disposed = true;
      disposePromise = tail.then(function () {
        try { if (context.unconfigure) context.unconfigure(); }
        finally { state.destroy(); bins.destroy(); for (var parameter of params) parameter.destroy(); }
      });
      return disposePromise;
    }

    return { render, dispose };
  }

  return { create };
})();
