/* Resident GPU statistics and images; final labels are read independently. */
var SPOTPOST = (function () {
  'use strict';

  async function create(device, shader) {
    var display, pipes = {};
    try {
      display = await SPOTDISPLAY.create(device);
      await Promise.all(['reduceSpot', 'finishSpotStats', 'rasterSpot', 'colorizeSpot'].map(async function (name) {
        pipes[name] = await device.createComputePipelineAsync({
          label: 'Spot ' + name, layout: 'auto', compute: { module: shader, entryPoint: name }
        });
      }));
    } catch (error) {
      if (display) await display.dispose();
      throw error;
    }

    var U = GPUBufferUsage;
    var rows = new Map(), owned = new Set(), rowContext = null;
    var activeReads = 0, disposed = false;

    function drop(...buffers) {
      for (var buffer of buffers) {
        if (buffer && owned.has(buffer)) {
          buffer.destroy();
          owned.delete(buffer);
        }
      }
    }

    function make(size, usage, label) {
      var buffer = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage, label });
      owned.add(buffer);
      return buffer;
    }

    function bind(pipe, entries) {
      return device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } }))
      });
    }

    function uniform(ctx, di, fi, li) {
      var row = getRow(ctx, di), field = ctx.depths[di].fields[fi];
      var data = new ArrayBuffer(64), ints = new Uint32Array(data), floats = new Float32Array(data);
      ints.set([(fi * ctx.lam.length + li) * row.groups, row.groups, ctx.lam.length, ctx.priPos], 0);
      var x = Number.isFinite(field.cx) ? field.cx : 0, y = Number.isFinite(field.cy) ? field.cy : 0;
      var cx = Math.fround(x), cy = Math.fround(y);
      floats.set([cx, cy, x - cx, y - cy], 4);
      floats[9] = row.width || 256;
      floats[10] = row.points[fi][li]?.offset || 0;
      ints.set([fi, ctx.depths[di].fields.length, li, ctx.depths[di].role === 'focus' ? 1 : 0], 12);
      return data;
    }

    function postParams(ctx, di, fi, li, label) {
      var params = make(64, U.UNIFORM | U.COPY_DST, label);
      device.queue.writeBuffer(params, 0, uniform(ctx, di, fi, li));
      return params;
    }

    // The worker uses contexts sequentially. Point and trace configuration
    // buffers belong to the task producer; postprocessing owns only its inputs.
    function retireRow(row) {
      drop(row.partial, row.stats, row.pendingStats && row.pendingStats.params, ...row.reduceParams);
      row.reduceParams = [];
      row.partial = null;
      row.stats = null;
      row.pendingStats = null;
    }

    function selectContext(ctx) {
      if (disposed) throw Error('Postprocessing has been disposed');
      if (rowContext !== null && rowContext !== ctx) {
        if (activeReads) throw Error('Postprocessing context replacement requires completed statistics and raster reads');
        for (var row of rows.values()) retireRow(row);
        rows.clear();
      }
      rowContext = ctx;
    }

    function sequentialRead(fn) {
      return async function (ctx, ...args) {
        selectContext(ctx);
        if (activeReads) throw Error('Postprocessing reads must be sequential');
        activeReads++;
        try { return await fn(ctx, ...args); }
        finally { activeReads--; }
      };
    }

    function getRow(ctx, di) {
      selectContext(ctx);
      var side = Math.max(4, Math.round(Math.sqrt(ctx.R * 4 / Math.PI)));
      var count = side * side, groups = Math.ceil(count / 256), row = rows.get(di);
      if (row) return row;
      var nF = ctx.depths[di].fields.length, nL = ctx.lam.length;
      row = {
        groups, count, nF, nL,
        reduceParams: [], points: Array.from({ length: nF }, () => []),
        partial: make(groups * nF * nL * 32, U.STORAGE | U.COPY_DST, 'Spot row partials'),
        stats: make(nF * 32, U.STORAGE | U.COPY_SRC, 'Spot row stats')
      };
      rows.set(di, row);
      return row;
    }

    function reduce(ctx, di, fi, li, encoder, results, cfgBuffer, base) {
      var row = getRow(ctx, di);
      if (activeReads || row.pendingStats) throw Error('Spot reduction requires completed statistics and raster reads');
      row.summary = null;
      row.statsSubmitted = false;
      if (!row.partial) row.partial = make(row.groups * row.nF * row.nL * 32, U.STORAGE | U.COPY_DST, 'Spot row partials');
      if (!row.stats) row.stats = make(row.nF * 32, U.STORAGE | U.COPY_SRC, 'Spot row stats');
      var descriptor = results.buffer ? results : { buffer: results, offset: 0 };
      row.points[fi][li] = descriptor;
      var params = postParams(ctx, di, fi, li, 'Spot reduction parameters');
      row.reduceParams.push(params);
      var pass = encoder.beginComputePass({ label: 'Spot reduce' });
      pass.setPipeline(pipes.reduceSpot);
      pass.setBindGroup(0, bind(pipes.reduceSpot, [[3, descriptor.buffer], [4, cfgBuffer], [10, params], [11, row.partial]]));
      pass.dispatchWorkgroups(Math.ceil((base.count - base.first) / 256));
      pass.end();
    }

    // Finalization performs no map. GPU raster/display can follow immediately;
    // the small statistics read for final labels is a separate operation.
    function submitStats(ctx, di) {
      var row = getRow(ctx, di);
      if (activeReads) throw Error('Spot statistics submission requires completed statistics and raster reads');
      if (row.pendingStats) return row.stats;
      if (!row.partial) throw Error('Spot statistics require a fresh reduction');
      if (!row.stats) row.stats = make(row.nF * 32, U.STORAGE | U.COPY_SRC, 'Spot row stats');
      var params = postParams(ctx, di, 0, 0, 'Spot stats parameters');
      try {
        var encoder = device.createCommandEncoder(), pass = encoder.beginComputePass({ label: 'Spot finish stats' });
        pass.setPipeline(pipes.finishSpotStats);
        pass.setBindGroup(0, bind(pipes.finishSpotStats, [[10, params], [11, row.partial], [12, row.stats]]));
        pass.dispatchWorkgroups(row.nF);
        pass.end();
        device.queue.submit([encoder.finish()]);
      } catch (error) { drop(params); throw error; }
      row.pendingStats = { params };
      row.statsSubmitted = true;
      row.summary = null;
      return row.stats;
    }

    function releaseStatsInputs(row, pending) {
      if (row.pendingStats !== pending) return;
      drop(pending.params, row.partial, ...row.reduceParams);
      row.partial = null;
      row.reduceParams = [];
      row.pendingStats = null;
    }

    async function readStats(ctx, di) {
      var row = getRow(ctx, di), pending = row.pendingStats;
      if (!pending) {
        if (row.summary) return row.summary;
        throw Error('Spot label statistics require submitted finalization');
      }
      var size = row.nF * 32, read = make(size, U.MAP_READ | U.COPY_DST, 'Spot stats read');
      try {
        var encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(row.stats, 0, read, 0, size);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        var values = new Float32Array(read.getMappedRange(), 0, row.nF * 8), stats = [];
        for (var fi = 0; fi < row.nF; fi++) {
          var j = fi * 8, field = ctx.depths[di].fields[fi];
          stats.push({ th: field.th, cx: field.cx, cy: field.cy, diam: values[j], rms: values[j + 1], T: values[j + 2], valid: values[j + 3], nIn: values[j + 4], rays: values[j + 5] });
          ctx.nRays += values[j + 5];
        }
        var rowDiameter = 0;
        for (var stat of stats) rowDiameter = Math.max(rowDiameter, stat.diam);
        for (var stat of stats) stat.half = Math.max((ctx.depths[di].role === 'focus' ? stat.diam : rowDiameter) * (1.12 / 2000), 0.004);
        read.unmap();
        row.summary = stats;
        return stats;
      } finally {
        drop(read);
        releaseStatsInputs(row, pending);
      }
    }

    async function raster(ctx, di, fi, width, colors, weights) {
      if (!Number.isSafeInteger(width) || width < 1) throw Error('GPU display width must be a positive integer');
      var row = getRow(ctx, di);
      row.width = width;
      if (!row.statsSubmitted) throw Error('Spot raster requires submitted statistics');
      var bytes = row.nL * width * width * 4;

      var pixels, rgb, color, bitmap, temporary = [];
      try {
        pixels = make(bytes, U.STORAGE | U.COPY_DST, 'Spot integer pixel counts');
        rgb = make(width * width * 12, U.STORAGE, 'Spot RGB image');
        color = make(row.nL * 16, U.STORAGE | U.COPY_DST, 'Spot colors');
        var table = new Float32Array(row.nL * 4);
        for (var li = 0; li < row.nL; li++) table.set([...colors[li], weights[li]], li * 4);
        device.queue.writeBuffer(color, 0, table);
        var encoder = device.createCommandEncoder(), firstParams;
        encoder.clearBuffer(pixels);
        for (var li = 0; li < row.nL; li++) {
          var params = postParams(ctx, di, fi, li, 'Spot raster parameters');
          if (!firstParams) firstParams = params;
          temporary.push(params);
          var cfg = make(32, U.UNIFORM | U.COPY_DST, 'Spot raster count');
          temporary.push(cfg);
          var counts = new Uint32Array(8);
          counts[0] = row.count;
          device.queue.writeBuffer(cfg, 0, counts);
          var pass = encoder.beginComputePass({ label: 'Spot raster' });
          pass.setPipeline(pipes.rasterSpot);
          pass.setBindGroup(0, bind(pipes.rasterSpot, [[3, row.points[fi][li].buffer], [4, cfg], [10, params], [12, row.stats], [13, pixels]]));
          pass.dispatchWorkgroups(row.groups);
          pass.end();
        }
        var pass = encoder.beginComputePass({ label: 'Spot colorize' });
        pass.setPipeline(pipes.colorizeSpot);
        pass.setBindGroup(0, bind(pipes.colorizeSpot, [[10, firstParams], [13, pixels], [14, rgb], [15, color]]));
        pass.dispatchWorkgroups(Math.ceil(width * width / 256));
        pass.end();
        device.queue.submit([encoder.finish()]);
        bitmap = await display.render(rgb, width);
        return bitmap;
      } catch (error) {
        if (bitmap && typeof bitmap.close === 'function') bitmap.close();
        throw error;
      } finally {
        drop(pixels, rgb, color, ...temporary);
      }
    }

    function dispose() {
      if (activeReads) throw Error('Postprocessing disposal requires completed statistics and raster reads');
      if (disposed) return;
      disposed = true;
      for (var row of rows.values()) retireRow(row);
      for (var buffer of owned) buffer.destroy();
      owned.clear();
      rows.clear();
      rowContext = null;
      return display.dispose();
    }

    var readLabels = sequentialRead(readStats);
    return {
      reduce, submitStats, readStats: readLabels,
      raster: sequentialRead(raster),
      dispose
    };
  }

  return { create };
})();
