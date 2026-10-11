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

    // A count histogram depends on points and scale, but not colors or weights.
    // The bound includes retired entries still used by an in-flight render.
    var histograms = new Map(), histBytes = 0, histLimit = 64 * 1024 * 1024;
    var frameProtected = null, framePanelCount = 0;

    function beginFrame(width, panels) {
      if (disposed) throw Error('Postprocessing has been disposed');
      if (activeReads) throw Error('Cannot change display frame while GPU read is pending');
      if (width !== undefined) {
        if (!Number.isInteger(width) || width < 1) throw Error('Invalid display frame width');
        for (var entry of histograms.values()) if (entry.width !== width) retireHistogram(entry);
      }
      if (panels !== undefined) {
        if (!Array.isArray(panels) || panels.some(p => !Array.isArray(p) || p.length !== 2 || p.some(n => !Number.isInteger(n) || n < 0))) {
          throw Error('Invalid display panel set');
        }
        var selected = new Set(panels.map(p => p.join('|')));
        framePanelCount = selected.size;
        for (var entry of histograms.values()) if (!selected.has(entry.di + '|' + entry.fi)) retireHistogram(entry);
      }
      frameProtected = new Set(histograms.values());
    }

    function retireHistogram(entry) {
      if (entry.stale) return;
      if (frameProtected) frameProtected.delete(entry);
      if (histograms.get(entry.key) === entry) histograms.delete(entry.key);
      entry.stale = true;
      if (!entry.users) {
        drop(entry.buffer);
        histBytes -= entry.bytes;
      }
    }

    function invalidateRowHistograms(row) {
      for (var entry of histograms.values()) if (entry.row === row) retireHistogram(entry);
    }

    function releaseHistogram(entry) {
      if (!entry) return;
      entry.users--;
      if (entry.stale && !entry.users) {
        drop(entry.buffer);
        histBytes -= entry.bytes;
      }
    }

    function reserveHistogram(bytes) {
      if (bytes > histLimit) return false;
      while (histBytes + bytes > histLimit) {
        var oldest = null;
        for (var entry of histograms.values()) {
          if (!entry.users && (!frameProtected || !frameProtected.has(entry))) {
            oldest = entry;
            break;
          }
        }
        if (!oldest) return false;
        retireHistogram(oldest);
      }
      return true;
    }

    function histogramMatches(entry, row, ctx, di, fi, width, half) {
      if (entry.pendingKey || entry.row !== row || entry.ctx !== ctx || entry.di !== di || entry.fi !== fi || entry.width !== width || entry.half !== half || entry.epoch !== row.epoch || entry.priPos !== ctx.priPos || entry.role !== ctx.depths[di].role) return false;
      var field = ctx.depths[di].fields[fi];
      if (entry.cx !== field.cx || entry.cy !== field.cy || entry.lam.length !== ctx.lam.length) return false;
      for (var li = 0; li < ctx.lam.length; li++) {
        var point = row.points[fi][li];
        if (entry.lam[li] !== ctx.lam[li] || !point || entry.points[li].buffer !== point.buffer || entry.points[li].offset !== (point.offset || 0)) return false;
      }
      return true;
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
      invalidateRowHistograms(row);
      drop(row.partial, row.stats, row.pendingStats && row.pendingStats.params, ...row.reduceParams);
      row.reduceParams = [];
      row.partial = null;
      row.stats = null;
      row.pendingStats = null;
    }

    function selectContext(ctx) {
      if (disposed) throw Error('Postprocessing has been disposed');
      var replacing = rowContext !== null && rowContext !== ctx;
      var shrinking = Array.from(rows.keys()).some(di => di >= ctx.depths.length);
      if ((replacing || shrinking) && activeReads) throw Error('Postprocessing context replacement requires completed statistics and raster reads');
      if (replacing) {
        for (var row of rows.values()) retireRow(row);
        rows.clear();
      } else if (shrinking) {
        for (var [di, row] of rows) {
          if (di >= ctx.depths.length) { retireRow(row); rows.delete(di); }
        }
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
      if (row) {
        if (row.count !== count || row.resultsGeneration !== (ctx.resultsGeneration || 0)) {
          if (activeReads || row.pendingStats) throw Error('Spot row changes require completed statistics and raster reads');
          if (row.count !== count) {
            drop(row.partial);
            row.partial = make(groups * row.nF * row.nL * 48, U.STORAGE | U.COPY_DST, 'Spot row partials');
            row.count = count;
            row.groups = groups;
          }
          row.resultsGeneration = ctx.resultsGeneration || 0;
          invalidateRowHistograms(row);
          row.epoch++;
        }
        return row;
      }
      var nF = ctx.depths[di].fields.length, nL = ctx.lam.length;
      row = {
        ctx, groups, count, nF, nL, epoch: 0, resultsGeneration: ctx.resultsGeneration || 0,
        reduceParams: [], points: Array.from({ length: nF }, () => []),
        partial: make(groups * nF * nL * 48, U.STORAGE | U.COPY_DST, 'Spot row partials'),
        stats: make(nF * 48, U.STORAGE | U.COPY_SRC, 'Spot row stats')
      };
      rows.set(di, row);
      return row;
    }

    function reduce(ctx, di, fi, li, encoder, results, cfgBuffer, base) {
      var row = getRow(ctx, di);
      if (activeReads || row.pendingStats) throw Error('Spot reduction requires completed statistics and raster reads');
      row.summary = null;
      row.statsSubmitted = false;
      if (!row.partial) row.partial = make(row.groups * row.nF * row.nL * 48, U.STORAGE | U.COPY_DST, 'Spot row partials');
      if (!row.stats) row.stats = make(row.nF * 48, U.STORAGE | U.COPY_SRC, 'Spot row stats');
      invalidateRowHistograms(row);
      row.epoch++;
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
      if (!row.stats) row.stats = make(row.nF * 48, U.STORAGE | U.COPY_SRC, 'Spot row stats');
      var params = postParams(ctx, di, 0, 0, 'Spot stats parameters');
      invalidateRowHistograms(row);
      row.epoch++;
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
      var size = row.nF * 48, read = make(size, U.MAP_READ | U.COPY_DST, 'Spot stats read');
      try {
        var encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(row.stats, 0, read, 0, size);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        var values = new Float32Array(read.getMappedRange(), 0, row.nF * 12), stats = [];
        var previous = row.lastRayCounts || new Float32Array(row.nF), nextCounts = new Float32Array(row.nF);
        for (var fi = 0; fi < row.nF; fi++) {
          var j = fi * 12, field = ctx.depths[di].fields[fi];
          stats.push({ th: field.th, cx: field.cx, cy: field.cy, diam: values[j], rms: values[j + 1], T: values[j + 2], valid: values[j + 3], nIn: values[j + 4], rays: values[j + 5] });
          nextCounts[fi] = values[j + 5];
          ctx.nRays += nextCounts[fi] - previous[fi];
        }
        row.lastRayCounts = nextCounts;
        var rowDiameter = 0;
        for (var stat of stats) rowDiameter = Math.max(rowDiameter, stat.diam);
        for (var stat of stats) stat.half = Math.max((ctx.depths[di].role === 'focus' ? stat.diam : rowDiameter) * (1.12 / 2000), 0.004);
        read.unmap();
        row.summary = stats;
        for (var entry of histograms.values()) {
          if (entry.row !== row || !entry.pendingKey) continue;
          if (entry.epoch !== row.epoch || entry.statsBuffer !== row.stats) { retireHistogram(entry); continue; }
          var stat = stats[entry.fi];
          entry.half = stat.half;
          entry.cx = stat.cx;
          entry.cy = stat.cy;
          entry.pendingKey = false;
        }
        return stats;
      } finally {
        drop(read);
        releaseStatsInputs(row, pending);
        if (!row.summary) for (var entry of histograms.values()) if (entry.row === row && entry.pendingKey) retireHistogram(entry);
      }
    }

    async function raster(ctx, di, fi, width, colors, weights) {
      if (!Number.isSafeInteger(width) || width < 1) throw Error('GPU display width must be a positive integer');
      var row = getRow(ctx, di);
      // Keep one displayed frame when it fits the cache. Allocation stays
      // lazy and capped at 192 MiB.
      var panels = framePanelCount || ctx.depths.reduce((sum, depth) => sum + depth.fields.length, 0);
      histLimit = Math.max(64 * 1024 * 1024, Math.min(192 * 1024 * 1024, panels * row.nL * width * width * 4));
      row.width = width;
      var hasSummary = !!(row.summary && row.summary[fi]);
      if (!hasSummary && !row.statsSubmitted) throw Error('Spot raster requires submitted statistics or a cached summary');
      var diameter = hasSummary ? row.summary[fi].diam : 0;
      if (hasSummary && ctx.depths[di].role !== 'focus') for (var stat of row.summary) diameter = Math.max(diameter, stat.diam);

      // The first render may precede label readback. Keep its cold counts, then
      // complete the cache key when readStats supplies the final display scale.
      var half = hasSummary ? Math.max(diameter * (1.12 / 2000), 0.004) : null;
      var key = di + '|' + fi + '|' + width, bytes = row.nL * width * width * 4, entry = histograms.get(key);
      if (entry && (!hasSummary || !histogramMatches(entry, row, ctx, di, fi, width, half))) { retireHistogram(entry); entry = null; }
      var hit = !!entry;
      if (entry) entry.users++;
      else if ((hasSummary || row.pendingStats) && reserveHistogram(bytes)) {
        var field = ctx.depths[di].fields[fi];
        entry = {
          key, ctx, row, di, fi, width, half, cx: field.cx, cy: field.cy,
          statsBuffer: row.stats, pendingKey: !hasSummary, priPos: ctx.priPos,
          role: ctx.depths[di].role, epoch: row.epoch, lam: Array.from(ctx.lam),
          points: row.points[fi].map(p => ({ buffer: p.buffer, offset: p.offset || 0 })),
          bytes, buffer: make(bytes, U.STORAGE | U.COPY_DST, 'Spot cached integer pixel counts'), users: 1, stale: false
        };
        histograms.set(key, entry);
        if (frameProtected) frameProtected.add(entry);
        histBytes += bytes;
      }

      var pixels, rgb, color, bitmap, temporary = [];
      try {
        pixels = entry ? entry.buffer : make(bytes, U.STORAGE | U.COPY_DST, 'Spot integer pixel counts');
        rgb = make(width * width * 12, U.STORAGE, 'Spot RGB image');
        color = make(row.nL * 16, U.STORAGE | U.COPY_DST, 'Spot colors');
        var table = new Float32Array(row.nL * 4);
        for (var li = 0; li < row.nL; li++) table.set([...colors[li], weights[li]], li * 4);
        device.queue.writeBuffer(color, 0, table);
        var encoder = device.createCommandEncoder(), firstParams;
        if (!hit) encoder.clearBuffer(pixels);
        if (hit) {
          firstParams = postParams(ctx, di, fi, 0, 'Spot colorize parameters');
          temporary.push(firstParams);
        } else {
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
        if (entry) retireHistogram(entry);
        throw error;
      } finally {
        releaseHistogram(entry);
        drop(rgb, color, ...temporary);
        if (!entry) drop(pixels);
      }
    }

    function dispose() {
      if (activeReads) throw Error('Postprocessing disposal requires completed statistics and raster reads');
      if (disposed) return;
      disposed = true;
      for (var row of rows.values()) retireRow(row);
      for (var buffer of owned) buffer.destroy();
      owned.clear();
      histograms.clear();
      frameProtected = null;
      histBytes = 0;
      rows.clear();
      rowContext = null;
      return display.dispose();
    }

    var readLabels = sequentialRead(readStats);
    return {
      beginFrame, reduce, submitStats, readStats: readLabels,
      raster: sequentialRead(raster),
      dispose
    };
  }

  return { create };
})();
