/* Bounded GPU tracing of CPU-prepared field/wavelength tasks. */
var SPOTTASK = (function () {
  'use strict';

  const TASK_BYTES = 176;
  const TASK_WORDS = TASK_BYTES / 4;

  async function create(opt, {
    device, shader, uniformShader, post,
    maxTasks = 20, queueChunk = 4096, traceChunk = 262144,
    scratchPoolMaxBytes = 192 * 1024 * 1024
  }) {
    if (!uniformShader) throw Error('Immutable uniform optical shader is required');
    if (!Number.isInteger(maxTasks) || maxTasks < 1) {
      throw Error('maxTasks must be a positive integer');
    }
    if (!Number.isInteger(queueChunk) || queueChunk < 1) {
      throw Error('queueChunk must be a positive integer');
    }
    if (!Number.isInteger(traceChunk) || traceChunk < 256) {
      throw Error('traceChunk must be at least 256 cells');
    }
    if (!Number.isSafeInteger(scratchPoolMaxBytes) || scratchPoolMaxBytes < 0) {
      throw Error('scratchPoolMaxBytes must be a nonnegative integer');
    }
    maxTasks = Math.min(maxTasks, 20);
    queueChunk = Math.min(queueChunk, 4096);

    const U = GPUBufferUsage;
    const pipelines = {};
    const entries = [
      'generateTraceOpticalTasks', 'prepareUniformTaskSliceDispatch',
      'aimOpticalUniformQueue'
    ];
    await Promise.all(entries.map(async entry => {
      pipelines[entry] = await device.createComputePipelineAsync({
        layout: 'auto', label: entry,
        compute: {
          module: entry === 'aimOpticalUniformQueue' ? uniformShader : shader,
          entryPoint: entry
        }
      });
    }));
    const buffers = new Map();
    const records = new Map();
    const retired = new Set();
    const pendingRows = new Set();
    let scratchBytes = 0;
    let activeContext = null, activeTrace = false, activeRow = null;
    let useSerial = 0, disposed = false;

    function makeBuffer(key, size, usage) {
      const category = key.startsWith('results-') ? 'points' : 'scratch';
      const b = device.createBuffer({size, usage, label: 'Spot tasks ' + key});
      records.set(b, {key, buffer: b, category, rows: new Set(), used: ++useSerial});
      if (category === 'scratch') scratchBytes += size;
      return b;
    }

    function useBuffer(b) {
      const record = records.get(b);
      if (record && activeRow !== null) {
        record.rows.add(activeRow);
        record.used = ++useSerial;
        pendingRows.add(activeRow);
      }
      return b;
    }

    function destroyBuffer(record) {
      if (record.rows.size) throw Error('Task buffer retirement requires completed statistics');
      record.buffer.destroy();
      records.delete(record.buffer);
      retired.delete(record);
      if (buffers.get(record.key) === record.buffer) buffers.delete(record.key);
      if (record.category === 'scratch') scratchBytes -= record.buffer.size;
    }

    function retireBuffer(b) {
      const record = records.get(b);
      if (!record) return;
      if (record.rows.size) {
        retired.add(record);
      } else {
        destroyBuffer(record);
      }
    }

    function buffer(key, size, usage, data) {
      let b = buffers.get(key);
      size = Math.max(4, Math.ceil(size / 4) * 4);
      if (!b || b.size < size) {
        if (b) retireBuffer(b);
        b = makeBuffer(key, size, usage);
        buffers.set(key, b);
      }
      useBuffer(b);
      if (data) device.queue.writeBuffer(b, 0, data);
      return b;
    }

    function requireIdle(action) {
      if (disposed) throw Error('Task backend is disposed');
      if (activeTrace || pendingRows.size) {
        throw Error(action + ' requires completed task statistics or settled GPU work');
      }
    }

    // The cap bounds retained idle scratch, rather than one batch's working
    // set. Points remain available to same-context statistics and display.
    function trimScratch(maxBytes = scratchPoolMaxBytes) {
      if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
        throw Error('Scratch trim budget must be a nonnegative integer');
      }
      requireIdle('Scratch trimming');
      let freed = 0;
      const candidates = Array.from(records.values())
        .filter(r => r.category === 'scratch').sort((a, b) => a.used - b.used);
      for (const record of candidates) {
        if (scratchBytes <= maxBytes) break;
        freed += record.buffer.size;
        destroyBuffer(record);
      }
      return freed;
    }

    // The caller must also finish point raster/readback before replacing the
    // optical context and its point cache.
    function activateContext(ctx) {
      if (disposed) throw Error('Task backend is disposed');
      if (activeContext === ctx) return;
      requireIdle('Task context replacement');
      if (activeContext !== null) {
        for (const record of Array.from(records.values())) destroyBuffer(record);
      }
      activeContext = ctx;
    }

    function markStatsReady(ctx, di) {
      if (disposed) throw Error('Task backend is disposed');
      if (activeTrace) throw Error('Task statistics cannot complete during dispatch setup');
      if (ctx !== activeContext) throw Error('Task statistics context does not match the active context');
      if (!pendingRows.has(di)) return;
      pendingRows.delete(di);
      for (const record of records.values()) record.rows.delete(di);
      for (const record of Array.from(retired)) {
        if (!record.rows.size) destroyBuffer(record);
      }
      if (!pendingRows.size) trimScratch();
    }

    // Failed operations have no statistics read as a completion fence. Wait
    // for the queue before reclaiming their buffers.
    async function settle() {
      if (disposed) throw Error('Task backend is disposed');
      if (activeTrace) throw Error('GPU settling requires completed dispatch setup');
      if (pendingRows.size) {
        if (typeof device.queue.onSubmittedWorkDone !== 'function') {
          throw Error('GPU completion fence is unavailable');
        }
        await device.queue.onSubmittedWorkDone();
        pendingRows.clear();
        for (const record of records.values()) record.rows.clear();
      }
      for (const record of Array.from(retired)) destroyBuffer(record);
      trimScratch();
    }


    function group(pipeline, index, bindings = []) {
      return device.createBindGroup({
        layout: pipeline.getBindGroupLayout(index),
        entries: bindings.map(([binding, b, size, offset]) => {
          const resource = {buffer: b};
          if (size !== undefined) resource.size = size;
          if (offset !== undefined) resource.offset = offset;
          return {binding, resource};
        })
      });
    }

    function packBatch(ctx, row, di, selected, side, cap) {
      const s = row.s;
      const taskData = new ArrayBuffer(selected.length * TASK_BYTES);
      const integers = new Uint32Array(taskData), floats = new Float32Array(taskData);
      const offsets = [], surfaces = [], nodes = [];
      const geometryEntries = new Map(), tableEntries = new Map();
      selected.forEach((task, ti) => {
        const {fi, li, field, lam} = task, table = field.tab;
        // Every bilinear cell uses four of these 9 x 24 nodes. A complete
        // table always interpolates successfully, so generation cannot enqueue aiming.
        // Match the truth test used to pack each node's valid flag.
        task.needsAim = !!s.aiming && !(table && table.ok.length === 216 && table.ok.every(Boolean));
        const offsetBase = geometryEntries.has(lam) ? geometryEntries.get(lam) : offsets.length;
        const nodeBase = tableEntries.has(fi) ? tableEntries.get(fi) : nodes.length / 4;
        task.offsetBase = offsetBase;
        if (!geometryEntries.has(lam)) {
          geometryEntries.set(lam, offsetBase);
          s.surfaces.forEach((surface, i) => {
            const scale = 2 ** Math.ceil(Math.log2(Math.max(1, s.sdStop || 0, ctx.ap?.[i] || 0)));
            const aperture = ctx.ap ? (ctx.ap[i] || -1) : (surface.sd === null ? -1 : surface.sd);
            offsets.push(surfaces.length);
            surfaces.push(
              surface.R ? 1 / surface.R : 0, surface.k, i ? s.surfaces[i - 1].T : 0,
              aperture, surface.n(lam), surface.asph.length,
              (surface.aspo || []).length, surface.doe ? surface.doe.b.length : 0, scale
            );
            surface.asph.forEach((a, j) => surfaces.push(a * scale ** (4 + 2 * j)));
            (surface.aspo || []).forEach((a, j) => surfaces.push(a * scale ** (j + 1)));
            (surface.doe ? surface.doe.b : []).forEach((a, j) => surfaces.push(a * scale ** (2 + 2 * j)));
          });
        }
        if (s.aiming) opt.aim(s, field.th, 0, 0, lam);
        const chief = s._aimC;
        let flags = (table ? 1 : 0) | (s.aiming ? 2 : 0) |
          (s.zObj !== null && s.zObj !== undefined ? 4 : 0) | (field.ell ? 8 : 0);
        if (chief?.ok && chief.det) flags |= 16;
        const offset = ti * TASK_WORDS;
        integers.set([cap, s.surfaces.length, 0, 0, ctx.ap ? 1 : 0, 0, 0, 0], offset);
        floats[offset + 2] = s.surfaces.at(-1).T + s.zImg - s.totalTrack;
        floats[offset + 3] = lam;
        integers.set([
          side, (0x5eed + di * 7919 + fi * 104729 + li * 1299709) >>> 0, flags, s.stopIdx
        ], offset + 8);
        const theta = field.th * Math.PI / 180;
        floats.set([Math.sin(theta), Math.cos(theta), s.zEP, s.zObj || 0], offset + 12);
        floats.set([s.zStart, s.span, s.sdStop, s.epd], offset + 16);
        floats.set([s.pupilMag, chief?.ex || 0, chief?.ey || 0, .001 * Math.max(1, s.epd / 2)], offset + 20);
        floats.set([chief?.a || 0, chief?.b || 0, chief?.c || 0, chief?.e || 0], offset + 24);
        floats.set(field.ell ? [field.ell.cx, field.ell.cy, field.ell.ax, field.ell.ay] : [0, 0, 1, 1], offset + 28);
        floats.set(table ? [...table.A, 0] : [0, 0, 0, 0], offset + 32);
        floats.set(table ? [...table.B, 0] : [0, 0, 0, 0], offset + 36);
        integers.set([offsetBase, nodeBase, ti * cap, fi], offset + 40);
        if (!tableEntries.has(fi)) {
          tableEntries.set(fi, nodeBase);
          if (table) {
            for (let i = 0; i < table.ok.length; i++) {
              nodes.push(table.ex[i], table.ey[i], table.ok[i] ? 1 : 0, 0);
            }
          } else nodes.push(0, 0, 0, 0);
        }
      });
      const packedSurfaces = new Float32Array(surfaces);
      if (!packedSurfaces.every(Number.isFinite)) {
        throw Error('Task surface coefficients exceed f32 range');
      }
      return {taskData, offsets, packedSurfaces, nodes};
    }

    async function traceRow(ctx, di) {
      activateContext(ctx);
      if (activeTrace) throw Error('Task dispatch setup must be sequential');
      if (pendingRows.has(di)) throw Error('Task row retracing requires completed task statistics');
      activeTrace = true;
      activeRow = di;
      try {
        const row = ctx.depths[di];
        const side = Math.max(4, Math.round(Math.sqrt(ctx.R * 4 / Math.PI))), cap = side * side;
        const limit = device.limits.maxStorageBufferBindingSize || 134217728;
        const dispatchLimit = device.limits.maxComputeWorkgroupsPerDimension || 65535;
        if (!Number.isSafeInteger(cap) || cap * 32 > limit || Math.ceil(cap / 64) > dispatchLimit) {
          throw Error('Single optical task exceeds GPU storage/dispatch limits');
        }
        const batchSize = Math.max(1, Math.min(
          maxTasks, dispatchLimit, Math.floor(limit / (cap * 32)), Math.floor(traceChunk / 256)
        ));
        const tasks = [];
        for (let fi = 0; fi < row.fields.length; fi++) {
          for (let li = 0; li < ctx.lam.length; li++) {
            tasks.push({fi, li, field: row.fields[fi], lam: ctx.lam[li]});
          }
        }
        for (let first = 0; first < tasks.length; first += batchSize) {
          const selected = tasks.slice(first, first + batchSize), count = selected.length;
          const {taskData, offsets, packedSurfaces, nodes} = packBatch(ctx, row, di, selected, side, cap);
          const queueSlices = Math.ceil(cap / queueChunk), slots = count * queueSlices;
          const hasAiming = selected.some(task => task.needsAim);
          if (slots * 12 > limit || Math.ceil(slots / 64) > dispatchLimit) {
            throw Error('GPU aim schedule exceeds storage/dispatch limits');
          }
          // One scratch slot is reused after each batch's final submission.
          // Queue ordering protects submitted work from later uploads.
          const data = buffer('descriptor-shared', taskData.byteLength, U.STORAGE | U.COPY_DST, taskData);
          const offsetBuffer = buffer('offsets-shared', offsets.length * 4, U.STORAGE | U.COPY_DST, new Uint32Array(offsets));
          const surfaceBuffer = buffer('surfaces-shared', packedSurfaces.byteLength, U.STORAGE | U.COPY_DST, packedSurfaces);
          const nodeBuffer = buffer('nodes-shared', nodes.length * 4, U.STORAGE | U.COPY_DST, new Float32Array(nodes));
          const cache = buffer('cache-shared', count * cap * 32, U.STORAGE | U.COPY_DST);
          const jobs = buffer('jobs-shared', (count + count * cap) * 4, U.STORAGE | U.COPY_DST);
          const indirect = hasAiming ? buffer('uniform-slice-indirect-shared', slots * 12, U.STORAGE | U.INDIRECT) : null;
          const dummyConfig = buffer('dummy-cfg', 32, U.UNIFORM | U.COPY_DST, new Uint32Array(8));
          const dummyGeneration = buffer('dummy-gen', 128, U.UNIFORM | U.COPY_DST, new Uint32Array(32));
          const output = buffer('results-' + di + '-' + first, count * cap * 16, U.STORAGE);
          let encoder = device.createCommandEncoder({label: 'Spot multi-task dispatch'});
          encoder.clearBuffer(cache, 0, count * cap * 32);
          encoder.clearBuffer(jobs, 0, count * 4);
          function flush() {
            device.queue.submit([encoder.finish()]);
            encoder = device.createCommandEncoder({label: 'Spot bounded optical work'});
          }
          const chunk = Math.max(256, Math.floor(traceChunk / count / 256) * 256);
          let pipeline, pass;
          function traceInterval(mode, begin, length) {
            const interval = taskData.slice(0), integers = new Uint32Array(interval);
            for (let ti = 0; ti < count; ti++) integers[ti * TASK_WORDS + 5] = begin;
            const descriptor = buffer('trace-descriptor-shared', interval.byteLength,
              U.STORAGE | U.COPY_DST, interval);
            const control = buffer('trace-control-shared', 16, U.UNIFORM | U.COPY_DST,
              new Uint32Array([mode, length, 0, 0]));
            pipeline = pipelines.generateTraceOpticalTasks;
            pass = encoder.beginComputePass({label: mode ? 'Spot queued final trace' : 'Spot sample and table trace'});
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group(pipeline, 0, [
              [0, offsetBuffer], [1, surfaceBuffer], [3, output], [4, dummyConfig],
              [5, dummyGeneration], [6, nodeBuffer], [7, cache]
            ]));
            pass.setBindGroup(1, group(pipeline, 1, [[4, descriptor, count * TASK_BYTES], [5, jobs], [11, control]]));
            pass.dispatchWorkgroups(Math.ceil(length / 64), count);
            pass.end();
            // Submit before overwriting shared descriptor/control uniforms.
            flush();
          }
          for (let begin = 0; begin < cap; begin += chunk) traceInterval(0, begin, Math.min(chunk, cap - begin));
          if (hasAiming) {
            const schedule = buffer('queue-schedule-shared', 16, U.UNIFORM | U.COPY_DST,
              new Uint32Array([count, queueSlices, queueChunk, 0]));
            pipeline = pipelines.prepareUniformTaskSliceDispatch;
            pass = encoder.beginComputePass({label: 'Spot GPU aim slice schedule'});
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group(pipeline, 0));
            pass.setBindGroup(1, group(pipeline, 1, [
              [4, data, count * TASK_BYTES], [5, jobs], [8, indirect], [10, schedule]
            ]));
            pass.dispatchWorkgroups(Math.ceil(slots / 64));
            pass.end();

            // Every slice binds an immutable aligned 16-byte control range. The
            // indirect buffer is never also bound as aiming storage.
            const controlStride = Math.max(16, device.limits.minUniformBufferOffsetAlignment || 256);
            const controls = new Uint32Array(slots * controlStride / 4);
            for (let ti = 0; ti < count; ti++) {
              for (let slice = 0; slice < queueSlices; slice++) {
                const begin = slice * queueChunk;
                controls.set([ti, count, begin, Math.min(queueChunk, cap - begin)],
                  (ti * queueSlices + slice) * controlStride / 4);
              }
            }
            const control = buffer('queue-controls-shared', controls.byteLength, U.UNIFORM | U.COPY_DST, controls);
            pipeline = pipelines.aimOpticalUniformQueue;
            pass = encoder.beginComputePass({label: 'Spot GPU scheduled aiming'});
            pass.setPipeline(pipeline);
            selected.forEach((task, ti) => {
              if (!task.needsAim) return;
              const cfgData = new ArrayBuffer(32);
              new Uint8Array(cfgData).set(new Uint8Array(taskData, ti * TASK_BYTES, 24));
              new Uint32Array(cfgData)[5] = ti * cap;
              const config = buffer('aim-cfg-shared-' + ti, 32, U.UNIFORM | U.COPY_DST, cfgData);
              const offsetsForTask = offsets.slice(task.offsetBase, task.offsetBase + row.s.surfaces.length);
              const aimOffsets = buffer('aim-offsets-shared-' + ti, offsetsForTask.length * 4,
                U.STORAGE | U.COPY_DST, new Uint32Array(offsetsForTask));
              const generation = buffer('aim-gen-shared-' + ti, 128, U.UNIFORM | U.COPY_DST,
                new Uint8Array(taskData, ti * TASK_BYTES + 32, 128));
              pass.setBindGroup(0, group(pipeline, 0, [
                [0, aimOffsets], [1, surfaceBuffer], [4, config],
                [5, generation], [7, cache]
              ]));
              for (let slice = 0; slice < queueSlices; slice++) {
                const slot = ti * queueSlices + slice;
                pass.setBindGroup(1, group(pipeline, 1, [[5, jobs], [9, control, 16, slot * controlStride]]));
                pass.dispatchWorkgroupsIndirect(indirect, slot * 12);
              }
            });
            pass.end();
          }
          flush();

          if (hasAiming) {
            for (let begin = 0; begin < cap; begin += chunk) traceInterval(1, begin, Math.min(chunk, cap - begin));
          }
          selected.forEach((task, ti) => {
            const cfg = new Uint32Array(8);
            cfg[0] = cap;
            const config = buffer('reduce-cfg-shared-' + ti, 32, U.UNIFORM | U.COPY_DST, cfg);
            post.reduce(ctx, di, task.fi, task.li, encoder,
              {buffer: output, offset: ti * cap}, config, {first: 0, count: cap});
          });
          flush();
        }
      } finally {
        activeRow = null;
        activeTrace = false;
      }
    }

    // Abandon is terminal cleanup after the owner has destroyed its device.
    function dispose({abandon = false} = {}) {
      if (disposed) return;
      if (abandon) {
        pendingRows.clear();
        for (const record of records.values()) record.rows.clear();
      } else requireIdle('Task disposal');
      for (const record of Array.from(records.values())) destroyBuffer(record);
      buffers.clear();
      activeContext = null;
      disposed = true;
    }
    return {traceRow, markStatsReady, settle, dispose};
  }
  return {create};
})();
