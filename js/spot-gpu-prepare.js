/* GPU axial defocus search. Only the final 48-byte depth labels cross to the
   CPU; OPT.spotPrepare builds the tables and chiefs used by Spot tracing. */
var SPOTPREP = (function () {
  'use strict';
  function buildNear(OPT, surfaces, opt) {
    var vertices=[], z=0;
    surfaces.forEach(function(s){vertices.push(z);z+=s.T;});
    var lambda=opt.lambdas[opt.primary||0].nm/1000, fo=OPT.firstOrder(surfaces,lambda,opt.stopIdx);
    var D=opt.objDist, finite=Number.isFinite(D)&&D>0&&D<1e7, zObj=finite?-D:null;
    var uEnd=finite?OPT.paraxFromObject(surfaces,lambda,D):0, epd;
    var stop=surfaces[opt.stopIdx], stopRadius=stop&&stop.sd;
    if(opt.apertureMode==='epd')epd=opt.epd;
    else if(opt.apertureMode==='stop'&&stopRadius&&Math.abs(fo.pupilMag)>1e-9)epd=2*stopRadius/Math.abs(fo.pupilMag);
    else if(opt.apertureMode==='fnoinf')epd=Math.abs(fo.efl)/opt.fno;
    else if(finite)epd=Math.abs(uEnd)>1e-12?2*(fo.zEP-zObj)/(2*opt.fno)/Math.abs(uEnd):Math.abs(fo.efl)/opt.fno;
    else epd=Math.abs(fo.efl)/opt.fno;
    var fno=Math.abs(fo.efl)/epd;
    if(finite&&Math.abs(uEnd)>1e-12)fno=1/(2*Math.abs((epd/2)/(fo.zEP-zObj)*uEnd));
    var sd=stopRadius||epd/2*(Math.abs(fo.pupilMag)>1e-9?Math.abs(fo.pupilMag):1);
    return {surfaces:surfaces,zVertex:vertices,zImg:z+(opt.defocus||0),totalTrack:z,stopIdx:opt.stopIdx,
      aiming:!!opt.rayAiming&&sd>0,sdStop:sd,efl:fo.efl,bfl:fo.bfl,zEP:fo.zEP,zXP:fo.zXP,
      epd:epd,fno:fno,pupilMag:fo.pupilMag,zObj:zObj,mag:finite&&Math.abs(uEnd)>1e-12?1/uEnd:0,
      objDist:finite?D:Infinity,vig:opt.vig||null,zStart:Math.min(0,fo.zEP)-Math.max(20,Math.abs(fo.efl)*.5),
      span:Math.max(Math.abs(z),2*surfaces.reduce(function(m,s){return Math.max(m,s.sd||0);},0)),
      _gpuStopKnown:!!stopRadius};
  }
  async function create(OPT, runtime) {
    runtime=runtime||{};
    var device=runtime.device,shader=runtime.shader;
    if(!device||!shader)throw new Error('GPU depth search requires the tracing device and compiled shader');
    var usage=GPUBufferUsage,disposed=false,active=new Set();
    var names=[
      'prepareStop','prepareChief','prepareChiefSeeds','prepareChiefFinish',
      'prepareAimNodes','prepareExpandAxis','prepareFitTable',
      'prepareAxialSamples','prepareBlurReduceParallel',
      'depthControlStage'
    ];
    var pipelines={};
    await Promise.all(names.map(async function(name){
      pipelines[name]=await device.createComputePipelineAsync({
        label:'spot-depth-'+name,layout:'auto',compute:{module:shader,entryPoint:name}
      });
    }));
    var layout={
      prepareStop:[[0,1,4,5],[0,1]],prepareChief:[[0,1,4,5],[1]],
      prepareChiefSeeds:[[0,1,4,5],[1,3]],prepareChiefFinish:[[0,1,4,5],[1,3]],
      prepareAimNodes:[[0,1,4,5],[1,3]],prepareExpandAxis:[[],[2,3]],
      prepareFitTable:[[],[1,2]],prepareAxialSamples:[[0,1,4,5,6],[0,1,3]],
      prepareBlurReduceParallel:[[],[0,1,3]]
    };
    function makeBuffer(owned,label,size,flags,data){
      var b=device.createBuffer({label:label,size:Math.max(4,Math.ceil(size/4)*4),usage:flags});
      owned.push(b);
      if(data)device.queue.writeBuffer(b,0,data);
      return b;
    }
    function release(owned){
      if(owned.released)return;owned.released=true;
      owned.forEach(function(b){b.destroy();});
      active.delete(owned);
    }
    function pack(sys,opt,lam,ap){
      var offsets=new Uint32Array(sys.surfaces.length),records=[];
      sys.surfaces.forEach(function(s,i){
        var radius=Math.max(1,sys.sdStop||0,opt.sdDraw&&opt.sdDraw[i]||0,ap&&ap[i]||0);
        var scale=Math.pow(2,Math.ceil(Math.log2(radius))),clear=ap?(ap[i]||-1):(s.sd===null?-1:s.sd);
        // Near a paraxial conjugate, a work-F/# candidate can have a huge
        // entrance pupil even though only the lens's clear apertures pass.
        // The normalization is a change of units, not an optical aperture.
        // Lower it by exact powers of two until its coefficients fit f32;
        // retain every candidate and let the ordinary tracer classify rays.
        for(var attempt=0;attempt<128;attempt++){
          var valid=Number.isFinite(scale)&&scale>0;
          s.asph.forEach(function(a,j){valid=valid&&Number.isFinite(Math.fround(a*Math.pow(scale,4+2*j)));});
          (s.aspo||[]).forEach(function(a,j){valid=valid&&Number.isFinite(Math.fround(a*Math.pow(scale,j+1)));});
          (s.doe?s.doe.b:[]).forEach(function(a,j){valid=valid&&Number.isFinite(Math.fround(a*Math.pow(scale,2+2*j)));});
          if(valid)break;scale*=.5;
        }
        offsets[i]=records.length;
        records.push(s.R?1/s.R:0,s.k,i?sys.surfaces[i-1].T:0,clear,s.n(lam),s.asph.length,(s.aspo||[]).length,s.doe?s.doe.b.length:0,scale);
        s.asph.forEach(function(a,j){records.push(a*Math.pow(scale,4+2*j));});
        (s.aspo||[]).forEach(function(a,j){records.push(a*Math.pow(scale,j+1));});
        (s.doe?s.doe.b:[]).forEach(function(a,j){records.push(a*Math.pow(scale,2+2*j));});
      });
      var data=new Float32Array(records);if(!data.every(Number.isFinite))throw new Error('GPU preparation surface data exceeds f32');
      return {offsets:offsets,data:data};
    }
    function makeJob(owned,sys,opt,ap,samples){
      var job={s:sys,opt:opt,refs:new Map(),controls:new Map(),ap:ap,samples:samples,owned:owned};
      job.meta=makeBuffer(owned,'depth-meta',128,usage.STORAGE);
      job.nodes=makeBuffer(owned,'depth-nodes',216*16,usage.STORAGE);
      job.scratch=makeBuffer(owned,'depth-scratch',Math.max(2048,640+3*samples)*16,usage.STORAGE);
      var data=new ArrayBuffer(128),u=new Uint32Array(data),f=new Float32Array(data);
      u.set([0,0,(sys.aiming?2:0)|(sys.zObj!==null?4:0),sys.stopIdx]);
      f.set([0,1,sys.zEP,sys.zObj||0],4);
      f.set([sys.zStart,sys.span,sys.sdStop,sys.epd],8);
      f.set([sys.pupilMag,0,0,1e-3*Math.max(1,sys.epd/2)],12);f.set([0,0,1,1],20);
      job.gen=makeBuffer(owned,'depth-generation',128,usage.UNIFORM|usage.COPY_DST,data);
      return job;
    }
    function reference(job,lam){
      if(job.refs.has(lam))return job.refs.get(lam);
      var packed=pack(job.s,job.opt,lam,job.ap),owned=job.owned;
      if(!job.offsets)job.offsets=makeBuffer(owned,'depth-offsets',packed.offsets.byteLength,usage.STORAGE|usage.COPY_DST,packed.offsets);
      var surface=makeBuffer(owned,'depth-surfaces',packed.data.byteLength,usage.STORAGE|usage.COPY_DST,packed.data);
      var params=new ArrayBuffer(32),u=new Uint32Array(params),f=new Float32Array(params);
      u[0]=400;u[1]=job.s.surfaces.length;
      f[2]=job.s.surfaces[job.s.surfaces.length-1].T+(job.s.zImg-job.s.totalTrack);
      f[3]=lam;u[4]=job.ap?1:0;
      var cfg=makeBuffer(owned,'depth-params',32,usage.UNIFORM|usage.COPY_DST,params);
      var ref={surface:surface,cfg:cfg};job.refs.set(lam,ref);return ref;
    }
    function control(job,rgb){
      var key=rgb;if(job.controls.has(key))return job.controls.get(key);
      var data=new Uint32Array([0,job.samples,job.s._gpuStopKnown?1:0,rgb]);
      var b=makeBuffer(job.owned,'depth-probe-control',16,usage.UNIFORM|usage.COPY_DST,data);
      job.controls.set(key,b);return b;
    }
    function op(list,job,name,lam,rgb){
      var ref=reference(job,lam),ctl=control(job,rgb||0);
      var resources=[
        {0:job.offsets,1:ref.surface,4:ref.cfg,5:job.gen,6:job.nodes},
        {0:ctl,1:job.meta,2:job.nodes,3:job.scratch}
      ];
      var pipeline=pipelines[name];
      var groups=layout[name].map(function(bindings,index){
        return device.createBindGroup({layout:pipeline.getBindGroupLayout(index),entries:bindings.map(function(binding){
          return{binding:binding,resource:{buffer:resources[index][binding]}};
        })});
      });
      var count=name==='prepareChiefSeeds'?10:name==='prepareExpandAxis'?4:name==='prepareAxialSamples'?Math.ceil(job.samples/64):1;
      var indirectSlot=name==='prepareChiefSeeds'?1:name==='prepareAimNodes'?2:name==='prepareExpandAxis'?3:name==='prepareAxialSamples'?4:0;
      list.push({pipeline:pipeline,groups:groups,count:count,indirect:job.depthIndirect,indirectOffset:indirectSlot*16});
    }
    function chief(list,job,lam){
      ['prepareChief','prepareChiefSeeds','prepareChiefFinish'].forEach(function(name){op(list,job,name,lam,0);});
    }
    function table(list,job,lam){
      // These axial tables serve the blur probes only. Final Spot tables are
      // prepared on the CPU after the depth labels have been read.
      op(list,job,'prepareAimNodes',lam,0);
      op(list,job,'prepareExpandAxis',lam,0);
      op(list,job,'prepareFitTable',lam,0);
    }
    async function execute(list){
      if(disposed)throw new Error('GPU depth search is disposed');
      device.pushErrorScope('validation');
      try{
        var encoder=device.createCommandEncoder({label:'GPU axial defocus search'});
        list.forEach(function(command){
          // Each dependent dispatch has its own compute pass.
          if(command.copy){encoder.copyBufferToBuffer.apply(encoder,command.copy);return;}
          var pass=encoder.beginComputePass({label:'Axial depth compute'});
          pass.setPipeline(command.pipeline);
          command.groups.forEach(function(group,index){pass.setBindGroup(index,group);});
          if(command.indirect)pass.dispatchWorkgroupsIndirect(command.indirect,command.indirectOffset||0);
          else pass.dispatchWorkgroups(command.count);
          pass.end();
        });
        device.queue.submit([encoder.finish()]);
      }finally{
        var error=await device.popErrorScope();
        if(error)throw new Error('GPU depth search validation: '+error.message);
      }
    }
    function depthControl(depth,record,validation,stage){
      var key=record+'|'+validation+'|'+stage;if(depth.controls.has(key))return depth.controls.get(key);
      var data=new Uint32Array([record,depth.backStart,depth.backCount,depth.frontStart,depth.frontCount,stage,validation?1:0,0]);
      var b=makeBuffer(depth.owned,'depth-search-control',32,usage.UNIFORM|usage.COPY_DST,data);
      depth.controls.set(key,b);return b;
    }
    function depthOp(list,depth,name,job,record,validation){
      var stages=['prepareDepthSearchGeometry','collectDepthBlur','depthCoarseSchedule',
        'depthSearchBracket','depthSearchNext','depthSearchUpdate','depthSearchFinish'];
      var stage=stages.indexOf(name);if(stage<0)throw new Error('Unknown depth control stage');
      if(!depth.fallback){
        depth.fallback={
          gen:makeBuffer(depth.owned,'depth-control-unused-gen',128,usage.UNIFORM),
          meta:makeBuffer(depth.owned,'depth-control-unused-meta',128,usage.STORAGE),
          geometry:makeBuffer(depth.owned,'depth-control-unused-geometry',128,usage.STORAGE),
          dispatch:makeBuffer(depth.owned,'depth-control-unused-dispatch',80,usage.STORAGE)
        };
      }
      var pipeline=pipelines.depthControlStage,ctl=depthControl(depth,record||0,validation,stage);
      var resources={5:job&&job.gen||depth.fallback.gen,1:job&&job.meta||depth.fallback.meta,
        15:depth.buffer,16:ctl,17:depth.constants,18:job&&job.depthGen||depth.fallback.geometry,
        // A buffer cannot be both writable storage and indirect dispatch input
        // in one pass. Stage 1 never accesses dispatch storage, so bind a dummy.
        19:name!=='collectDepthBlur'&&job&&job.depthIndirect||depth.fallback.dispatch};
      var bindingsByGroup=[[5],[1],[15,16,17,18,19]];
      var groups=bindingsByGroup.map(function(bindings,index){
        return device.createBindGroup({layout:pipeline.getBindGroupLayout(index),entries:bindings.map(function(binding){
          return{binding:binding,resource:{buffer:resources[binding]}};
        })});
      });
      list.push({pipeline:pipeline,groups:groups,count:1,indirect:name==='collectDepthBlur'&&job.depthIndirect,indirectOffset:0});
      if(name==='prepareDepthSearchGeometry')list.push({copy:[job.depthGen,0,job.gen,0,128]});
    }
    function depthConstants(owned,sys,opt){
      var lam=opt.lambdas[opt.primary||0].nm/1000;
      var inf=buildNear(OPT,sys.surfaces,Object.assign({},opt,{objDist:Infinity}));
      var stop=sys.surfaces[opt.stopIdx],mode=0;
      if(opt.apertureMode==='epd')mode=1;
      else if(opt.apertureMode==='stop'&&stop&&stop.sd&&Math.abs(sys.pupilMag)>1e-9)mode=2;
      else if(opt.apertureMode==='fnoinf')mode=3;
      // The paraxial object ray is linear in D: first-order EFL supplies its
      // slope, and the D=0 ray supplies its constant term.
      var data=new Float32Array([Math.abs(sys.efl),sys.zEP,sys.pupilMag,-1/sys.efl,
        OPT.paraxFromObject(sys.surfaces,lam,0),opt.fno,inf.epd,stop&&stop.sd||0,
        sys.zStart,sys.span,mode,opt.rayAiming?1:0]);
      if(!data.every(Number.isFinite))throw new Error('Depth geometry exceeds f32');
      return makeBuffer(owned,'depth-geometry',48,usage.UNIFORM|usage.COPY_DST,data);
    }
    function axialJob(owned,sys,opt,depth,record,samples,dynamic){
      var s=buildNear(OPT,sys.surfaces,Object.assign({},opt,{objDist:dynamic?Infinity:record}));s.vig=sys.vig;
      var ap=OPT.hardApertures(s,opt),job=makeJob(owned,s,opt,ap,samples);
      if(dynamic){
        job.depthGen=makeBuffer(owned,'depth-search-generation',128,usage.STORAGE|usage.COPY_SRC);
        var counts=[1,10,1,4,Math.ceil(samples/64)],dispatch=new Uint32Array(20);
        counts.forEach(function(count,i){dispatch.set([count,1,1,count],i*4);});
        job.depthIndirect=makeBuffer(owned,'depth-search-dispatch',80,usage.STORAGE|usage.INDIRECT|usage.COPY_DST,dispatch);
      }
      job.axial=function(list,target,validation){
        if(dynamic)depthOp(list,depth,'prepareDepthSearchGeometry',job,target,validation);
        var lam0=opt.lambdas[opt.primary||0].nm/1000;
        op(list,job,'prepareStop',lam0,0);
        chief(list,job,.545);if(s.aiming)table(list,job,.545);
        [.620,.545,.465].forEach(function(lam,li){
          chief(list,job,lam);op(list,job,'prepareAxialSamples',lam,li);
        });
        op(list,job,'prepareBlurReduceParallel',.465,0);
        depthOp(list,depth,'collectDepthBlur',job,target,validation);
      };
      return job;
    }
    async function readLabels(depth,Df,targetUm,budget,sides){
      var read=device.createBuffer({label:'Final depth labels',size:48,usage:usage.MAP_READ|usage.COPY_DST});
      var mapped=false;
      try{
        var encoder=device.createCommandEncoder({label:'Read final depth labels'});
        encoder.copyBufferToBuffer(depth.buffer,16,read,0,48);device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);mapped=true;
        var values=new Float32Array(read.getMappedRange()).slice();
        var result={focus:Df,focusUm:values[1],targetUm:targetUm,search:'secant',validationSamples:budget};
        sides.forEach(function(side){
          var name=side===0?'back':'front',k=4+side*4;
          result[name]=values[k]<=0?Infinity:1/values[k];
          result[name+'Um']=values[k+1];result[name+'Reached']=values[k+2]!==0;
        });
        return result;
      }finally{if(mapped)read.unmap();read.destroy();}
    }
    async function defocusDepths(sys,opt,targetUm){
      if(disposed)throw new Error('GPU depth search is disposed');
      if(!Number.isFinite(targetUm)||targetUm<=0)throw new Error('Depth target must be positive and finite');
      var budget=runtime.validationSamples||1600;
      if(!Number.isInteger(budget)||budget<400||budget>16384)throw new Error('validationSamples must be an integer in [400, 16384]');
      var Df=sys.objDist,efl=Math.abs(sys.efl),fin=Number.isFinite(Df)&&Df>0,uf=fin?1/Df:0;
      var du0=(targetUm/1000)/Math.max(efl*sys.epd,1e-9);
      var Dmin=fin?Math.max(1,.03*Math.min(Df,1e5)):Math.max(1,3*efl),uMax=1/Dmin;
      var ks=Array.from({length:25},function(_,i){return Math.pow(2,-3+6*i/24);}),all=[uf],grid={};
      (fin?['back','front']:['front']).forEach(function(side){
        var sg=side==='back'?-1:1,dus=[];
        ks.forEach(function(k){if(sg<0?uf-k*du0>0:uf+k*du0<uMax)dus.push(k*du0);});
        dus.push(sg<0?uf:uMax-uf);dus.sort(function(a,b){return a-b;});
        dus=dus.filter(function(v,i){return!i||v!==dus[i-1];});
        grid[side]={start:all.length,count:dus.length};
        dus.forEach(function(d){all.push(uf+sg*d);});
      });
      var owned=[],list=[],labelsRead=false;active.add(owned);
      try{
        var data=new Float32Array((16+all.length)*4);data.set([uf,targetUm,uMax,fin?1:0]);
        all.forEach(function(u,i){data[(16+i)*4]=u;});
        var depth={owned:owned,
          buffer:makeBuffer(owned,'depth-search-state',data.byteLength,usage.STORAGE|usage.COPY_SRC|usage.COPY_DST,data),
          constants:depthConstants(owned,sys,opt),controls:new Map(),
          backStart:grid.back?grid.back.start:0,backCount:grid.back?grid.back.count:0,
          frontStart:grid.front.start,frontCount:grid.front.count
        };
        var coarseDispatchData=new Uint32Array(20);
        [1,10,1,4,Math.ceil(400/64)].forEach(function(n,i){coarseDispatchData.set([n,1,1,n],i*4);});
        var coarseDispatch={};
        Object.keys(grid).forEach(function(side){
          coarseDispatch[side]=makeBuffer(owned,'depth-coarse-dispatch-'+side,80,usage.STORAGE|usage.INDIRECT|usage.COPY_DST,coarseDispatchData);
        });
        all.forEach(function(u,i){
          var job=axialJob(owned,sys,opt,depth,u<=0?Infinity:1/u,400,false);
          var side=i>=grid.front.start?'front':'back',sideGrid=grid[side];
          if(i>0&&i>sideGrid.start)job.depthIndirect=coarseDispatch[side];
          job.axial(list,16+i,false);
          if(i>0&&i===sideGrid.start)depthOp(list,depth,'depthCoarseSchedule',{depthIndirect:coarseDispatch[side]},16+i,false);
        });
        depthOp(list,depth,'depthSearchBracket',null,0,false);
        var sides=fin?[0,1]:[1];
        var trials=sides.map(function(side){return{side:side,job:axialJob(owned,sys,opt,depth,0,400,true)};});
        for(var round=0;round<7;round++){
          depthOp(list,depth,'depthSearchNext',null,0,false);
          trials.forEach(function(t){t.job.axial(list,9+t.side,false);});
          depthOp(list,depth,'depthSearchUpdate',null,0,false);
        }
        depthOp(list,depth,'depthSearchFinish',null,0,false);
        sides.forEach(function(side){
          var job=axialJob(owned,sys,opt,depth,0,budget,true);job.axial(list,2+side,true);
        });
        await execute(list);
        var labels=await readLabels(depth,Df,targetUm,budget,sides);labelsRead=true;return labels;
      }finally{
        // A successful map completes every preceding search command. On error,
        // wait for submitted work before retiring its temporary allocations.
        if(!labelsRead&&device.queue.onSubmittedWorkDone){try{await device.queue.onSubmittedWorkDone();}catch(error){}}
        release(owned);
      }
    }
    return {
      buildSystem:function(s,o){return buildNear(OPT,s,o);},defocusDepths:defocusDepths,
      dispose:function(){if(disposed)return;disposed=true;active.forEach(release);}
    };
  }
  return {create:create};
})();
