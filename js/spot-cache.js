/* Preparation caches survive runs when their Worker/backend survives.
   Display, RGB colors, weights and ray count do not invalidate optical prep. */
var SPOTCACHE=(function(){
 function keyJSON(value){return JSON.stringify(value,function(k,v){return typeof v==='number'&&!Number.isFinite(v)?{nonfinite:String(v)}:v})}
 function opticalKey(sys,o){return keyJSON({system:[sys.efl,sys.epd],surfaces:sys.surfaces.map(s=>[s.R,s.T,s.k,s.sd,s.matLabel,s.asph,s.aspo,s.doe,o.lambdas.map(w=>s.n(w.nm/1000))]),opt:[o.lambdas.map(w=>w.nm),o.primary,o.stopIdx,o.apertureMode,o.fno,o.epd,o.objDist,o.defocus,o.rayAiming,o.sdAp],vig:sys.vig})}
 function cache(limit){var map=new Map();return {get(k){if(!map.has(k))return;var v=map.get(k);map.delete(k);map.set(k,v);return v},set(k,v){map.delete(k);map.set(k,v);while(map.size>limit)map.delete(map.keys().next().value)},clear(){map.clear()}}}
 function clone(ctx,o){return Object.assign({},ctx,{R:o.spotRays||50000,nRays:0,depths:ctx.depths.map(row=>Object.assign({},row,{fields:row.fields.map(f=>Object.assign({},f,{grids:[]}))}))})}
 function create(opt,preparation){var depths=cache(8),tables=cache(8);return {
  async defocusDepths(s,o,target){var key=opticalKey(s,o)+'|'+target;var hit=depths.get(key);if(hit){return Object.assign({},hit)}var v=await preparation.defocusDepths(s,o,target);depths.set(key,Object.assign({},v));return v},
  async spotPrepare(s,o){var key=opticalKey(s,o)+'|'+keyJSON([o.spotDepths,o.spotFields,o.spotLambdas]);var hit=tables.get(key);if(hit){return clone(hit,o)}var v=opt.spotPrepare(s,o);tables.set(key,v);return clone(v,o)},
  clear(){depths.clear();tables.clear()}
 }}
 return {create};
})();
