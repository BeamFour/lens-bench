// GPU axial defocus probes. These temporary aiming tables are used only by
// the depth search; final Spot tables and chiefs are prepared on the CPU.
// control = reserved, radial sample count, flags(knownStop=1), RGB slot.
const PREP_GROUP_SIZE:u32=64u;
struct PrepParams { control:vec4u }
@group(1) @binding(0) var<uniform> prep:PrepParams;
@group(1) @binding(1) var<storage,read_write> prepMeta:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> prepNodes:array<vec4f>;
@group(1) @binding(3) var<storage,read_write> prepScratch:array<vec4f>;
fn beginPreparation(reference:bool) {
  generationOverride=gen; generationOverrideEnabled=true;
  generationOverride.geometry.z=prepMeta[0].z;
  generationOverride.grid.z&=~17u;
  if(reference) {
    generationOverride.aim.y=prepMeta[0].x; generationOverride.aim.z=prepMeta[0].y;
    generationOverride.jac=prepMeta[1];
    let j=prepMeta[1]; let det=differenceOfProducts(j.x,j.w,j.y,j.z);
    if(prepMeta[0].w!=0.0&&abs(det)>1e-14) { generationOverride.grid.z|=16u; }
  }
}
fn usePreparedTable() {
  generationOverride.affineX=prepMeta[2]; generationOverride.affineY=prepMeta[3];
  if(prepMeta[2].w!=0.0) { generationOverride.grid.z|=1u; }
}
fn prepareNodeUV(index:u32)->vec2f {
  let ring=index/24u; let angle=index%24u;
  let phi=6.283185307179586*f32(angle)/24.0;
  return f32(ring)/8.0*vec2f(cos(phi),sin(phi));
}
fn writeChief(q:vec2f,valid:bool) {
  prepMeta[0]=vec4f(q,prepMeta[0].z,select(0.0,1.0,valid));
  var jac=vec4f(0.0);
  if(valid) {
    let h=stopHit(q); let d=max(1e-4,generation().aim.w);
    var dx=d; var dy=d;
    var hx=stopHit(q+vec2f(dx,0.0)); var hy=stopHit(q+vec2f(0.0,dy));
    if(!hx.ok) { dx=-d; hx=stopHit(q+vec2f(dx,0.0)); }
    if(!hy.ok) { dy=-d; hy=stopHit(q+vec2f(0.0,dy)); }
    if(h.ok&&hx.ok&&hy.ok) {
      let x=(hx.q-h.q)/dx; let y=(hy.q-h.q)/dy;
      jac=vec4f(x.x,y.x,x.y,y.y);
    }
  }
  prepMeta[1]=jac;
}
@compute @workgroup_size(1)
fn prepareStop() {
  generationOverride=gen; generationOverrideEnabled=true;
  generationOverride.field.x=0.0; generationOverride.field.y=1.0;
  var radius=gen.geometry.z;
  if((prep.control.z&1u)==0u) {
    let r=gen.geometry.w*0.5*0.9995;
    let a=stopHit(vec2f(0.0,r)); let b=stopHit(vec2f(0.0,-r));
    var actual=0.0;
    if(a.ok) { actual=max(actual,abs(a.q.y)); }
    if(b.ok) { actual=max(actual,abs(b.q.y)); }
    if(actual>1e-9) { radius=actual; }
  }
  prepMeta[0]=vec4f(0.0,0.0,radius,0.0);
}
@compute @workgroup_size(1)
fn prepareChief() {
  beginPreparation(false);
  prepMeta[0]=vec4f(0.0,0.0,prepMeta[0].z,0.0);
  if((generation().grid.z&2u)==0u) { return; }
  let answer=aimNewton(vec2f(0.0),vec2f(0.0),false);
  prepMeta[0]=vec4f(answer.q,prepMeta[0].z,select(0.0,1.0,answer.ok));
}
@compute @workgroup_size(PREP_GROUP_SIZE)
fn prepareChiefSeeds(@builtin(global_invocation_id) gid:vec3u) {
  let i=gid.x; if(i>600u||prepMeta[0].w!=0.0||(gen.grid.z&2u)==0u) { return; }
  beginPreparation(false);
  let span=3.0*max(gen.geometry.y,gen.geometry.w);
  let y=-span+2.0*span*f32(i)/600.0; let hit=stopHit(vec2f(0.0,y));
  prepScratch[i]=vec4f(y,hit.q.y,select(0.0,1.0,hit.ok),0.0);
}
@compute @workgroup_size(1)
fn prepareChiefFinish() {
  beginPreparation(false);
  if((gen.grid.z&2u)==0u) { writeChief(vec2f(0.0),false); return; }
  var answer=AimResult(prepMeta[0].xy,prepMeta[0].w!=0.0);
  if(!answer.ok) {
    var best=0.0; var error=3.4028234e38; var crossing=0.0; var hasCrossing=false;
    var distance=3.4028234e38;
    var previous=vec4f(0.0);
    for(var i=0u;i<=600u;i++) {
      let point=prepScratch[i];
      if(point.z!=0.0) {
        if(abs(point.y)<error) { error=abs(point.y); best=point.x; }
        if(previous.z!=0.0&&opposite(previous.y,point.y)) {
          var y=0.5*(previous.x+point.x);
          if(previous.y!=point.y) { y=previous.x+(point.x-previous.x)*previous.y/(previous.y-point.y); }
          let d=f32(i);
          if(d<distance) { crossing=y; distance=d; hasCrossing=true; }
        }
      }
      previous=point;
    }
    if(hasCrossing) { answer=aimNewton(vec2f(0.0,crossing),vec2f(0.0),false); }
    if(!answer.ok&&error<3.4028234e38) { answer=aimNewton(vec2f(0.0,best),vec2f(0.0),false); }
    if(!answer.ok) { answer=aimNewton(vec2f(0.0),vec2f(0.0),false); }
  }
  writeChief(answer.q,answer.ok);
}
@compute @workgroup_size(PREP_GROUP_SIZE)
fn prepareAimNodes(@builtin(global_invocation_id) gid:vec3u) {
  let i=gid.x; if(i>=9u) { return; }
  beginPreparation(true);
  var answer=AimResult(prepMeta[0].xy,prepMeta[0].w!=0.0);
  let uv=vec2f(f32(i)/8.0,0.0);
  if(i>0u||!answer.ok) { answer=aimGenerated(uv); }
  prepScratch[608u+i]=vec4f(answer.q,select(0.0,1.0,answer.ok),0.0);
}
@compute @workgroup_size(PREP_GROUP_SIZE)
fn prepareExpandAxis(@builtin(global_invocation_id) gid:vec3u) {
  let i=gid.x; if(i>=216u) { return; }
  let node=prepScratch[608u+i/24u]; let angle=6.283185307179586*f32(i%24u)/24.0;
  let sn=sin(angle); let cs=cos(angle);
  prepNodes[i]=vec4f(fma(-node.y,sn,node.x*cs),fma(node.x,sn,node.y*cs),node.z,0.0);
}
@compute @workgroup_size(1)
fn prepareFitTable() {
  var m=0.0; var u=0.0; var v=0.0; var uu=0.0; var uv=0.0; var vv=0.0;
  var tx=vec3f(0.0); var ty=vec3f(0.0);
  for(var i=0u;i<216u;i++) {
    if(prepNodes[i].z==0.0||(i>0u&&i<24u)) { continue; }
    let q=prepareNodeUV(i); let basis=vec3f(1.0,q);
    m+=1.0; u+=q.x; v+=q.y; uu+=q.x*q.x; uv+=q.x*q.y; vv+=q.y*q.y;
    tx+=prepNodes[i].x*basis; ty+=prepNodes[i].y*basis;
  }
  let r0=vec3f(uu*vv-uv*uv,v*uv-u*vv,u*uv-v*uu);
  let r1=vec3f(r0.y,m*vv-v*v,u*v-m*uv);
  let r2=vec3f(r0.z,r1.z,m*uu-u*u);
  let det=dot(vec3f(m,u,v),r0);
  var x=vec3f(0.0); var y=vec3f(0.0);
  if(m>=3.0&&abs(det)>1e-18) {
    x=vec3f(dot(r0,tx),dot(r1,tx),dot(r2,tx))/det;
    y=vec3f(dot(r0,ty),dot(r1,ty),dot(r2,ty))/det;
  }
  prepMeta[2]=vec4f(x,1.0); prepMeta[3]=vec4f(y,0.0);
  for(var i=0u;i<216u;i++) {
    if(prepNodes[i].z!=0.0) {
      let q=prepareNodeUV(i); let basis=vec3f(1.0,q);
      prepNodes[i].x-=dot(x,basis); prepNodes[i].y-=dot(y,basis);
    }
  }
}
fn preparePupil(uv:vec2f)->AimResult {
  if((generation().grid.z&2u)==0u) { return AimResult(uv*(generation().geometry.w*0.5*0.9995),true); }
  let guess=pupilGuess(uv);
  if(guess.ok) { return guess; }
  return aimGenerated(uv);
}
fn prepareImage(q:vec2f)->AimResult {
  let ray=launchGenerated(q); let t=traceOptics(ray.p,ray.v,ray.t0,cfg.ns,true);
  if(!t.ok||abs(t.v.z)<1e-12) { return AimResult(vec2f(0.0),false); }
  let xy=fma(vec2f((cfg.zImg-t.p.z)/t.v.z),t.v.xy,t.p.xy);
  return AimResult(xy,finite(xy.x)&&finite(xy.y));
}
@compute @workgroup_size(PREP_GROUP_SIZE)
fn prepareAxialSamples(@builtin(global_invocation_id) gid:vec3u) {
  let i=gid.x; if(i>=prep.control.y) { return; }
  beginPreparation(true); usePreparedTable();
  let u=(f32(i)+0.5)/f32(prep.control.y); let q=preparePupil(vec2f(u,0.0));
  var image=AimResult(vec2f(0.0),false); if(q.ok) { image=prepareImage(q.q); }
  prepScratch[640u+prep.control.w*prep.control.y+i]=vec4f(abs(image.q.x),select(0.0,1.0,image.ok),0.0,0.0);
}

// One workgroup reduces an axial probe. Max is order-independent for the
// finite, nonnegative accepted radii.
// The same samples, acceptance predicate and final multiplication are used.
var<workgroup> axialRadiusMax:array<f32,64>;
@compute @workgroup_size(PREP_GROUP_SIZE)
fn prepareBlurReduceParallel(@builtin(local_invocation_id) local:vec3u) {
  let lane=local.x;var best=0.0;
  for(var i=lane;i<3u*prep.control.y;i+=64u) {
    let sample=prepScratch[640u+i];
    if(sample.y!=0.0){best=max(best,sample.x);}
  }
  axialRadiusMax[lane]=best;
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride>>=1u){
    if(lane<stride){axialRadiusMax[lane]=max(axialRadiusMax[lane],axialRadiusMax[lane+stride]);}
    workgroupBarrier();
  }
  if(lane==0u){prepMeta[7]=vec4f(2000.0*axialRadiusMax[0],0.0,0.0,0.0);}
}

// Bounded depth search controller. CPU submits a fixed set of optical stages;
// only this buffer decides brackets, trial distances and early completion.
// 0 header=(focus inverse distance,target,uMax,finite), 1 focus result,
// 2/3 back/front final results=(inverse distance,blur,reached,exists),
// 4/5 brackets=(lo delta,lo blur,hi delta,hi blur),
// 6/7 work=(chosen delta,done,active,0), 9/10 current trials,
// 16+ coarse measurements=(inverse distance,blur,reserved,reserved).
struct DepthSearchControl { record:u32, backStart:u32, backCount:u32, frontStart:u32, frontCount:u32, reserved:u32, padding:vec2u }
struct DepthNearConstants { optical:vec4f, aperture:vec4f, geometry:vec4f }
@group(2) @binding(15) var<storage,read_write> depthSearchState:array<vec4f>;
@group(2) @binding(16) var<uniform> depthSearchControl:DepthSearchControl;
@group(2) @binding(17) var<uniform> depthNear:DepthNearConstants;
@group(2) @binding(18) var<storage,read_write> depthGeneration:GenParams;
@group(2) @binding(19) var<storage,read_write> depthOpticalDispatch:array<vec4u>;
fn depthGeometryValue()->GenParams {
  let u=depthSearchState[depthSearchControl.record].x;
  let finiteObject=u>1e-7;let D=select(0.0,1.0/max(u,1e-30),finiteObject);
  let efl=depthNear.optical.x;let zEP=depthNear.optical.y;let pupilMag=depthNear.optical.z;
  let uEnd=fma(depthNear.optical.w,D,depthNear.aperture.x);
  let fno=depthNear.aperture.y;let mode=u32(depthNear.geometry.z);
  var epd=depthNear.aperture.z;
  if(mode==0u){epd=efl/fno;if(finiteObject&&abs(uEnd)>1e-12){epd=(zEP+D)/fno/abs(uEnd);}}
  var sd=depthNear.aperture.w;
  if(sd==0.0){sd=0.5*epd*select(1.0,abs(pupilMag),abs(pupilMag)>1e-9);}
  var g=gen;g.grid.z&=~6u;
  if(finiteObject){g.grid.z|=4u;}
  if(depthNear.geometry.w!=0.0&&sd>0.0){g.grid.z|=2u;}
  g.field.z=zEP;g.field.w=-D;
  g.geometry=vec4f(depthNear.geometry.xy,sd,epd);
  g.aim=vec4f(pupilMag,0.0,0.0,1e-3*max(1.0,epd*0.5));
  return g;
}
@compute @workgroup_size(1)
fn prepareDepthSearchGeometry() {
  depthGeneration=depthGeometryValue();let record=depthSearchControl.record;var runWork=true;
  if(record==9u||record==10u){let work=depthSearchState[6u+record-9u];runWork=work.z!=0.0&&work.y==0.0;}
  if(depthSearchControl.padding.x!=0u&&depthSearchState[record].z==0.0){runWork=false;}
  // This masks only already-completed or unreachable adaptive search work.
  // Coarse probes, uncompleted brackets and final reached-distance validation
  // keep their original ray budgets, shaders and failure classification.
  for(var i=0u;i<arrayLength(&depthOpticalDispatch);i++){
    let normal=depthOpticalDispatch[i].w;depthOpticalDispatch[i]=vec4u(select(0u,normal,runWork),1u,1u,normal);
  }
}
@compute @workgroup_size(1)
fn collectDepthBlur() {
  let i=depthSearchControl.record;
  if(depthSearchControl.padding.x!=0u&&depthSearchState[i].z==0.0){return;}
  depthSearchState[i].y=prepMeta[7].x;
}
fn depthSideSign(side:u32)->f32{return select(-1.0,1.0,side==1u);}
fn depthSideStart(side:u32)->u32{return select(depthSearchControl.backStart,depthSearchControl.frontStart,side==1u);}
fn depthSideCount(side:u32)->u32{return select(depthSearchControl.backCount,depthSearchControl.frontCount,side==1u);}
@compute @workgroup_size(1)
fn depthCoarseSchedule() {
  let focus=depthSearchState[16].y;
  let first=depthSearchState[depthSearchControl.record].y;
  let skip=finite(focus)&&finite(first)&&focus>=depthSearchState[0].y&&first>=depthSearchState[0].y;
  for(var i=0u;i<arrayLength(&depthOpticalDispatch);i++){
    let normal=depthOpticalDispatch[i].w;
    depthOpticalDispatch[i]=vec4u(select(normal,0u,skip),1u,1u,normal);
  }
}
@compute @workgroup_size(1)
fn depthSearchBracket() {
  let header=depthSearchState[0];let focus=depthSearchState[16];
  depthSearchState[1]=vec4f(header.x,focus.y,1.0,1.0);
  for(var side=0u;side<2u;side++){
    let count=depthSideCount(side);if(count==0u){continue;}
    if(finite(focus.y)&&focus.y>=header.y&&depthSearchState[16u+depthSideStart(side)].y>=header.y){
      depthSearchState[2u+side]=vec4f(header.x,focus.y,1.0,1.0);
      depthSearchState[6u+side]=vec4f(0.0,1.0,0.0,0.0);continue;
    }
    let start=depthSideStart(side);let sg=depthSideSign(side);
    let last=depthSearchState[16u+start+count-1u];
    depthSearchState[2u+side]=vec4f(last.xy,0.0,1.0);
    depthSearchState[6u+side]=vec4f(0.0,1.0,0.0,0.0);
    for(var i=0u;i<count;i++){
      let point=depthSearchState[16u+start+i];if(point.y<header.y){continue;}
      let previous=select(depthSearchState[16u+start+max(1u,i)-1u],focus,i==0u);
      let lo=sg*(previous.x-header.x);let hi=sg*(point.x-header.x);
      let done=previous.y>=header.y;
      depthSearchState[4u+side]=vec4f(lo,previous.y,hi,point.y);
      depthSearchState[6u+side]=vec4f(lo,select(0.0,1.0,done),1.0,0.0);
      depthSearchState[2u+side]=vec4f(previous.x,previous.y,1.0,1.0);break;
    }
  }
}
@compute @workgroup_size(1)
fn depthSearchNext() {
  let header=depthSearchState[0];
  for(var side=0u;side<2u;side++){
    var work=depthSearchState[6u+side];let br=depthSearchState[4u+side];var d=work.x;
    if(work.z!=0.0&&work.y==0.0){
      let width=br.z-br.x;d=br.x+(header.y-br.y)*width/(br.w-br.y);
      if(!finite(d)||d<=br.x+0.1*width||d>=br.z-0.1*width){d=0.5*(br.x+br.z);}
      work.x=d;depthSearchState[6u+side]=work;
    }
    let u=select(depthSearchState[2u+side].x,header.x+depthSideSign(side)*d,work.z!=0.0);
    depthSearchState[9u+side]=vec4f(u,0.0,0.0,0.0);
  }
}
@compute @workgroup_size(1)
fn depthSearchUpdate() {
  let header=depthSearchState[0];
  for(var side=0u;side<2u;side++){
    var work=depthSearchState[6u+side];if(work.z==0.0||work.y!=0.0){continue;}
    var br=depthSearchState[4u+side];let value=depthSearchState[9u+side].y;
    if(value>=header.y){br.z=work.x;br.w=value;}else{br.x=work.x;br.y=value;}
    if(abs(value-header.y)<=max(0.02,header.y*0.00005)){work.y=1.0;}
    depthSearchState[4u+side]=br;depthSearchState[6u+side]=work;
  }
}
@compute @workgroup_size(1)
fn depthSearchFinish() {
  let header=depthSearchState[0];
  for(var side=0u;side<2u;side++){
    var work=depthSearchState[6u+side];if(work.z==0.0){continue;}
    let br=depthSearchState[4u+side];var d=work.x;
    if(work.y==0.0){d=br.x+(header.y-br.y)*(br.z-br.x)/(br.w-br.y);d=select(br.z,clamp(d,br.x,br.z),finite(d));}
    depthSearchState[2u+side].x=header.x+depthSideSign(side)*d;
  }
}
