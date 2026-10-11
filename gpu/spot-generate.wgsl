// Appended to spot.wgsl. All per-cell sampling, launch construction and
// unsuccessful-table aiming work executes on the GPU.
// GenParams: eight 16-byte vectors; grid is u32, all following vectors f32.
struct GenParams {
  grid:vec4u,      // side, seed, flags(tab=1,aim=2,finite=4,ellipse=8,chief=16), stop
  field:vec4f,     // sin(theta), cos(theta), zEP, zObj
  geometry:vec4f,  // zStart, span, sdStop, epd
  aim:vec4f,       // pupilMag, chief ex, chief ey, finite-difference step
  jac:vec4f,       // dStopX/dEx, dStopX/dEy, dStopY/dEx, dStopY/dEy
  ellipse:vec4f,   // cx, cy, ax, ay
  affineX:vec4f,   // A0, A1, A2, reserved
  affineY:vec4f    // B0, B1, B2, reserved
}
@group(0) @binding(5) var<uniform> gen:GenParams;
@group(0) @binding(6) var<storage,read> aimNodes:array<vec4f>; // ex residual, ey residual, valid, reserved
// Preparation kernels can change field/chief parameters within one GPU
// invocation. Ordinary generation keeps the original uniform path.
var<private> generationOverride:GenParams;
var<private> generationOverrideEnabled:bool=false;
fn generation()->GenParams {
  if(generationOverrideEnabled) { return generationOverride; }
  return gen;
}
struct LaunchRay { p:vec3f, v:vec3f, t0:f32 }
struct AimResult { q:vec2f, ok:bool }
struct AimGuess { q:vec2f, quality:u32 } // 0 unavailable, 1 partial-cell seed, 2 full interpolation

fn mulberryAt(index:u32)->u32 {
  let state=generation().grid.y+0x6D2B79F5u*(index+1u);
  var t=(state^(state>>15u))*(1u|state);
  t^=t+(t^(t>>7u))*(61u|t);
  return t^(t>>14u);
}
fn multiply32(a:u32,b:u32)->vec2u {
  let a0=a&65535u; let a1=a>>16u; let b0=b&65535u; let b1=b>>16u;
  let w0=a0*b0; let t=a1*b0+(w0>>16u); let w1=t&65535u; let w2=t>>16u;
  let t2=a0*b1+w1;
  return vec2u((t2<<16u)|(w0&65535u),a1*b1+w2+(t2>>16u));
}
fn add96(a:vec3u,b:vec3u)->vec3u {
  let lo=a.x+b.x; let carry=select(0u,1u,lo<a.x);
  let mid=a.y+b.y; let hi=a.z+b.z+select(0u,1u,mid<a.y);
  let carried=mid+carry;
  return vec3u(lo,carried,hi+select(0u,1u,carried<mid));
}
fn sampleNumerator(cell:u32,random:u32,side:u32)->vec2u {
  let lo=random<<1u; let hi=2*i32(cell)-i32(side)+i32(random>>31u);
  if(hi<0) { return vec2u(0u-lo,u32(-hi)-select(1u,0u,lo==0u)); }
  return vec2u(lo,u32(hi));
}
fn squareCoordinate(v:vec2u)->vec3u {
  let low=multiply32(v.x,v.x); let cross=multiply32(v.x,v.y);
  return add96(vec3u(low,0u),vec3u(0u,cross.x<<1u,(cross.y<<1u)|(cross.x>>31u)))+vec3u(0u,0u,v.y*v.y);
}
fn insideSampleCircle(i:u32,j:u32,ru:u32,rv:u32,uv:vec2f)->bool {
  let radius2=dot(uv,uv);
  if(abs(radius2-1.0)>2e-6) { return radius2<=1.0; }
  // Each coordinate is an exact integer divided by side*2^32. At the
  // boundary, compare 96-bit integer squares so converting the random
  // numbers to f32 cannot change the count of sampled pupil cells.
  let x=sampleNumerator(i,ru,generation().grid.x); let y=sampleNumerator(j,rv,generation().grid.x);
  let sum=add96(squareCoordinate(x),squareCoordinate(y)); let limit=generation().grid.x*generation().grid.x;
  return sum.z<limit||(sum.z==limit&&sum.y==0u&&sum.x==0u);
}
fn launchGenerated(q:vec2f)->LaunchRay {
  let sn=generation().field.x; let cs=generation().field.y; let zep=generation().field.z;
  let wide=cs<0.17364817766693033;
  let finiteObject=(generation().grid.z&4u)!=0u;
  let dz=zep-generation().field.w;
  let lf=zep-generation().geometry.x+generation().geometry.y+50.0;
  if(wide&&(!finiteObject||dz>0.0)) {
    let o=vec3f(q.x,q.y*cs,fma(-q.y,sn,zep));
    if(finiteObject) {
      let rw=dz/0.17364817766693033;
      let scaled=vec3f(q.x/rw,fma(q.y/rw,cs,sn),fma(-q.y/rw,sn,cs));
      let len=length(scaled); let v=scaled/len;
      return LaunchRay(o,v,-min(rw*len,lf));
    }
    return LaunchRay(o,vec3f(0.0,sn,cs),-lf);
  }
  let o=vec3f(q,zep);
  if(finiteObject) {
    // The ray goes through o. Working with scaled direction components avoids
    // constructing a distant object point and cancelling it back to the lens.
    let scale=max(1.0,abs(dz));
    let raw=vec3f(q.x/scale,fma(sn/cs,dz/scale,q.y/scale),dz/scale);
    let len=length(raw);
    let orientation=select(-1.0,1.0,dz>=0.0);
    return LaunchRay(o,orientation*raw/len,-orientation*scale*len);
  }
  return LaunchRay(o,vec3f(0.0,sn,cs),(generation().geometry.x-zep)/cs);
}
fn stopHit(q:vec2f)->AimResult {


  let ray=launchGenerated(q);
  let t=traceOptics(ray.p,ray.v,ray.t0,min(generation().grid.w+1u,cfg.ns),false);
  let valid=t.ok&&finite(t.p.x)&&finite(t.p.y);


  return AimResult(t.p.xy,valid);
}
fn affinePupil(uv:vec2f)->vec2f {
  return vec2f(fma(generation().affineX.y,uv.x,fma(generation().affineX.z,uv.y,generation().affineX.x)),
               fma(generation().affineY.y,uv.x,fma(generation().affineY.z,uv.y,generation().affineY.x)));
}
fn pupilGuess(uv:vec2f)->AimGuess {
  if((generation().grid.z&1u)==0u) { return AimGuess(vec2f(0.0),0u); }
  let radius=min(1.0,length(uv));
  // atan2(0,0) is outside WGSL's defined domain. The duplicated centre
  // nodes represent a single point, so its angular coordinate is exactly 0.
  var phi=0.0;
  if(radius>0.0) { phi=atan2(uv.y,uv.x); }
  if(phi<0.0) { phi+=6.283185307179586; }
  let fr=radius*8.0; let ring=min(u32(floor(fr)),7u); let a=fr-f32(ring);
  let fp=phi*(24.0/6.283185307179586); let angle=u32(floor(fp)); let b=fp-f32(angle);
  let j0=angle%24u; let j1=(j0+1u)%24u;
  let indices=array<u32,4>(ring*24u+j0,ring*24u+j1,(ring+1u)*24u+j0,(ring+1u)*24u+j1);
  let weights=array<f32,4>((1.0-a)*(1.0-b),(1.0-a)*b,a*(1.0-b),a*b);
  var residual=vec2f(0.0); var weight=0.0; var count=0u;
  for(var i=0u;i<4u;i++) {
    let node=aimNodes[indices[i]];
    if(node.z!=0.0) { residual+=weights[i]*node.xy; weight+=weights[i]; count++; }
  }
  if(count==4u) { return AimGuess(affinePupil(uv)+residual,2u); }
  // A partial table cell requires the ordinary aiming solver; it is neither
  // a proven blocked ray nor an already-aimed ray.
  if(weight>0.0) { return AimGuess(affinePupil(uv)+residual/weight,1u); }
  return AimGuess(vec2f(0.0),0u);
}
fn aimCorrection(j:vec4f,g:vec2f)->AimResult {
  let det=differenceOfProducts(j.x,j.w,j.y,j.z);
  if(!finite(det)||abs(det)<1e-14) { return AimResult(vec2f(0.0),false); }
  let d=vec2f(differenceOfProducts(j.y,g.y,g.x,j.w),differenceOfProducts(j.z,g.x,j.x,g.y))/det;
  return AimResult(d,finite(d.x)&&finite(d.y));
}
struct AimJacobian { value:vec4f, ok:bool }
fn estimateAimJacobian(q:vec2f,point:vec2f)->AimJacobian {
  let probe=max(1e-4,generation().aim.w);
  let hx=stopHit(q+vec2f(probe,0.0)); let hy=stopHit(q+vec2f(0.0,probe));
  if(!hx.ok||!hy.ok) { return AimJacobian(vec4f(0.0),false); }
  let jx=(hx.q-point)/probe; let jy=(hy.q-point)/probe;
  return AimJacobian(vec4f(jx.x,jy.x,jx.y,jy.y),true);
}
// Match OPT.aim: undamped Newton with the same two deterministic seeds.
// Keep f32-stable arithmetic in aimCorrection and the optical tracer.
fn aimNewton(seed:vec2f,stopGoal:vec2f,useReference:bool)->AimResult {
  let scl=max(1.0,select(0.5*generation().geometry.w,generation().geometry.z,generation().geometry.z>0.0));
  let tolerance2=1e-12*scl*scl; let acceptable2=1e-8*scl*scl;
  var q=seed;
  for(var iteration=0u;iteration<14u;iteration++) {
    let current=stopHit(q);
    if(!current.ok) { return AimResult(q,false); }
    let error=current.q-stopGoal; let e2=dot(error,error);
    if(e2<tolerance2) { return AimResult(q,true); }
    var jac=generation().jac;
    if(!useReference||iteration>=4u) {
      let derivative=estimateAimJacobian(q,current.q);
      if(!derivative.ok) { return AimResult(q,false); }
      jac=derivative.value;
    }
    let correction=aimCorrection(jac,error);
    if(!correction.ok) { return AimResult(q,false); }
    let delta=correction.q;
    q+=delta;
    // Like the CPU, only a negligible Newton step permits the looser residual.
    if(dot(delta,delta)<1e-16) { return AimResult(q,e2<acceptable2); }
  }
  return AimResult(q,false);
}
fn aimGenerated(uv:vec2f)->AimResult {
  let stopGoal=uv*(generation().geometry.z*0.9995);
  let hasReference=(generation().grid.z&16u)!=0u;
  let mag=select(1.0,generation().aim.x,abs(generation().aim.x)>1e-9);
  let paraxial=stopGoal/mag;
  var predicted=paraxial;
  if(hasReference) {
    let step=aimCorrection(generation().jac,-stopGoal);
    if(step.ok) { predicted=generation().aim.yz+step.q; }
  }
  var solution=aimNewton(predicted,stopGoal,hasReference);
  if(!solution.ok&&hasReference) {
    solution=aimNewton(paraxial,stopGoal,false);
  }
  return solution;
}
struct SampleCell { uv:vec2f, status:f32 }
fn sampleCell(cell:u32)->SampleCell {
  let side=generation().grid.x; let i=cell/side; let j=cell%side;
  let rawU=mulberryAt(2u*cell); let rawV=mulberryAt(2u*cell+1u);
  let ru=f32(rawU)*(1.0/4294967296.0);
  let rv=f32(rawV)*(1.0/4294967296.0);
  let uv=vec2f(-1.0)+(vec2f(f32(i),f32(j))+vec2f(ru,rv))*(2.0/f32(side));
  if(!insideSampleCircle(i,j,rawU,rawV,uv)) { return SampleCell(uv,-1.0); }
  if((generation().grid.z&8u)!=0u) {
    let ell=(uv-generation().ellipse.xy)/generation().ellipse.zw;
    if(dot(ell,ell)>1.0) { return SampleCell(uv,-2.0); }
  }
  return SampleCell(uv,1.0);
}

// Shared ray cache between generation, bounded aiming and final tracing.
struct GeneratedRay { p:vec4f, v:vec4f }
@group(0) @binding(7) var<storage,read_write> generatedRays:array<GeneratedRay>;
