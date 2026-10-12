// f32 optical tracer shared by depth search and final Spot tracing.
// Coefficients use a normalized radius to avoid f32 overflow/underflow in r^N.
struct Params { count:u32, ns:u32, zImg:f32, lam:f32, hard:u32, base:u32 }
struct Shape { z:f32, d:f32, ok:bool }
struct Hit { t:f32, ok:bool }
@group(0) @binding(0) var<storage,read> offsets:array<u32>;
@group(0) @binding(1) var<storage,read> surfaces:array<f32>;
@group(0) @binding(3) var<storage,read_write> results:array<vec4f>;
@group(0) @binding(4) var<uniform> cfg:Params;
fn at(s:u32, i:u32)->f32 { return surfaces[s+i]; }
fn finite(x:f32)->bool { return abs(x)<=3.402823466e38; }
fn shape(s:u32, r2:f32, r:f32)->Shape {
  let c=at(s,0u); var z=0.0; var d=0.0;
  if(c!=0.0) {
    let cq=(1.0+at(s,1u))*c*c*r2;
    let q=1.0-cq;
    if(q<=0.0) { return Shape(0.0,0.0,false); }
    let sq=sqrt(q); z=c*r2/(1.0+sq); d=c*r/sq;
  }
  let ne=u32(at(s,5u)); let no=u32(at(s,6u)); let scale=at(s,8u);
  let rho=r/scale; let rho2=rho*rho;
  // Horner in rho² for even terms and rho for general terms.
  // Separate derivative polynomials avoid accumulating rounded powers.
  var pe=0.0; var de=0.0;
  for(var i=ne;i>0u;i--) {
    let a=at(s,9u+i-1u); let power=f32(2u+2u*i);
    pe=fma(pe,rho2,a); de=fma(de,rho2,power*a);
  }
  var po=0.0; var dd=0.0;
  for(var j=no;j>0u;j--) {
    let a=at(s,9u+ne+j-1u); let power=f32(j);
    po=fma(po,rho,a); dd=fma(dd,rho,power*a);
  }
  let r3=rho2*rho; let r4=rho2*rho2;
  z=fma(pe,r4,fma(po,rho,z)); d+=(de*r3+dd)/scale;
  return Shape(z,d,finite(z)&&finite(d));
}
// Keep the closest-point frame for f32 stability; compare roots against
// the JS tangent-plane reference, without constraining propagation direction.
struct RayFrame { p:vec3f, t0:f32 }
fn rayPoint(p:vec3f,v:vec3f,t:f32)->vec3f { return fma(vec3f(t),v,p); }
fn centerRay(p:vec3f,v:vec3f,t0:f32)->RayFrame {
  let vv=dot(v,v);
  if(vv<=0.0) { return RayFrame(p,t0); }
  let shift=-dot(p,v)/vv;
  return RayFrame(rayPoint(p,v,shift),t0-shift);
}
fn tangentParameter(p:vec3f,v:vec3f,t0:f32)->f32 {
  if(abs(v.z)>=1e-9) { return -p.z/v.z; }
  return t0;
}
fn tTolerance(t:f32)->f32 { return max(1e-6,4.7683716e-7*(1.0+abs(t))); }
// FMA keeps the product's rounding remainder in a difference of products.
fn differenceOfProducts(a:f32,b:f32,c:f32,d:f32)->f32 {
  let cd=c*d;
  return fma(a,b,-cd)+fma(-c,d,cd);
}
// Here Shape.d is df/dt, rather than dsag/dr.
fn value(s:u32,p:vec3f,v:vec3f,t:f32)->Shape {
  let h=rayPoint(p,v,t); let r2=dot(h.xy,h.xy); let r=sqrt(r2);
  let ss=shape(s,r2,r);
  var dr=0.0; if(r>1e-12) { dr=dot(h.xy,v.xy)/r; }
  return Shape(h.z-ss.z,fma(-ss.d,dr,v.z),ss.ok);
}
fn opposite(a:f32,b:f32)->bool {
  return (a<=0.0&&b>=0.0)||(a>=0.0&&b<=0.0);
}
fn refineBracket(s:u32,p:vec3f,v:vec3f,left:f32,right:f32,fl:f32)->Hit {
  var a=left; var b=right; var fa=fl;
  for(var j=0u;j<48u;j++) {
    let m=a+0.5*(b-a);
    if(m==a||m==b||b-a<=tTolerance(m)) { return Hit(m,true); }
    let fm=value(s,p,v,m);
    if(!fm.ok) { return Hit(0.0,false); }
    if(fm.z==0.0) { return Hit(m,true); }
    if(opposite(fa,fm.z)) { b=m; } else { a=m; fa=fm.z; }
  }
  return Hit(a+0.5*(b-a),true);
}
fn scan(s:u32,p:vec3f,v:vec3f,t0:f32)->Hit {
  // The domain can be large for a weak base curvature. Its radius bounds the
  // search, but does not set the sampling step: a nearly flat asphere can have
  // two nearby roots even when its base radius is millions of millimetres.
  let local=max(1.0,max(at(s,8u),length(p)))*4.0+10.0;
  var radius=0.0; if(at(s,0u)!=0.0) { radius=abs(1.0/at(s,0u)); }
  let span=max(local,4.0*radius+10.0);
  var a=-0.25*span; var hi=max(span,a+local);
  var domainMiddle=hi;
  let c=at(s,0u); let ck=1.0+at(s,1u);
  // A conic with ck>0 only defines sag inside a radial cylinder. Restrict
  // the search analytically instead of repeatedly sampling outside it.
  // Bounds are expanded for roundoff; shape() still tests the original
  // domain, so this does not admit an otherwise invalid intersection.
  if(c!=0.0&&ck>0.0) {
    let denominator=ck*c*c;
    if(denominator>0.0) {
      let radialLimit2=1.0/denominator;
      if(finite(radialLimit2)) {
        let vv=dot(v.xy,v.xy);
        if(vv==0.0) {
          if(dot(p.xy,p.xy)>radialLimit2*1.0000009536743164) { return Hit(0.0,false); }
        } else {
          let middle=-dot(p.xy,v.xy)/vv;
          let nearest=rayPoint(p,v,middle).xy;
          let size=abs(p.xy)+abs(middle)*abs(v.xy);
          let radialError=1.9073486e-6*(radialLimit2+dot(size,size));
          let margin=radialLimit2-dot(nearest,nearest)+radialError;
          if(margin<0.0) { return Hit(0.0,false); }
          let halfWidth=sqrt(margin/vv);
          let parameterError=9.5367432e-7*(1.0+abs(middle)+halfWidth);
          a=max(a,middle-halfWidth-parameterError);
          hi=min(hi,middle+halfWidth+parameterError);
          if(hi<a) { return Hit(0.0,false); }
          domainMiddle=middle;
        }
      }
    }
  }
  if(c!=0.0&&ck>0.0) {
    // Bound the entire mathematical conic domain, not a mechanical aperture.
    // This is also valid for virtual aiming segments. Each polynomial term
    // contributes [min(0,edge),max(0,edge)] throughout 0<=r<=domainRadius.
    // Only a proved disjoint z interval exits; scanning order is unchanged.
    let radius2=1.0/(ck*c*c); let domainRadius=sqrt(radius2);
    let rho=domainRadius/at(s,8u); let rho2=rho*rho;
    let edge=c*radius2; var lo=min(0.0,edge); var up=max(0.0,edge);
    var magnitude=abs(edge); var power=rho2*rho2;
    let ne=u32(at(s,5u)); let no=u32(at(s,6u));
    for(var j=0u;j<ne;j++) {
      let term=at(s,9u+j)*power;
      lo+=min(0.0,term); up+=max(0.0,term); magnitude+=abs(term); power*=rho2;
    }
    power=rho;
    for(var j=0u;j<no;j++) {
      let term=at(s,9u+ne+j)*power;
      lo+=min(0.0,term); up+=max(0.0,term); magnitude+=abs(term); power*=rho;
    }
    // Broad outward padding covers f32 domain radius, powers and Horner
    // evaluation. Overflow/ill-defined envelopes simply retain the old scan.
    let pad=0.01*(1.0+magnitude);
    let za=fma(a,v.z,p.z); let zb=fma(hi,v.z,p.z);
    let rayPad=0.0001*(1.0+abs(p.z)+max(abs(a),abs(hi))*abs(v.z));
    if(ne+no<=64u&&finite(lo)&&finite(up)&&finite(pad)&&finite(za)&&finite(zb)&&
       (max(za,zb)+rayPad<lo-pad||min(za,zb)-rayPad>up+pad)) {
      return Hit(0.0,false);
    }
  }
  var fa=value(s,p,v,a); var step=local/64.0;
  for(var i=0u;i<768u;i++) {
    if(fa.ok&&fa.z==0.0) { return Hit(a,true); }
    let minStep=max(1e-6,2.3841858e-7*(1.0+abs(a)));
    var dt=max(minStep,min(step,hi-a));
    if(fa.ok&&abs(fa.d)>1e-20) { dt=min(dt,max(minStep,1.25*abs(fa.z/fa.d))); }
    var b=min(hi,a+dt);
    // Both expanded endpoints can be outside a very narrow real domain.
    // Visiting its centre ensures the entry/exit refinement can see it.
    if(a<domainMiddle&&b>domainMiddle) { b=domainMiddle; }
    if(b<=a) { break; }
    let fb=value(s,p,v,b);
    // A root may lie just inside the conic's radial domain. If the first
    // valid sample is already beyond it, simply resetting the previous
    // sample would lose that root. Resolve both domain transitions first.
    if(fa.ok!=fb.ok) {
      var l=a; var r=b;
      for(var j=0u;j<32u;j++) {
        let m=l+0.5*(r-l); if(m==l||m==r) { break; }
        let fm=value(s,p,v,m);
        if(fm.ok==fa.ok) { l=m; } else { r=m; }
      }
      if(fb.ok) {
        let fe=value(s,p,v,r);
        if(fe.ok&&opposite(fe.z,fb.z)) { return refineBracket(s,p,v,r,b,fe.z); }
        a=r; fa=fe; step=max(b-r,minStep); continue;
      } else {
        let fe=value(s,p,v,l);
        if(fe.ok&&opposite(fa.z,fe.z)) { return refineBracket(s,p,v,a,l,fa.z); }
      }
    }
    if(fa.ok&&fb.ok) {
      if(opposite(fa.z,fb.z)) { return refineBracket(s,p,v,a,b,fa.z); }
      // A sign change of the derivative can hide a pair of roots between
      // equal-sign samples. Find that extremum before advancing past it.
      if(opposite(fa.d,fb.d)&&fa.d!=fb.d) {
        var l=a; var r=b; var dl=fa.d;
        for(var j=0u;j<24u;j++) {
          let m=l+0.5*(r-l); if(m==l||m==r) { break; }
          let fm=value(s,p,v,m); if(!fm.ok) { break; }
          if(opposite(dl,fm.d)) { r=m; } else { l=m; dl=fm.d; }
        }
        let e=l+0.5*(r-l); let fe=value(s,p,v,e);
        if(fe.ok&&opposite(fa.z,fe.z)) { return refineBracket(s,p,v,a,e,fa.z); }
      }
      // Resolve strongly curved intervals rather than stepping across them.
      let bend=abs(fb.z-fma(fa.d,b-a,fa.z));
      if(bend>0.5*max(abs(fa.z),abs(fb.z))&&dt>4.0*minStep) {
        step=0.5*dt; continue;
      }
    } else if(fa.ok&&!fb.ok&&dt>4.0*minStep) {
      step=0.5*dt; continue;
    }
    a=b; fa=fb; if(a>=hi) { break; }
    step=min(max(local/64.0,1.5*dt),span/16.0);
  }
  return Hit(0.0,false);
}
fn conicValue(c:f32,k:f32,h:vec3f)->f32 {
  let radial=fma(h.x,h.x,h.y*h.y);
  return fma(c,radial,fma(c*(1.0+k)*h.z,h.z,-2.0*h.z));
}
fn intersect(s:u32,p:vec3f,v:vec3f,t0:f32)->Hit {
  let c=at(s,0u); let k=at(s,1u); let poly=at(s,5u)!=0.0||at(s,6u)!=0.0;
  let plane=tangentParameter(p,v,t0);
  var t=0.0; var valid=false;
  if(c==0.0) { if(abs(v.z)>=1e-14) { t=-p.z/v.z; valid=finite(t); } }
  else {
    let a=c*fma(v.x,v.x,fma(v.y,v.y,(1.0+k)*v.z*v.z));
    let b=2.0*fma(c,fma(p.x,v.x,fma(p.y,v.y,(1.0+k)*p.z*v.z)),-v.z);
    let cc=conicValue(c,k,p);
    // Only an exactly linear equation takes the linear path; an absolute
    // 1e-14 threshold changed the equation when lens units/curvature changed.
    if(a==0.0) { if(b!=0.0) { t=-cc/b; valid=finite(t); } }
    else {
      let disc=differenceOfProducts(b,b,4.0*a,cc);
      if(disc>=0.0) {
        let sq=sqrt(disc); let q=-0.5*(b+select(-sq,sq,b>=0.0));
        let t1=q/a; var t2=t1; if(q!=0.0) { t2=cc/q; }
        let low=min(t1,t2); let high=max(t1,t2);
        // Comparing two large |t-plane| values can erase the difference for
        // almost-horizontal rays. Outside the root interval, the nearest
        // endpoint is exact without that subtraction; retain JS's tie rule.
        if(plane<low) { t=low; }
        else if(plane>high) { t=high; }
        else { t=select(t2,t1,abs(t1-plane)<abs(t2-plane)); }
        valid=finite(t);
        // This is JS's signed wide-ray ordering rule, not a direction cutoff.
        // Only choose the earlier root when both lie on the vertex sheet.
        let h1=rayPoint(p,v,low); let h2=rayPoint(p,v,high);
        let epsilon=1e-9*(1.0+abs(t0-plane));
        if(v.z<0.17364817766693033&&t1!=t2&&low>t0+epsilon&&t!=low&&
           fma(-(1.0+k)*c,h1.z,1.0)>0.0&&fma(-(1.0+k)*c,h2.z,1.0)>0.0) { t=low; }
      }
    }
  }
  if(!poly) {
    if(!valid) { return Hit(0.0,false); }
    // Refine the selected conic root for f32 accuracy, without changing
    // its sheet or imposing a propagation sign constraint.
    if(c!=0.0) {
      for(var j=0u;j<8u;j++) {
        let h=rayPoint(p,v,t); let f=conicValue(c,k,h);
        let scale=abs(c)*(dot(h.xy,h.xy)+abs(1.0+k)*h.z*h.z)+2.0*abs(h.z);
        let fp=2.0*fma(c,fma(h.x,v.x,fma(h.y,v.y,(1.0+k)*h.z*v.z)),-v.z);
        // Near a tangent root, a residual at rounding scale divided by an
        // almost-zero derivative can send Newton far away from the surface.
        if(abs(fp)<1e-3*(1.0+abs(c)*length(h))&&abs(f)<=4.7683716e-7*max(1e-6,scale)) { break; }
        if(fp==0.0) { break; }
        let next=t-f/fp; if(!finite(next)||next==t) { break; }
        let nf=conicValue(c,k,rayPoint(p,v,next));
        if(!finite(nf)||abs(nf)>=abs(f)) { break; }
        t=next;
      }

    }
    return Hit(t,finite(t));
  }
  if(!valid) { return scan(s,p,v,t0); }
  for(var it=0u;it<40u;it++) {
    let h=rayPoint(p,v,t); let r2=dot(h.xy,h.xy); let r=sqrt(r2); let ss=shape(s,r2,r);
    if(!ss.ok) { break; }
    let f=h.z-ss.z;
    var dr=0.0; if(r>1e-12) { dr=dot(h.xy,v.xy)/r; }
    let fp=fma(-ss.d,dr,v.z);
    if(fp==0.0) { if(f==0.0) { return Hit(t,true); } break; }
    let delta=f/fp;
    // A small axial residual alone is insufficient at grazing incidence.
    if(abs(f)<=max(1e-6,2e-7*(abs(h.z)+abs(ss.z)))&&abs(delta)<=tTolerance(t)) {
      return Hit(t,true);
    }
    let next=t-delta;
    if(!finite(next)||next==t) { break; }
    t=next;
  }
  return scan(s,p,v,t0);
}
// Error terms remain f32; fma recovers rounding lost by the product/difference.
fn sumError(a:f32,b:f32,s:f32)->f32 {
  let bb=s-a; return (a-(s-bb))+(b-bb);
}
// Robust cutoff predicate: accumulate f32 mantissa squares in two u32 limbs.
// Inputs/outputs stay f32; integer arithmetic preserves the sign lost by
// subtracting nearly equal norms. No second precision mode or CPU retrace.
fn addLimbs(a:vec2u,b:vec2u)->vec2u {
  let lo=a.x+b.x; return vec2u(lo,a.y+b.y+select(0u,1u,lo<a.x));
}
fn subLimbs(a:vec2u,b:vec2u)->vec2u {
  return vec2u(a.x-b.x,a.y-b.y-select(0u,1u,a.x<b.x));
}
fn rawSquare(x:f32)->vec2u {
  let bits=bitcast<u32>(x)&0x7fffffffu; let e=(bits>>23u)&255u;
  let m=(bits&0x7fffffu)|select(0u,0x800000u,e!=0u);
  let a=m&65535u; let b=m>>16u; let base=a*a; let cross=2u*a*b;
  let lo=base+(cross<<16u); let hi=b*b+(cross>>16u)+select(0u,1u,lo<base);
  return vec2u(lo<<12u,(hi<<12u)|(lo>>20u));
}
fn squareLimbs(x:f32,exponent:u32)->vec2u {
  let e=max(1u,(bitcast<u32>(x)>>23u)&255u); let pair=rawSquare(x);
  let shift=2u*(exponent-e);
  if(shift==0u) { return pair; }
  if(shift<32u) { return vec2u((pair.x>>shift)|(pair.y<<(32u-shift)),pair.y>>shift); }
  if(shift<64u) { return vec2u(pair.y>>(shift-32u),0u); }
  return vec2u(0u);
}
fn squareTail(x:f32,exponent:u32)->f32 {
  let e=max(1u,(bitcast<u32>(x)>>23u)&255u); var pair=rawSquare(x);
  let shift=2u*(exponent-e); let power=2*i32(e)-312;
  if(shift==0u) { return 0.0; }
  if(shift<32u) { pair=vec2u(pair.x&((1u<<shift)-1u),0u); }
  else if(shift<64u) { pair.y&=((1u<<(shift-32u))-1u); }
  return ldexp(f32(pair.y),power+32)+ldexp(f32(pair.x),power);
}
fn normRemainder(n:f32,v:vec3f)->f32 {
  let en=(bitcast<u32>(n)>>23u)&255u;
  let ev=(bitcast<vec3u>(v)>>vec3u(23u))&vec3u(255u);
  let e=max(1u,max(en,max(ev.x,max(ev.y,ev.z))));
  let a=squareLimbs(n,e);
  let b=addLimbs(addLimbs(squareLimbs(v.x,e),squareLimbs(v.y,e)),squareLimbs(v.z,e));
  // n carries the maximum exponent for any possible transmitted ray, so
  // its square has no discarded tail. A tie can still hide small positive
  // transverse squares; restore them rather than accepting a false cutoff.
  if(all(a==b)&&en==e) {
    let tail=squareTail(v.x,e)+squareTail(v.y,e)+squareTail(v.z,e);
    if(tail>0.0) { return -tail; }
    // Squares below f32's range still have a definite negative sign.
    if((v.x!=0.0&&all(squareLimbs(v.x,e)==vec2u(0u)))||
       (v.y!=0.0&&all(squareLimbs(v.y,e)==vec2u(0u)))||
       (v.z!=0.0&&all(squareLimbs(v.z,e)==vec2u(0u)))) { return -1.17549435e-38; }
  }
  let negative=a.y<b.y||(a.y==b.y&&a.x<b.x);
  let delta=select(subLimbs(a,b),subLimbs(b,a),negative);
  let power=2*i32(e)-312;
  let magnitude=ldexp(f32(delta.y),power+32)+ldexp(f32(delta.x),power);
  return select(magnitude,-magnitude,negative);
}
fn refractRadicand(mu:f32,ci:f32)->f32 {
  let m2=mu*mu; let c2=ci*ci;
  let em=fma(mu,mu,-m2); let ec=fma(ci,ci,-c2);
  let sin2=1.0-c2; let es=sumError(1.0,-c2,sin2)-ec;
  let q=m2*sin2; let eq=fma(m2,sin2,-q)+m2*es+em*sin2;
  let d=1.0-q; return d+(sumError(1.0,-q,d)-eq);
}

struct TraceState { p:vec3f, v:vec3f, ok:bool }
// Shared by uploaded rays and generated rays; aiming traces through the stop
// with identical optics and disabled aperture clipping.
fn traceOptics(p0:vec3f,v0:vec3f,firstT0:f32,nEnd:u32,checkAp:bool)->TraceState {
  var p=p0; var v=v0;

  var np=1.0; var good=true;
  for(var i=0u;i<nEnd;i++) {
    let s=offsets[i];
    let start=vec3f(p.xy,p.z-at(s,2u));
    var startT=0.0; if(i==0u) { startT=firstT0; }
    let frame=centerRay(start,v,startT);
    let lp=frame.p; let hit=intersect(s,lp,v,frame.t0);
    if(!hit.ok) {  good=false; break; }
    let h=rayPoint(lp,v,hit.t); let r2=h.x*h.x+h.y*h.y; let r=sqrt(r2); let ap=at(s,3u);
    if(checkAp&&ap>=0.0) {
      if(r>ap*1.0000001) {  good=false; break; }
    }
    p=h; // p.z stays relative to this surface; the next record holds delta-z.
    let ss=shape(s,r2,r); if(!ss.ok) {  good=false; break; }
    var nn=vec3f(0.0,0.0,1.0); if(r>1e-12) { let normScale=max(1.0,abs(ss.d)); nn=normalize(vec3f(-(ss.d/normScale)*h.xy/r,1.0/normScale)); }
    let n=at(s,4u); let nb=u32(at(s,7u));
    if(nb>0u) {
      let scale=at(s,8u); let rho=r/scale; let rho2=rho*rho;
      let coeff=9u+u32(at(s,5u))+u32(at(s,6u));
      var dp=0.0;
      for(var bi=nb;bi>0u;bi--) { dp=fma(dp,rho2,f32(2u*bi)*at(s,coeff+bi-1u)); }
      let dph=dp*rho/scale;
      var grad=vec3f(0.0); if(r>1e-12) { grad=vec3f(dph*h.xy/r,0.0); }
      let gk=(cfg.lam/1000.0)/(2.0*3.141592653589793);
      var tv=np*v+gk*grad; tv-=dot(tv,nn)*nn;
      let n2=n*n; var nn2=n2-dot(tv,tv);
      if(abs(nn2)<1e-4*n2) { nn2=normRemainder(n,tv); }
      if(nn2<0.0) {  good=false; break; }
      v=normalize((tv+sqrt(nn2)*nn)/n);
    } else {
      let mu=np/n; let ci=dot(v,nn);
      // The CPU reference chooses a positive-normal transmitted branch.
      // Equal indices and forward normal need no subtraction or sqrt.
      if(np==n&&ci>=0.0) {
        // Identity: preserve the small grazing component rather than rounding it away.
      } else {
        let s2=refractRadicand(mu,ci);
        if(s2<0.0) {  good=false; break; }
        let tangent=v-ci*nn;
        v=normalize(mu*tangent+sqrt(s2)*nn);
      }
    }
    np=n;
  }
  return TraceState(p,v,good);
}
