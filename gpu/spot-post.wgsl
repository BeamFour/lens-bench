// GPU-resident spot statistics and rasterization. Inputs remain on the GPU.
struct PostParams { grid:vec4u, center:vec4f, raster:vec4f, extra:vec4u }
// grid: partial offset, groups/grid, wavelengths, primary wavelength
// center: chief x high, y high, x low, y low
// raster: reserved, image width, point offset, reserved
// extra: field, number of fields, wavelength, focus-row flag
@group(0) @binding(10) var<uniform> post:PostParams;
@group(0) @binding(11) var<storage,read_write> postPartials:array<vec4f>;
@group(0) @binding(12) var<storage,read_write> postStats:array<vec4f>;
@group(0) @binding(13) var<storage,read_write> postPixels:array<atomic<u32>>;
@group(0) @binding(14) var<storage,read_write> postRgb:array<f32>;
@group(0) @binding(15) var<storage,read> postColors:array<vec4f>;
var<workgroup> postA:array<vec4f,256>;
var<workgroup> postB:array<vec4f,256>;
// Chan pairwise merge: mean and centered M2, never large raw moments.
fn postMoments(a:vec4f,b:vec4f)->vec4f {
  if(a.w==0.0) { return b; } if(b.w==0.0) { return a; }
  let n=a.w+b.w; let delta=b.xy-a.xy;
  let mean=fma(delta,vec2f(b.w/n),a.xy);
  let m2=a.z+b.z+dot(delta,delta)*(a.w*b.w/n);
  return vec4f(mean,m2,n);
}
fn postMerge(a:vec4f,b:vec4f)->vec4f { return vec4f(max(a.x,b.x),a.yzw+b.yzw); }
@compute @workgroup_size(256)
fn reduceSpot(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) block:vec3u) {
  let cell=cfg.base+block.x*256u+lane;
  var a=vec4f(0.0); var b=vec4f(0.0);
  if(cell<cfg.count) {
    let p=results[cell+u32(post.raster.z)];
    if(p.z!=-1.0) { b.y=1.0; }
    if(p.z!=-1.0&&p.z!=-2.0) { b.z=1.0; }
    if(p.z==1.0) {
      let d=(p.xy-post.center.xy)-post.center.zw;
      let r2=dot(d,d); a=vec4f(d,0.0,1.0); b.x=r2;
    }
  }
  postA[lane]=a; postB[lane]=b; workgroupBarrier();
  for(var step=128u;step>0u;step>>=1u) {
    if(lane<step) { postA[lane]=postMoments(postA[lane],postA[lane+step]); postB[lane]=postMerge(postB[lane],postB[lane+step]); }
    workgroupBarrier();
  }
  if(lane==0u) {
    let dest=2u*(post.grid.x+cfg.base/256u+block.x);
    postPartials[dest]=postA[0]; postPartials[dest+1u]=postB[0];
  }
}
@compute @workgroup_size(64)
fn finishSpotStats(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) block:vec3u) {
  let field=block.x;
  var a=vec4f(0.0); var b=vec4f(0.0);
  let groups=post.grid.y; let count=groups*post.grid.z;
  for(var item=lane;item<count;item+=64u) {
    let offset=2u*(field*count+item); let wavelength=item/groups;
    if(wavelength==post.grid.w) { a=postMoments(a,postPartials[offset]); b.y+=postPartials[offset+1u].y; }
    let v=postPartials[offset+1u]; b.x=max(b.x,v.x); b.z+=v.z;
  }
  postA[lane]=a; postB[lane]=b; workgroupBarrier();
  for(var step=32u;step>0u;step>>=1u) {
    if(lane<step) { postA[lane]=postMoments(postA[lane],postA[lane+step]); postB[lane]=postMerge(postB[lane],postB[lane+step]); }
    workgroupBarrier();
  }
  if(lane==0u) {
    let sum=postA[0]; let counts=postB[0];
    var rms=0.0; var throughput=0.0;
    if(sum.w>0.0) { rms=1000.0*sqrt(max(0.0,sum.z/sum.w)); throughput=min(1.0,sum.w/max(1.0,counts.y)); }
    postStats[field*2u]=vec4f(2000.0*sqrt(counts.x),rms,throughput,sum.w);
    postStats[field*2u+1u]=vec4f(counts.yz,0.0,0.0);
  }
}
fn postHalf()->f32 {
  var diameter=postStats[post.extra.x*2u].x;
  if(post.extra.w==0u) { for(var field=0u;field<post.extra.y;field++) { diameter=max(diameter,postStats[field*2u].x); } }
  return max(diameter*(1.12/2000.0),0.004);
}
@compute @workgroup_size(256)
fn rasterSpot(@builtin(global_invocation_id) gid:vec3u) {
  let cell=gid.x; if(cell>=cfg.count) { return; }
  let point=results[cell+u32(post.raster.z)]; if(point.z!=1.0) { return; }
  let width=u32(post.raster.y); let scale=f32(width)*0.5/postHalf();
  let d=(point.xy-post.center.xy)-post.center.zw;
  let pixel=vec2i(floor(vec2f(d.x,-d.y)*scale+f32(width)*0.5));
  if(any(pixel<vec2i(0))||any(pixel>=vec2i(i32(width)))) { return; }
  let offset=post.extra.z*width*width+u32(pixel.y)*width+u32(pixel.x);
  atomicAdd(&postPixels[offset],1u);
}
@compute @workgroup_size(256)
fn colorizeSpot(@builtin(global_invocation_id) gid:vec3u) {
  let width=u32(post.raster.y); let pixel=gid.x; if(pixel>=width*width) { return; }
  var normalization=vec3f(0.0);
  for(var li=0u;li<post.grid.z;li++) { let color=postColors[li]; normalization+=color.xyz*color.w; }
  normalization=max(normalization,vec3f(1e-6));
  var rgb=vec3f(0.0);
  for(var li=0u;li<post.grid.z;li++) {
    let color=postColors[li]; let count=f32(atomicLoad(&postPixels[li*width*width+pixel]));
    rgb+=(color.xyz*color.w/normalization)*count;
  }
  postRgb[pixel*3u]=rgb.x; postRgb[pixel*3u+1u]=rgb.y; postRgb[pixel*3u+2u]=rgb.z;
}
