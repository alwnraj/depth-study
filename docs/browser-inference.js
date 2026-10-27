// ONNX Runtime Web 1.23.2 (2025-11-04); models are pinned to immutable revisions.
const modelSources = {
  relative: {base:'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/34f40a4c93991327074907b51792dbc27e7d6f1c/onnx/', file:'model.onnx', hash:'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c'},
  indoor: {base:'https://huggingface.co/77ukhtar/depth-anything-v2-metric-onnx/resolve/a4259a3c45137b6eb32c84fcd95b86cd54c255b9/', file:'model.onnx', hash:'badcaa28c923da4b0bfaa370ed709acfa00e9f743d295d5443e2149a383413c9'},
  outdoor: {base:'https://huggingface.co/sollaholla/depth-anything-v2-metric-vits-vkitti/resolve/f3f42bbfb266b8c7dbd0ef6cf24bfc4b167717f5/', file:'depth_anything_v2_metric_vits_vkitti_518x518.onnx', hash:'d2962720aa4c73677bff72f230f814df80d60b7d72759cf1d36f3846c979b568', data:'depth_anything_v2_metric_vits_vkitti_518x518.onnx.data', dataHash:'eb38b775e6247f59dbef340364cf6c991b87cbf221a92937018777ec51a84431'}
};
const loadedSessions = {};
async function verifiedDownload(url, hash) {
  const response=await fetch(url); if(!response.ok) throw Error('Model download failed ('+response.status+').');
  const bytes=await response.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)), b=>b.toString(16).padStart(2,'0')).join('');
  if(digest!==hash) throw Error('Model checksum did not match.');
  return new Uint8Array(bytes);
}
async function loadBrowserModel(kind, update) {
  if(loadedSessions[kind]) return loadedSessions[kind];
  if(typeof ort==='undefined') throw Error('Inference runtime did not load. Check your connection.');
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
  update('Downloading '+kind+' model (about 100 MB on first use)…');
  const cfg=modelSources[kind], graph=await verifiedDownload(cfg.base+cfg.file,cfg.hash);
  const options={executionProviders:['wasm']};
  if(cfg.data) {const weights=await verifiedDownload(cfg.base+cfg.data,cfg.dataHash);options.externalData=[{path:cfg.data,data:weights}];}
  update('Preparing '+kind+' model…');
  const session=await ort.InferenceSession.create(graph,options);
  loadedSessions[kind]=session; return session;
}
function canvasImage(img,w,h) {const c=document.createElement('canvas');c.width=w;c.height=h;const cx=c.getContext('2d');cx.imageSmoothingQuality='high';cx.drawImage(img,0,0,w,h);return c;}
function tensorFrom(canvas) {
  const {width:w,height:h}=canvas, rgba=canvas.getContext('2d').getImageData(0,0,w,h).data, n=w*h, values=new Float32Array(3*n);
  const mean=[.485,.456,.406],sd=[.229,.224,.225];
  for(let i=0;i<n;i++) for(let c=0;c<3;c++) values[c*n+i]=(rgba[4*i+c]/255-mean[c])/sd[c];
  return new ort.Tensor('float32',values,[1,3,h,w]);
}
function resizeMap(input,iw,ih,ow,oh) {
  const out=new Float32Array(ow*oh);
  for(let y=0;y<oh;y++) {const fy=Math.max(0,Math.min(ih-1,(y+.5)*ih/oh-.5)), y0=Math.floor(fy),y1=Math.min(y0+1,ih-1),dy=fy-y0;
    for(let x=0;x<ow;x++){const fx=Math.max(0,Math.min(iw-1,(x+.5)*iw/ow-.5)),x0=Math.floor(fx),x1=Math.min(x0+1,iw-1),dx=fx-x0;
      out[y*ow+x]=(input[y0*iw+x0]*(1-dx)+input[y0*iw+x1]*dx)*(1-dy)+(input[y1*iw+x0]*(1-dx)+input[y1*iw+x1]*dx)*dy;}}
  return out;
}
async function runBrowserModel(session,canvas) {
  const result=await session.run({[session.inputNames[0]]:tensorFrom(canvas)}), value=result[session.outputNames[0]];
  const dims=value.dims, raw=value.data, h=dims[dims.length-2], w=dims[dims.length-1];
  if(raw.length!==h*w || !raw.every(Number.isFinite)) throw Error('Model returned an invalid depth map.');
  return {data:raw,w,h};
}
async function estimateInBrowser(file,scene,update) {
  const image=await createImageBitmap(file,{imageOrientation:'from-image'});
  try {
    if(image.width*image.height>30000000) throw Error('Choose an image under 30 megapixels.');
    const scale=518/Math.max(image.width,image.height), mult=n=>Math.max(14,Math.round(n/14)*14);
    const rw=mult(image.width*scale),rh=mult(image.height*scale);
    const relative=await loadBrowserModel('relative',update);update('Estimating relative depth…');
    const result=await runBrowserModel(relative,canvasImage(image,rw,rh));
    const metric=await loadBrowserModel(scene,update);update('Estimating metric depth…');
    const meters=await runBrowserModel(metric,canvasImage(image,518,518));
    const width=Math.min(160,Math.max(2,Math.round(400*image.width/image.height))), height=Math.min(400,Math.max(2,Math.round(width*image.height/image.width)));
    const sampled=resizeMap(result.data,result.w,result.h,width,height), ordered=Array.from(sampled).sort((a,b)=>a-b);
    const percentile=p=>{let i=(ordered.length-1)*p,lo=Math.floor(i),hi=Math.ceil(i);return ordered[lo]+(ordered[hi]-ordered[lo])*(i-lo);};
    const lo=percentile(.02),hi=percentile(.98),near=Array.from(sampled,v=>hi===lo?0:Math.max(0,Math.min(1,(v-lo)/(hi-lo))));
    let min=Infinity,max=-Infinity;for(const v of result.data){min=Math.min(min,v);max=Math.max(max,v);}
    const gray=document.createElement('canvas');gray.width=result.w;gray.height=result.h;const gx=gray.getContext('2d'), imageData=gx.createImageData(result.w,result.h);
    for(let i=0;i<result.data.length;i++){let v=max===min?0:Math.round((result.data[i]-min)/(max-min)*255);imageData.data.set([v,v,v,255],i*4);}gx.putImageData(imageData,0,0);
    const displayScale=Math.min(1,1600/Math.max(image.width,image.height)),dw=Math.max(1,Math.round(image.width*displayScale)),dh=Math.max(1,Math.round(image.height*displayScale));
    return {image:canvasImage(image,dw,dh).toDataURL('image/png').split(',')[1],depth:canvasImage(gray,dw,dh).toDataURL('image/png').split(',')[1],colors:canvasImage(image,width,height).toDataURL('image/png').split(',')[1], width,height,near,original_size:[image.width,image.height],metric:{values:Array.from(resizeMap(meters.data,meters.w,meters.h,256,256)),width:256,height:256,scene}};
  } finally {image.close();}
}
