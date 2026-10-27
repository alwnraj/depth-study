const file = document.querySelector('#file'), drop = document.querySelector('#drop');
const status = document.querySelector('#status'), results = document.querySelector('#results');
const original = document.querySelector('#original'), depth = document.querySelector('#depth');
const slider = document.querySelector('#slider'), clip = document.querySelector('#depth-clip'), divider = document.querySelector('#divider');
const surface = document.querySelector('#surface');
let renderer = null, metric = null, metricPoint = [.5, .5];
const scene = document.querySelector('#scene'), measureImage = document.querySelector('#measure-image');
function selectMetric(x,y) {
  if (!metric) return;
  x=Math.max(0,Math.min(1,x)); y=Math.max(0,Math.min(1,y)); metricPoint=[x,y];
  const col=Math.min(metric.width-1,Math.floor(x*metric.width)), row=Math.min(metric.height-1,Math.floor(y*metric.height));
  const marker=document.querySelector('#measure-marker'); marker.hidden=false;marker.style.left=(x*100)+'%';marker.style.top=(y*100)+'%';
  document.querySelector('#metric-value').innerHTML=metric.values[row*metric.width+col].toFixed(1)+' <small>m</small>';
  document.querySelector('#metric-instructions').textContent='Estimated depth · '+metric.scene+' model';
}
scene.addEventListener('change',()=>{if(metric) status.textContent='Scene selection changed. Upload the image again to apply it.';});
measureImage.addEventListener('click',e=>{const r=measureImage.getBoundingClientRect();selectMetric((e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height);});
measureImage.addEventListener('keydown',e=>{const d={ArrowLeft:[-.01,0],ArrowRight:[.01,0],ArrowUp:[0,-.01],ArrowDown:[0,.01]}[e.key];if(d){e.preventDefault();selectMetric(metricPoint[0]+d[0],metricPoint[1]+d[1]);}});
function compare() { clip.style.width = slider.value + '%'; divider.style.left = slider.value + '%'; }
slider.addEventListener('input', compare);
document.querySelector('#compare').addEventListener('pointerdown', e => {
  const box = e.currentTarget.getBoundingClientRect(); slider.value = Math.max(0, Math.min(100, 100 * (e.clientX - box.left) / box.width)); compare();
  e.currentTarget.setPointerCapture(e.pointerId);
});
document.querySelector('#compare').addEventListener('pointermove', e => {
  if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
  const box = e.currentTarget.getBoundingClientRect(); slider.value = Math.max(0, Math.min(100, 100 * (e.clientX - box.left) / box.width)); compare();
});
async function upload(image) {
  if (!image) return;
  if (!image.type.startsWith('image/') || image.size > 20 * 1024 * 1024) { status.textContent = 'Choose an image under 20 MB.'; return; }
  status.textContent = 'Estimating depth locally…'; file.disabled = true; drop.classList.add('busy');
  try {
    const data = await estimateInBrowser(image, scene.value, message => { status.textContent = message; });
    metric=data.metric; metricPoint=[.5,.5]; document.querySelector('#measure-marker').hidden=true;
    document.querySelector('#metric-value').innerHTML='— <small>m</small>';
    document.querySelector('#metric-instructions').textContent=metric ? 'Click a point in the image to see its estimated depth.' : 'Metric model unavailable. Run the model download command and restart the server.';
    document.querySelector('#measure-photo').src='data:image/png;base64,'+data.image;
    original.src = 'data:image/png;base64,' + data.image;
    depth.src = 'data:image/png;base64,' + data.depth;
    document.querySelector('#download').href = depth.src;
    results.hidden = false; status.textContent = 'Depth estimate ready · ' + data.original_size.join(' × ') + ' pixels';
    slider.value = 50; compare();
    try { renderer = await createSurface(data); document.querySelector('#surface-error').hidden = true; }
    catch (err) { renderer = null; const box = document.querySelector('#surface-error'); box.textContent = '3D preview unavailable: ' + err.message; box.hidden = false; }
    results.scrollIntoView({behavior:'smooth', block:'start'});
  } catch (err) { status.textContent = 'Could not estimate depth: ' + err.message; }
  finally { file.disabled = false; file.value = ''; drop.classList.remove('busy'); }
}
file.addEventListener('change', () => upload(file.files[0]));
for (const event of ['dragenter','dragover']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('drag'); });
for (const event of ['dragleave','drop']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('drag'); });
drop.addEventListener('drop', e => { if (!file.disabled) upload(e.dataTransfer.files[0]); });

async function createSurface(data) {
  const gl = surface.getContext('webgl', {antialias:true}); if (!gl) throw new Error('WebGL is not available in this browser.');
  const colors = new Image(); colors.src = 'data:image/png;base64,' + data.colors; await colors.decode();
  const c = document.createElement('canvas'); c.width=data.width; c.height=data.height;
  const ctx=c.getContext('2d'); ctx.drawImage(colors,0,0); const pixels=ctx.getImageData(0,0,c.width,c.height).data;
  const verts=[], indices=[], w=data.width, h=data.height, n=data.near, focal=w/(2*Math.tan(Math.PI/6));
  for(let y=0;y<h;y++) for(let x=0;x<w;x++) { const i=y*w+x, z=3-2*n[i]; verts.push((x-(w-1)/2)*z/focal, ((h-1)/2-y)*z/focal, z-2, pixels[4*i]/255,pixels[4*i+1]/255,pixels[4*i+2]/255); }
  function add(a,b,c) { if(Math.max(n[a],n[b],n[c])-Math.min(n[a],n[b],n[c])<=0.12) indices.push(a,b,c); }
  for(let y=0;y<h-1;y++) for(let x=0;x<w-1;x++) {let a=y*w+x,b=a+1,c=a+w,d=c+1;add(a,c,b);add(b,c,d);}
  function shader(type, source) { const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));return s; }
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 p; attribute vec3 color; uniform vec2 angle; uniform vec2 scale; varying vec3 vColor; void main(){ float cy=cos(angle.x),sy=sin(angle.x),cx=cos(angle.y),sx=sin(angle.y); vec3 q=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z); q=vec3(q.x,cx*q.y-sx*q.z,sx*q.y+cx*q.z); gl_Position=vec4(q.x*scale.x,q.y*scale.y,q.z*0.15,1.0); vColor=color; }`));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float; varying vec3 vColor; void main(){gl_FragColor=vec4(vColor,1.0);}`)); gl.linkProgram(program); if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error('Could not initialize the surface.'); gl.useProgram(program);
  const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);
  for(const [name,offset] of [['p',0],['color',12]]) {const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,24,offset);}
  const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW);
  let yaw=-0.28,pitch=0.06;
  function draw(){const rect=surface.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);surface.width=Math.max(1,Math.round(rect.width*dpr));surface.height=Math.max(1,Math.round(rect.height*dpr));gl.viewport(0,0,surface.width,surface.height);gl.clearColor(.105,.13,.125,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.uniform2f(gl.getUniformLocation(program,'angle'),yaw,pitch);const aspect=surface.width/surface.height, factor=Math.min(.55, .8*w/h);gl.uniform2f(gl.getUniformLocation(program,'scale'),factor/Math.max(aspect,1),factor*Math.min(aspect,1));gl.drawElements(gl.TRIANGLES,indices.length,gl.UNSIGNED_SHORT,0);}
  draw(); return {draw, rotate(dx,dy){yaw+=dx*.009;pitch=Math.max(-1.1,Math.min(1.1,pitch+dy*.009));draw();}, reset(){yaw=-.28;pitch=.06;draw();}};
}
let last=null;surface.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];surface.setPointerCapture(e.pointerId);});
surface.addEventListener('pointermove',e=>{if(last&&renderer){renderer.rotate(e.clientX-last[0],e.clientY-last[1]);last=[e.clientX,e.clientY];}});
surface.addEventListener('pointerup',()=>{last=null;});surface.addEventListener('pointercancel',()=>{last=null;});
document.querySelector('#reset').addEventListener('click',()=>renderer?.reset());window.addEventListener('resize',()=>renderer?.draw());
