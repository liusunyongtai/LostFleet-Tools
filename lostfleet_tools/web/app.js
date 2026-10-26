'use strict';
const $ = id => document.getElementById(id);
const names = {
  head:'头部', body:'身体', left_upper_arm:'左上臂', right_upper_arm:'右上臂',
  left_forearm:'左下臂', right_forearm:'右下臂', left_hand:'左手', right_hand:'右手',
  left_thigh:'左大腿', right_thigh:'右大腿', left_shin:'左小腿', right_shin:'右小腿',
  left_foot:'左脚', right_foot:'右脚', weapon:'武器', backpack:'背包'
};
const symbols = {head:'◇', body:'▣', weapon:'⚔', backpack:'▤'};
const toolLabels = {rectangle:'矩形框选', polygon:'多边形选区', eraser:'擦除框', pivot:'关节枢轴点', pan:'平移画布'};
const freshPart = name => ({name, regions:[], pivot:null, edge:0, feather:0, overlap:0});
const freshParts = () => Object.fromEntries(Object.keys(names).map(name => [name, freshPart(name)]));
let parts = freshParts(), labels = {...names}, hiddenParts = new Set();
let source = null, imageId = null, sourceHash = null, pending = [], drag = null, previewUrl = null;
let processedSource = null, previewBounds = null, showOriginal = false, working = false, spacePan = false, panDrag = null, pivotDrag = false;
const view = {scale:1, x:0, y:0};
const canvas = $('canvas'), ctx = canvas.getContext('2d'), viewport = $('viewport');
for (const [name,label] of Object.entries(names)) {
  const option = document.createElement('option'); option.value=name; option.textContent=label;
  $('part').append(option);
}
$('part').value='head';
const current = () => parts[$('part').value];
const appBase = new URL(document.baseURI);
// Preview entrypoints may represent a directory without a trailing slash.
if(!appBase.pathname.endsWith('/')&&!appBase.pathname.endsWith('.html'))appBase.pathname+='/';
const apiURL = path => new URL(path.replace(/^\//, ''), appBase).toString();
const status = message => { $('status').textContent=message; $('status').parentElement.classList.toggle('error',message.startsWith('错误：')); };
function background() { return {mode:$('background').value, threshold:Number($('threshold').value)}; }
function invalidate() {
  $('previewImage').hidden=true; $('previewEmpty').hidden=false; processedSource=null;previewBounds=null;
  if(previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl=null; }
  draw();
}
function refreshLabels() {
  let ready=0;
  for (const option of $('part').options) {
    const count=parts[option.value].regions.filter(r=>r.operation==='add').length;
    option.textContent=labels[option.value]+(count?' ●':''); if(count)ready++;
  }
  $('partCount').textContent=ready+' / 16'; $('exportCount').textContent=ready+' 个部件';
  const list=$('partList'); list.replaceChildren();
  for(const name of Object.keys(names)) {
    const item=document.createElement('li'); item.className='part-row'; item.dataset.part=name;
    item.classList.toggle('selected',name===$('part').value); item.classList.toggle('hidden-layer',hiddenParts.has(name));
    const visibility=document.createElement('button'); visibility.className='part-visibility';
    visibility.textContent=hiddenParts.has(name)?'◌':'◉'; visibility.title=hiddenParts.has(name)?'显示编辑选区':'隐藏编辑选区';
    visibility.setAttribute('aria-label',visibility.title+'：'+labels[name]); visibility.setAttribute('aria-pressed',String(!hiddenParts.has(name)));
    visibility.onclick=()=>{if(working)return;hiddenParts.has(name)?hiddenParts.delete(name):hiddenParts.add(name);refreshLabels();draw();};
    const select=document.createElement('button'); select.className='part-select'; select.title=labels[name]+' / '+name;
    select.setAttribute('aria-pressed',String(name===$('part').value));
    const symbol=document.createElement('span');symbol.className='part-symbol';symbol.textContent=symbols[name]||'╱';
    const label=document.createElement('span');label.className='part-name';label.textContent=labels[name];
    select.append(symbol,label);select.onclick=()=>{if(working)return;$('part').value=name;syncControls();};
    const count=document.createElement('span');count.className='region-count';
    const n=parts[name].regions.length;count.textContent=n||'—';count.classList.toggle('has-regions',n>0);
    item.append(visibility,select,count);list.append(item);
  }
  updateMetrics();
}
function updateMetrics() {
  $('regionCount').textContent=current().regions.length;
  const pivot=pivotOf(current());
  $('pivotValue').textContent=pivot?pivot.map(v=>v.toFixed(1)).join(', '):'自动居中';
}
function syncControls() {
  for(const key of ['edge','feather','overlap']) $(key).value=current()[key];
  $('displayName').value=labels[current().name];$('selectedId').textContent=current().name;
  pending=[];drag=null;pivotDrag=false;invalidate();refreshLabels();
}
function applyView() {
  if(!source)return;
  canvas.style.width=source.width*view.scale+'px';canvas.style.height=source.height*view.scale+'px';
  canvas.style.transform=`translate(${view.x}px, ${view.y}px)`;draw();
}
function zoomLabel() {
  if($('zoom').value==='fit')return;
  let custom=$('zoom').querySelector('[data-custom]');
  const known=[...$('zoom').options].find(o=>o.value!=='fit'&&!o.dataset.custom&&Math.abs(Number(o.value)-view.scale)<0.0001);
  if(known){$('zoom').value=known.value;return;}
  if(!custom){custom=document.createElement('option');custom.dataset.custom='true';$('zoom').append(custom);}
  custom.value=String(view.scale);custom.textContent=Math.round(view.scale*100)+'%';$('zoom').value=custom.value;
}
function setZoom(scale, anchor=null) {
  if(!source)return;
  const x=anchor?anchor[0]:viewport.clientWidth/2,y=anchor?anchor[1]:viewport.clientHeight/2;
  const factor=Math.max(.05,Math.min(8,scale))/view.scale;
  view.x=x-(x-view.x)*factor;view.y=y-(y-view.y)*factor;view.scale*=factor;
  if($('zoom').value==='fit')$('zoom').value='1';zoomLabel();applyView();
}
function resize() {
  if(!source)return;
  if($('zoom').value==='fit') {
    view.scale=Math.max(.01,Math.min(4,(viewport.clientWidth-64)/source.width,(viewport.clientHeight-64)/source.height));
    view.x=(viewport.clientWidth-source.width*view.scale)/2;view.y=(viewport.clientHeight-source.height*view.scale)/2;
    applyView();
  } else setZoom(Number($('zoom').value));
}
function fitView() { $('zoom').value='fit';resize(); }
function path(points, color, fill=true) {
  if(!points.length)return;
  const scale=view.scale;
  ctx.beginPath();ctx.moveTo(...points[0]);for(const p of points.slice(1))ctx.lineTo(...p);
  if(fill)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=1.5/scale;
  if(fill){ctx.fillStyle=color+'22';ctx.fill();}ctx.stroke();
  for(const p of points){ctx.fillStyle=color;ctx.fillRect(p[0]-2/scale,p[1]-2/scale,4/scale,4/scale);}
}
function pivotOf(part) {
  if(part.pivot)return part.pivot;
  const points=part.regions.filter(r=>r.operation==='add').flatMap(r=>r.points);
  if(!points.length)return null;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  return [(Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2];
}
function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);if(!source)return;
  const maskView=$('viewMode').value==='mask'&&processedSource&&!showOriginal;
  ctx.drawImage(maskView?processedSource:source,0,0);
  if(showOriginal||maskView)return;
  for(const part of Object.values(parts)) {
    if(hiddenParts.has(part.name))continue;
    for(const region of part.regions)path(region.points,part===current()?(region.operation==='subtract'?'#ff8c87':'#83e4ba'):'#8094ac',true);
  }
  path(pending,'#f6ce82',false);if(drag)path(rectangle(drag.start,drag.end),drag.operation==='subtract'?'#ff8c87':'#f6ce82');
  const pivot=pivotOf(current());
  if(pivot&&!hiddenParts.has(current().name)) {
    const r=9/view.scale;ctx.strokeStyle='#f6ce82';ctx.lineWidth=1.5/view.scale;ctx.beginPath();
    ctx.moveTo(pivot[0]-r,pivot[1]);ctx.lineTo(pivot[0]+r,pivot[1]);ctx.moveTo(pivot[0],pivot[1]-r);ctx.lineTo(pivot[0],pivot[1]+r);ctx.stroke();
    ctx.beginPath();ctx.arc(...pivot,4/view.scale,0,2*Math.PI);ctx.stroke();
    if(current().overlap){ctx.beginPath();ctx.arc(...pivot,current().overlap,0,2*Math.PI);ctx.stroke();}
  }
}
function position(e) {
  const r=canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(canvas.width-1,(e.clientX-r.left)/view.scale)),Math.max(0,Math.min(canvas.height-1,(e.clientY-r.top)/view.scale))];
}
function rectangle(a,b){const x0=Math.min(a[0],b[0]),y0=Math.min(a[1],b[1]),x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]);return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]];}
function addRegion(points,operation=$('operation').value) {
  current().regions.push({operation,points});hiddenParts.delete(current().name);invalidate();refreshLabels();
  status('选区已保存。可继续绘制、调整枢轴点，或更新透明 PNG 预览。');
}
function editingView() {
  if(working||!source||showOriginal)return false;
  if($('viewMode').value==='mask'){$('viewMode').value='edit';draw();}
  return true;
}
canvas.addEventListener('pointerdown',e=>{
  if(e.button!==0||!editingView())return;
  viewport.focus({preventScroll:true});const p=position(e);
  if($('tool').value==='pivot'){
    current().pivot=p;pivotDrag=true;hiddenParts.delete(current().name);canvas.setPointerCapture(e.pointerId);invalidate();updateMetrics();
    status('枢轴点：'+p.map(v=>v.toFixed(1)).join(', ')+' · 可拖动调整');
  } else if($('tool').value==='polygon'){pending.push(p);draw();}
  else if($('tool').value!=='pan'){
    drag={start:p,end:p,operation:$('tool').value==='eraser'?'subtract':$('operation').value};canvas.setPointerCapture(e.pointerId);draw();
  }
});
canvas.addEventListener('pointermove',e=>{
  if(!source)return;const p=position(e);$('coordinates').textContent='X '+p[0].toFixed(1)+'   Y '+p[1].toFixed(1);
  if(drag){drag.end=p;draw();}if(pivotDrag){current().pivot=p;invalidate();updateMetrics();}
});
canvas.addEventListener('pointerup',e=>{
  if(pivotDrag){pivotDrag=false;updateMetrics();}
  if(drag){const d=drag;drag=null;d.end=position(e);if(Math.abs(d.start[0]-d.end[0])>=1&&Math.abs(d.start[1]-d.end[1])>=1)addRegion(rectangle(d.start,d.end),d.operation);else draw();}
});
canvas.addEventListener('pointercancel',()=>{drag=null;pivotDrag=false;draw();});
viewport.addEventListener('pointerdown',e=>{
  if(working||!source)return;
  if(e.button===1||spacePan||$('tool').value==='pan'){
    e.preventDefault();e.stopPropagation();viewport.focus({preventScroll:true});
    panDrag={x:e.clientX,y:e.clientY,originX:view.x,originY:view.y};viewport.setPointerCapture(e.pointerId);viewport.classList.add('panning');
  }
},true);
viewport.addEventListener('pointermove',e=>{if(panDrag){view.x=panDrag.originX+e.clientX-panDrag.x;view.y=panDrag.originY+e.clientY-panDrag.y;applyView();}});
function stopPan(){panDrag=null;viewport.classList.remove('panning');}
viewport.addEventListener('pointerup',stopPan);viewport.addEventListener('pointercancel',stopPan);
viewport.addEventListener('wheel',e=>{if(!source||working)return;e.preventDefault();const r=viewport.getBoundingClientRect();setZoom(view.scale*Math.exp(-e.deltaY*.0015),[e.clientX-r.left,e.clientY-r.top]);},{passive:false});
function finish(){if(working)return;if(pending.length<3){status('多边形至少需要 3 个点。');return;}const points=pending;pending=[];addRegion(points);}
function syncTool() {
  pending=[];drag=null;pivotDrag=false;draw();viewport.classList.toggle('pan-tool',$('tool').value==='pan');
  for(const button of document.querySelectorAll('[data-tool]')){const active=button.dataset.tool===$('tool').value;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}
  $('viewBadge').textContent=toolLabels[$('tool').value]+($('tool').value==='polygon'?' · Enter 完成':'');
}
for(const button of document.querySelectorAll('[data-tool]'))button.onclick=()=>{$('tool').value=button.dataset.tool;syncTool();};
$('finish').onclick=finish;
$('undo').onclick=()=>{if(pending.length)pending.pop();else current().regions.pop();invalidate();refreshLabels();};
$('clear').onclick=()=>{parts[$('part').value]=freshPart($('part').value);syncControls();};
$('part').onchange=syncControls;$('tool').onchange=syncTool;
$('operation').onchange=()=>{pending=[];draw();};
$('zoom').onchange=resize;$('zoomIn').onclick=()=>setZoom(view.scale*1.25);$('zoomOut').onclick=()=>setZoom(view.scale/1.25);$('resetView').onclick=fitView;
window.addEventListener('resize',resize);
$('compare').onclick=()=>{showOriginal=!showOriginal;$('compare').setAttribute('aria-pressed',String(showOriginal));$('compareBadge').hidden=!showOriginal;viewport.classList.toggle('comparing',showOriginal);draw();};
$('checker').onclick=()=>{const enabled=viewport.classList.toggle('checkerboard');$('checker').setAttribute('aria-pressed',String(enabled));};
$('viewMode').onchange=()=>{if($('viewMode').value==='mask'&&!processedSource)status('分割预览尚未更新，请点击右侧“更新预览”；当前显示原图选区。');draw();};
$('displayName').oninput=()=>{labels[current().name]=$('displayName').value.trim()||names[current().name];refreshLabels();};
$('displayName').onblur=()=>{$('displayName').value=labels[current().name];};
const textInput = () => ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)||document.activeElement.isContentEditable;
document.addEventListener('keydown',e=>{
  if(textInput()||working)return;
  if(e.code==='Space'){e.preventDefault();spacePan=true;viewport.classList.add('space-pan');}
  const shortcuts={r:'rectangle',p:'polygon',e:'eraser',j:'pivot',h:'pan'};
  if(!e.ctrlKey&&!e.metaKey&&shortcuts[e.key.toLowerCase()]){$('tool').value=shortcuts[e.key.toLowerCase()];syncTool();}
  if(e.key==='Enter'&&document.activeElement.tagName!=='BUTTON'){e.preventDefault();finish();}
  if(e.key==='Escape'){pending=[];drag=null;pivotDrag=false;stopPan();draw();}
});
document.addEventListener('keyup',e=>{if(e.code==='Space'){spacePan=false;viewport.classList.remove('space-pan');}});
window.addEventListener('blur',()=>{spacePan=false;stopPan();viewport.classList.remove('space-pan');drag=null;pivotDrag=false;draw();});
for(const key of ['edge','feather','overlap']) {
  $(key).oninput=()=>{
    const input=$(key),value=Number(input.value);
    if(!Number.isFinite(value)||value<Number(input.min)||value>Number(input.max)){
      status('错误：参数应在 '+input.min+'～'+input.max+' px 之间');return;
    }
    current()[key]=value;invalidate();
  };
  $(key).onblur=()=>{$(key).value=current()[key];};
}
for(const key of ['background','threshold']) $(key).onchange=invalidate;
for(const label of document.querySelectorAll('.button[tabindex]'))label.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();const input=label.querySelector('input')||$(label.htmlFor);if(input&&!input.disabled)input.click();}});

function fitPreview() {
  if(!previewBounds||!processedSource)return;
  const image=$('previewImage'),box=image.parentElement;
  const [left,top,right,bottom]=previewBounds;
  const scale=Math.min((box.clientWidth-24)/(right-left),(box.clientHeight-24)/(bottom-top));
  image.style.width=image.naturalWidth*scale+'px';image.style.height=image.naturalHeight*scale+'px';
  image.style.left=((box.clientWidth-(right-left)*scale)/2-left*scale)+'px';
  image.style.top=((box.clientHeight-(bottom-top)*scale)/2-top*scale)+'px';
}
function measurePreview(image) {
  const sample=document.createElement('canvas');sample.width=image.naturalWidth;sample.height=image.naturalHeight;
  const context=sample.getContext('2d',{willReadFrequently:true});context.drawImage(image,0,0);
  const data=context.getImageData(0,0,sample.width,sample.height).data;
  let left=sample.width,top=sample.height,right=0,bottom=0;
  for(let y=0;y<sample.height;y++)for(let x=0;x<sample.width;x++)if(data[(y*sample.width+x)*4+3]){
    left=Math.min(left,x);top=Math.min(top,y);right=Math.max(right,x+1);bottom=Math.max(bottom,y+1);
  }
  previewBounds=[left,top,right,bottom];fitPreview();
}
new ResizeObserver(fitPreview).observe($('previewImage').parentElement);

async function request(url, body, raw=false){const response=await fetch(apiURL(url),{method:'POST',headers:raw?{'Content-Type':'application/octet-stream'}:{'Content-Type':'application/json'},body:raw?body:JSON.stringify(body)});if(!response.ok){let error;try{error=(await response.json()).error;}catch{error='服务器请求失败';}throw new Error(error);}return response;}
function busy(fn){return async()=>{if(working)return;working=true;const controls=[...document.querySelectorAll('button,input,select')];controls.forEach(c=>c.disabled=true);try{await fn();}catch(error){status('错误：'+error.message);}finally{working=false;controls.forEach(c=>c.disabled=false);document.querySelectorAll('#partList button').forEach(c=>c.disabled=false);}};}
async function fingerprint(file){
  // SHA-256 pins project coordinates to the exact source, not only its dimensions.
  const bytes=new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer()));
  return [...bytes].map(b=>b.toString(16).padStart(2,'0')).join('');
}
$('upload').onchange=busy(async()=>{
  const file=$('upload').files[0];if(!file)return;
  if(file.size>32*1024*1024)throw new Error('图片超过 32 MiB');
  const hash=await fingerprint(file);
  const info=await (await request('/api/upload',file,true)).json();
  const image=new Image();image.src=apiURL('/api/image/'+info.image_id);await image.decode();
  source=image;imageId=info.image_id;sourceHash=hash;
  parts=freshParts();labels={...names};hiddenParts=new Set();$('part').value='head';
  canvas.width=info.width;canvas.height=info.height;$('empty').hidden=true;$('empty').style.display='none';
  $('filename').textContent=file.name+' · '+info.width+' × '+info.height;
  syncControls();fitView();status('图片已载入。选择部件，拖动框选或用多边形沿轮廓绘制。新上传会清空旧选区。');
});
$('preview').onclick=busy(async()=>{
  if(!imageId)throw new Error('请先上传图片');
  const response=await request('/api/preview',{image_id:imageId,part:current(),background:background()});
  if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(await response.blob());
  $('previewImage').src=previewUrl;await $('previewImage').decode();processedSource=$('previewImage');$('previewImage').hidden=false;$('previewEmpty').hidden=true;measurePreview(processedSource);draw();status('预览已更新：棋盘格表示透明，请检查白边与关节重叠。');
});
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export').onclick=busy(async()=>{
  if(!imageId)throw new Error('请先上传图片');if(pending.length)throw new Error('请先完成或取消当前多边形');
  const selected=Object.values(parts).filter(p=>p.regions.some(r=>r.operation==='add'));
  const response=await request('/api/export',{image_id:imageId,parts:selected,background:background()});
  download(await response.blob(),'lostfleet-assets.zip');status('ZIP 已导出：'+selected.length+' 个部件，每个部件含统一画布与紧凑裁切图，以及 metadata.json。');
});
$('save').onclick=()=>{if(!source){status('请先上传图片');return;}if(pending.length){status('请先完成或取消当前多边形');return;}download(new Blob([JSON.stringify({schema_version:1,source_sha256:sourceHash,canvas_size:[source.width,source.height],parts:Object.values(parts),background:background(),ui:{part_labels:labels,hidden_parts:[...hiddenParts]}},null,2)],{type:'application/json'}),'lostfleet-project.json');status('工程已保存；请同时保留原始图片。');};
$('load').onchange=busy(async()=>{
  const file=$('load').files[0];if(!file)return;if(!source)throw new Error('请先上传原始图片');
  if(file.size>2*1024*1024)throw new Error('工程文件超过 2 MiB');
  const project=JSON.parse(await file.text());
  if(project.schema_version!==1||project.source_sha256!==sourceHash||!Array.isArray(project.canvas_size)||project.canvas_size[0]!==source.width||project.canvas_size[1]!==source.height)throw new Error('工程版本或原始图片不匹配，请上传保存工程时的同一图片');
  if(!Array.isArray(project.parts)||project.parts.length!==Object.keys(names).length)throw new Error('部件列表格式错误');
  const incoming={};
  for(const part of project.parts){
    if(!Object.hasOwn(names,part.name)||Object.hasOwn(incoming,part.name)||!Array.isArray(part.regions)||part.regions.length>100)throw new Error('工程部件格式错误');
    for(const region of part.regions){if(!['add','subtract'].includes(region.operation)||!Array.isArray(region.points)||region.points.length<3||region.points.length>1000||!region.points.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<source.width&&p[1]>=0&&p[1]<source.height))throw new Error('工程选区坐标无效');}
    if(part.pivot!==null&&(!Array.isArray(part.pivot)||part.pivot.length!==2||!part.pivot.every(Number.isFinite)||part.pivot[0]<0||part.pivot[0]>=source.width||part.pivot[1]<0||part.pivot[1]>=source.height))throw new Error('枢轴点无效');
    for(const [key,min,max] of [['edge',-32,32],['feather',0,16],['overlap',0,128]])if(!Number.isFinite(part[key])||part[key]<min||part[key]>max)throw new Error('边缘参数无效');
    incoming[part.name]=part;
  }
  const bg=project.background;if(!bg||!['border_white','all_white','keep'].includes(bg.mode)||!Number.isFinite(bg.threshold)||bg.threshold<0||bg.threshold>255)throw new Error('背景参数无效');
  const nextLabels={...names},nextHidden=new Set();
  if(project.ui){
    if(typeof project.ui!=='object'||!project.ui.part_labels||typeof project.ui.part_labels!=='object'||!Array.isArray(project.ui.hidden_parts))throw new Error('工程显示设置无效');
    for(const [name,label] of Object.entries(project.ui.part_labels)){if(!Object.hasOwn(names,name)||typeof label!=='string'||!label.trim()||label.length>40)throw new Error('部件显示名称无效');nextLabels[name]=label;}
    for(const name of project.ui.hidden_parts){if(!Object.hasOwn(names,name))throw new Error('部件显示状态无效');nextHidden.add(name);}
  }
  labels=nextLabels;hiddenParts=nextHidden;
  parts=incoming;$('background').value=bg.mode;$('threshold').value=bg.threshold;syncControls();status('工程已载入，可继续编辑。');
});

syncControls();syncTool();

const cspViolations=[];
document.addEventListener('securitypolicyviolation',event=>{
  cspViolations.push({directive:event.effectiveDirective,blocked:event.blockedURI==='inline'?'inline':'external resource'});
});
function pageDiagnostics() {
  const layout=document.querySelector('.app-layout');
  const box=selector=>{const element=document.querySelector(selector),rect=element.getBoundingClientRect();return {class:element.className,display:getComputedStyle(element).display,x:Math.round(rect.x),y:Math.round(rect.y),width:Math.round(rect.width),height:Math.round(rect.height)};};
  return {
    build:document.querySelector('meta[name=lostfleet-build]')?.content||'unbundled-source',
    page:location.origin+location.pathname,
    apiBase:appBase.origin+appBase.pathname,
    embedded:window!==window.top,
    assets:{inlineCSS:!!$('editor-styles'),inlineJS:!!$('editor-script'),styleSheets:document.styleSheets.length,cssRules:[...document.styleSheets].map(sheet=>{try{return sheet.cssRules.length;}catch{return 'unreadable';}})},
    viewport:{width:innerWidth,height:innerHeight},
    layout:{display:getComputedStyle(layout).display,columns:getComputedStyle(layout).gridTemplateColumns},
    panels:{left:box('.parts-panel'),center:box('.workspace'),right:box('.properties-panel')},
    partCount:document.querySelectorAll('.part-row').length,
    overflow:{horizontal:document.documentElement.scrollWidth>innerWidth,vertical:document.documentElement.scrollHeight>innerHeight},
    cspViolations
  };
}
let lastDiagnostics;
$('diagnostics').onclick=()=>{lastDiagnostics=pageDiagnostics();$('diagnosticsReport').textContent=JSON.stringify(lastDiagnostics,null,2);$('diagnosticsDialog').showModal();};
$('closeDiagnostics').onclick=()=>$('diagnosticsDialog').close();
$('saveDiagnostics').onclick=()=>download(new Blob([JSON.stringify(lastDiagnostics,null,2)],{type:'application/json'}),'lostfleet-diagnostics.json');
document.documentElement.dataset.editorReady='true';
