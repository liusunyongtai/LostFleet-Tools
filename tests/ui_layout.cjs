/* UI regressions: real Chromium, two desktop sizes, and a path-prefixed proxy. */
const {chromium}=require('playwright');
const {spawn,spawnSync}=require('node:child_process');
const {mkdtempSync,readFileSync,writeFileSync,rmSync,existsSync}=require('node:fs');
const {tmpdir}=require('node:os');
const path=require('node:path');
const http=require('node:http');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),python=process.env.PYTHON||'python3';
const temp=mkdtempSync(path.join(tmpdir(),'lostfleet-ui-'));
let server,proxy,browser,page;
const checks=[];
const pass=name=>{checks.push(name);console.log('PASS:',name);};
async function idle(){await page.waitForFunction(()=>!document.getElementById('upload').disabled);}
async function point(x,y){const b=await page.locator('#canvas').boundingBox();return [b.x+x*b.width/512,b.y+y*b.height/640];}
async function clickPoint(x,y){await page.mouse.click(...await point(x,y));}
async function rectangle(x0,y0,x1,y1){await page.mouse.move(...await point(x0,y0));await page.mouse.down();await page.mouse.move(...await point(x1,y1),{steps:5});await page.mouse.up();}
async function editorStatus(){return page.locator('#status').textContent();}
async function preview(){await page.click('#preview');await idle();await page.locator('#previewImage').evaluate(im=>im.decode());assert.equal(await page.locator('#previewImage').isVisible(),true);}
async function canvasAlpha(x,y){return page.locator('#canvas').evaluate((canvas,[x,y])=>canvas.getContext('2d').getImageData(x,y,1,1).data[3],[x,y]);}
(async()=>{
  const fixture=spawnSync(python,['-c',`from PIL import Image,ImageDraw
import sys
im=Image.new('RGB',(512,640),(247,249,250)); d=ImageDraw.Draw(im)
d.rounded_rectangle((144,210,368,389),20,fill='#283d4b',outline='#6ea38e',width=5)
d.rectangle((229,176,283,220),fill='#9daea5')
d.rounded_rectangle((176,60,336,190),34,fill='#cbd7cb',outline='#83a994',width=5)
d.rounded_rectangle((192,96,320,151),14,fill='#1c353e');d.line((208,112,300,112),fill='#83e4ba',width=6)
d.polygon([(133,218),(93,233),(70,345),(114,356),(153,268)],fill='#738f83')
d.polygon([(378,218),(418,233),(442,345),(399,356),(360,268)],fill='#738f83')
d.rounded_rectangle((68,345,110,453),14,fill='#314b55');d.rounded_rectangle((402,345,444,453),14,fill='#314b55')
d.ellipse((67,440,112,486),fill='#b5c7b8');d.ellipse((400,440,445,486),fill='#b5c7b8')
d.rectangle((179,384,235,480),fill='#819c8d');d.rectangle((278,384,334,480),fill='#819c8d')
d.rounded_rectangle((179,475,235,581),12,fill='#364f59');d.rounded_rectangle((278,475,334,581),12,fill='#364f59')
d.rounded_rectangle((157,567,236,609),10,fill='#bcccbf');d.rounded_rectangle((277,567,357,609),10,fill='#bcccbf')
d.rounded_rectangle((207,239,306,315),8,fill='#172f38',outline='#6c9986',width=3);d.rectangle((224,256,242,275),fill='#86e1b6')
d.line((222,295,291,295),fill='#83a994',width=4);d.rectangle((156,365,356,384),fill='#172f38')
im.save(sys.argv[1])`,path.join(temp,'character.png')],{encoding:'utf8'});
  assert.equal(fixture.status,0,fixture.stderr);
  let backendPort;
  if(process.env.LOSTFLEET_URL){backendPort=Number(new URL(process.env.LOSTFLEET_URL).port);}
  else {
  server=spawn(python,['-m','lostfleet_tools.server','--port','0'],{cwd:root,stdio:['ignore','pipe','pipe']});
  backendPort=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('startup timeout')),10000);server.once('error',reject);server.once('exit',code=>reject(Error('server exit '+code)));server.stdout.on('data',data=>{const m=data.toString().match(/127\.0\.0\.1:(\d+)/);if(m){clearTimeout(timer);resolve(Number(m[1]));}});});
  }
  const prefix='/preview/lostfleet/';
  proxy=http.createServer((req,res)=>{
    if(req.url!==prefix.slice(0,-1)&&!req.url.startsWith(prefix)){res.writeHead(404);res.end('Unexpected root resource');return;}
    const upstream=http.request({hostname:'127.0.0.1',port:backendPort,path:req.url===prefix.slice(0,-1)?'/':'/'+req.url.slice(prefix.length),method:req.method,headers:req.headers},response=>{res.writeHead(response.statusCode,response.headers);response.pipe(res);});
    upstream.on('error',()=>{res.writeHead(502);res.end();});req.pipe(upstream);
  });
  await new Promise(resolve=>proxy.listen(0,'127.0.0.1',resolve));
  const executablePath=process.env.CHROMIUM_PATH||(existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined);
  browser=await chromium.launch({headless:true,executablePath});
  const liveURL=process.env.LOSTFLEET_URL||`http://127.0.0.1:${backendPort}/`;
  const targets=[{width:1366,height:768,url:liveURL,prefix:'/'},{width:1920,height:1080,url:liveURL,prefix:'/'},
    {width:1366,height:768,url:`http://127.0.0.1:${proxy.address().port}${prefix.slice(0,-1)}`,prefix,proxy:true}];
  for(const target of targets){
    page=await browser.newPage({viewport:{width:target.width,height:target.height}});page.setDefaultTimeout(12000);
    const errors=[],resources=[],consoleErrors=[],failedRequests=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',message=>{if(message.type()==='error')consoleErrors.push(message.text());});page.on('requestfailed',request=>failedRequests.push({url:request.url(),error:request.failure()?.errorText}));page.on('response',r=>resources.push({url:r.url(),status:r.status(),type:r.headers()['content-type']}));
    const documentResponse=await page.goto(target.url);
    const layout=await page.evaluate(()=>{
      const rect=selector=>{const r=document.querySelector(selector).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
      return {display:getComputedStyle(document.querySelector('main')).display,left:rect('.parts-panel'),center:rect('.workspace'),right:rect('.properties-panel'),canvas:rect('#viewport'),export:rect('#export'),width:innerWidth,height:innerHeight,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight};
    });
    assert.equal(layout.display,'grid');assert.equal(layout.left.x,0);assert.equal(layout.left.right,layout.center.x);assert.equal(layout.center.right,layout.right.x);
    assert.equal(layout.right.right,target.width);assert.ok(layout.canvas.height>target.height*.6);assert.ok(layout.export.bottom<target.height);
    assert.ok(layout.scrollWidth<=target.width);assert.ok(layout.scrollHeight<=target.height);
    assert.equal(documentResponse.headers()['x-lostfleet-document'],'self-contained');
    assert.equal(await page.locator('#editor-styles').count(),1);assert.equal(await page.locator('#editor-script').count(),1);
    assert.equal(resources.filter(r=>/style\.css|app\.js/.test(r.url)).length,0);
    const staticBase=target.url.endsWith('/')?target.url:target.url+'/';
    const css=await page.request.get(new URL('style.css?v=3',staticBase).toString());assert.equal(css.status(),200);assert.match(css.headers()['content-type'],/text\/css/);
    const js=await page.request.get(new URL('app.js?v=3',staticBase).toString());assert.equal(js.status(),200);assert.match(js.headers()['content-type'],/text\/javascript/);
    await page.click('#diagnostics');
    const diagnostics=JSON.parse(await page.locator('#diagnosticsReport').textContent());
    assert.equal(diagnostics.build,documentResponse.headers()['x-lostfleet-build']);assert.equal(diagnostics.layout.display,'grid');assert.deepEqual(diagnostics.cspViolations,[]);
    await page.click('#closeDiagnostics');
    assert.equal(await page.locator('.part-row').count(),16);
    await page.screenshot({path:path.join(tmpdir(),`lostfleet-empty-${target.width}${target.proxy?'-proxy':''}.png`)});
    pass(`${target.width}×${target.height}: CSS MIME / three columns / no page overflow / visible export`);
    await page.locator('#upload').setInputFiles(path.join(temp,'character.png'));await idle();
    assert.match(await editorStatus(),/^图片已载入/);
    const bounds=await page.locator('#canvas').boundingBox();assert.ok(bounds.height<layout.canvas.height);assert.ok(bounds.y>=layout.canvas.y);
    await page.click('[data-tool=rectangle]');await rectangle(163,50,348,203);
    assert.equal(await page.locator('[data-part=head] .region-count').textContent(),'1');
    await page.fill('#displayName','头部 · 试作');await page.locator('#displayName').blur();
    assert.equal(await page.locator('[data-part=head] .part-name').textContent(),'头部 · 试作');
    const visibleImage=await page.locator('#canvas').evaluate(c=>c.toDataURL());
    await page.click('[data-part=head] .part-visibility');
    assert.equal(await page.locator('[data-part=head] .part-visibility').getAttribute('aria-pressed'),'false');
    assert.notEqual(await page.locator('#canvas').evaluate(c=>c.toDataURL()),visibleImage);
    await page.click('[data-part=head] .part-visibility');
    pass(`${target.width}: part selection / rename / overlay visibility`);
    // Pan without altering selections, then zoom around a fixed image location.
    await page.click('[data-tool=pan]');const before=await page.locator('#canvas').boundingBox();
    await page.mouse.move(...await point(256,320));await page.mouse.down();await page.mouse.move(...[...await point(256,320)].map((v,i)=>v+(i===0?43:-17)));await page.mouse.up();
    const after=await page.locator('#canvas').boundingBox();assert.ok(Math.abs(after.x-before.x-43)<1);assert.ok(Math.abs(after.y-before.y+17)<1);
    await page.click('[data-tool=rectangle]');await page.locator('#viewport').focus();await page.keyboard.down('Space');
    await page.mouse.move(...await point(256,320));await page.mouse.down();const mid=await point(256,320);await page.mouse.move(mid[0]-20,mid[1]+10);await page.mouse.up();await page.keyboard.up('Space');
    assert.equal(await page.locator('[data-part=head] .region-count').textContent(),'1');
    const anchor=await point(256,320);await page.mouse.move(...anchor);const preZoom=await page.locator('#canvas').boundingBox();
    await page.mouse.wheel(0,-120);await page.waitForFunction(w=>document.getElementById('canvas').getBoundingClientRect().width>w,preZoom.width);
    const zoomedAnchor=await point(256,320);assert.ok(Math.abs(anchor[0]-zoomedAnchor[0])<1);assert.ok(Math.abs(anchor[1]-zoomedAnchor[1])<1);
    await page.click('#resetView');
    pass(`${target.width}: centered fit / hand pan / Space drag / cursor-anchored wheel zoom`);
    await page.click('[data-tool=pivot]');await page.mouse.move(...await point(256,170));await page.mouse.down();await page.mouse.move(...await point(260,174),{steps:3});await page.mouse.up();
    assert.match(await page.locator('#pivotValue').textContent(),/260\.0, 174\.0/);
    await page.click('[data-tool=eraser]');await rectangle(240,80,250,90);
    assert.equal(await page.locator('[data-part=head] .region-count').textContent(),'2');
    await page.fill('#edge','1');await page.fill('#feather','1');await page.fill('#overlap','4');await preview();
    await page.selectOption('#viewMode','mask');assert.equal(await canvasAlpha(10,10),0);
    await page.click('#compare');assert.equal(await canvasAlpha(10,10),255);assert.equal(await page.locator('#compare').getAttribute('aria-pressed'),'true');
    await page.click('#compare');assert.equal(await canvasAlpha(10,10),0);
    assert.notEqual(await page.locator('#viewport').evaluate(e=>getComputedStyle(e).backgroundImage),'none');
    await page.click('#checker');assert.equal(await page.locator('#viewport').evaluate(e=>getComputedStyle(e).backgroundImage),'none');await page.click('#checker');
    await page.fill('#feather','2');assert.equal(await page.locator('#previewImage').isVisible(),false);await preview();
    await page.selectOption('#viewMode','edit');
    await page.click('[data-part=body] .part-select');await page.click('[data-tool=polygon]');
    await clickPoint(145,210);await clickPoint(369,210);await clickPoint(360,385);await clickPoint(152,385);await page.click('#finish');
    assert.equal(await page.locator('[data-part=body] .region-count').textContent(),'1');
    pass(`${target.width}: pivot drag / erase / polygon / PNG preview / original comparison / checker toggle / stale-preview invalidation`);
    await page.click('[data-part=head] .part-select');await page.click('[data-part=head] .part-visibility');
    const download=page.waitForEvent('download');await page.click('#save');await(await download).saveAs(path.join(temp,'project.json'));
    const project=JSON.parse(readFileSync(path.join(temp,'project.json'),'utf8'));assert.equal(project.schema_version,1);assert.equal(project.ui.part_labels.head,'头部 · 试作');assert.deepEqual(project.ui.hidden_parts,['head']);
    assert.equal(project.parts.find(p=>p.name==='head').name,'head');
    await page.click('#clear');await page.locator('#load').setInputFiles(path.join(temp,'project.json'));await idle();assert.match(await editorStatus(),/^工程已载入/);
    assert.equal(await page.inputValue('#displayName'),'头部 · 试作');assert.equal(await page.locator('[data-part=head] .part-visibility').getAttribute('aria-pressed'),'false');
    const exportDownload=page.waitForEvent('download');await page.click('#export');await(await exportDownload).saveAs(path.join(temp,'assets.zip'));await idle();assert.match(await editorStatus(),/2 个部件/);
    delete project.ui;writeFileSync(path.join(temp,'legacy.json'),JSON.stringify(project));
    await page.locator('#load').setInputFiles(path.join(temp,'legacy.json'));await idle();assert.match(await editorStatus(),/^工程已载入/);assert.equal(await page.inputValue('#displayName'),'头部');
    await preview();await page.screenshot({path:path.join(tmpdir(),`lostfleet-${process.env.LOSTFLEET_URL?'live-8000':'ui'}-${target.width}${target.proxy?'-proxy':''}.png`)});
    assert.deepEqual(errors,[]);assert.deepEqual(consoleErrors,[]);assert.deepEqual(failedRequests,[]);assert.ok(resources.filter(r=>new URL(r.url).pathname.includes('/api/')).every(r=>r.status===200&&new URL(r.url).pathname.startsWith(target.prefix)));
    assert.ok(resources.every(r=>r.status===200));
    pass(`${target.width}: UI state persistence / legacy project compatibility / unchanged export IDs / ZIP / no JS or HTTP errors`);
    const report={scope:process.env.LOSTFLEET_URL?'Running port 8000 via local Chromium; NOT Codex embedded preview':'Isolated test server',proxy:!!target.proxy,
      document:{status:documentResponse.status(),type:documentResponse.headers()['content-type'],cache:documentResponse.headers()['cache-control'],csp:documentResponse.headers()['content-security-policy']||null,build:documentResponse.headers()['x-lostfleet-build']},
      staticEndpoints:{css:{status:css.status(),type:css.headers()['content-type']},js:{status:js.status(),type:js.headers()['content-type']}},
      diagnostics:await page.evaluate(()=>pageDiagnostics()),consoleErrors,failedRequests,pageErrors:errors,
      requests:resources.map(r=>({path:new URL(r.url).pathname,status:r.status,type:r.type}))};
    writeFileSync(path.join(tmpdir(),`lostfleet-${process.env.LOSTFLEET_URL?'live-8000':'ui'}-${target.width}${target.proxy?'-proxy':''}-report.json`),JSON.stringify(report,null,2));
    await page.close();page=null;
  }
  console.log(`${checks.length} UI checks passed; tested upstream: ${liveURL}; screenshots in ${tmpdir()}`);
})().catch(async error=>{console.error(error);if(page){console.error('Status:',await editorStatus());await page.screenshot({path:path.join(tmpdir(),'lostfleet-ui-failure.png')});}process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();if(proxy)await new Promise(resolve=>proxy.close(resolve));if(server)server.kill();rmSync(temp,{recursive:true,force:true});});
