/* Optional real-browser smoke test. Requires Playwright + Chromium in the test environment. */
const { chromium } = require('playwright');
const { spawn, spawnSync } = require('node:child_process');
const { mkdtempSync, rmSync, readFileSync, existsSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const python = process.env.PYTHON || 'python3';
const temp = mkdtempSync(path.join(tmpdir(), 'lostfleet-browser-'));
let server, browser, page;
function pythonRun(code, args=[]) {
  const result=spawnSync(python,['-c',code,...args],{cwd:root,encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);return result.stdout;
}
(async()=>{
  pythonRun('from PIL import Image; import sys; from pathlib import Path; p=Path(sys.argv[1]); im=Image.new("RGB",(128,128),"white"); im.paste((180,20,30),(24,24,104,104)); im.save(p/"source.png"); im.save(p/"source.jpg")',[temp]);
  server=spawn(python,['-m','lostfleet_tools.server','--port','0'],{cwd:root,stdio:['ignore','pipe','pipe']});
  const port=await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('Server startup timed out')),10000);
    server.once('error',reject);server.once('exit',code=>reject(new Error('Server exited '+code)));
    server.stdout.on('data',chunk=>{const match=chunk.toString().match(/127\.0\.0\.1:(\d+)/);if(match){clearTimeout(timer);resolve(Number(match[1]));}});
  });
  const executablePath=process.env.CHROMIUM_PATH || (existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined);
  browser=await chromium.launch({headless:true,executablePath});
  page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.setDefaultTimeout(10000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}`);
  for(const format of ['jpg','png']){
    await page.locator('#upload').setInputFiles(path.join(temp,'source.'+format));
    await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('图片已载入'));
    await page.waitForFunction(()=>!document.getElementById('upload').disabled);
  }
  await page.selectOption('#zoom','2');
  const canvas=page.locator('#canvas');
  async function click(x,y){const b=await canvas.boundingBox();await page.mouse.click(b.x+x*b.width/128,b.y+y*b.height/128);}
  async function rect(x0,y0,x1,y1){const b=await canvas.boundingBox();await page.mouse.move(b.x+x0*b.width/128,b.y+y0*b.height/128);await page.mouse.down();await page.mouse.move(b.x+x1*b.width/128,b.y+y1*b.height/128,{steps:4});await page.mouse.up();}
  await rect(10,10,115,115);
  await page.selectOption('#tool','pivot');await click(60,50);
  await page.fill('#edge','1');await page.fill('#feather','1');await page.fill('#overlap','4');
  await page.selectOption('#tool','rectangle');await page.selectOption('#operation','subtract');await rect(55,55,65,65);
  await page.click('#preview');await page.waitForFunction(()=>!document.getElementById('previewImage').hidden);
  await page.waitForFunction(()=>!document.getElementById('preview').disabled);
  await page.locator('#previewImage').evaluate(im=>im.decode());
  assert.equal(await page.locator('#previewImage').evaluate(im=>im.naturalWidth),128);
  await page.selectOption('#part','body');await page.selectOption('#operation','add');await page.selectOption('#tool','polygon');
  assert.equal(await page.locator('#previewImage').isVisible(),false);
  await click(30,30);await click(90,30);await click(30,90);await page.click('#finish');
  assert.match(await page.locator('#part option[value=body]').textContent(),/●/);
  const savePromise=page.waitForEvent('download');await page.click('#save');const project=await savePromise;await project.saveAs(path.join(temp,'project.json'));
  const data=JSON.parse(readFileSync(path.join(temp,'project.json'),'utf8'));assert.equal(data.parts.find(p=>p.name==='head').regions.length,2);
  await page.selectOption('#part','head');await page.click('#clear');
  await page.locator('#load').setInputFiles(path.join(temp,'project.json'));
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('工程已载入'));
  await page.waitForFunction(()=>!document.getElementById('export').disabled);
  assert.equal(await page.inputValue('#overlap'),'4');
  const downloadPromise=page.waitForEvent('download');await page.click('#export');const zip=await downloadPromise;await zip.saveAs(path.join(temp,'assets.zip'));
  const result=pythonRun(`import sys,json; from zipfile import ZipFile; from io import BytesIO; from PIL import Image
with ZipFile(sys.argv[1]) as z:
 m=json.loads(z.read('metadata.json')); assert len(m['parts'])==2
 h=next(p for p in m['parts'] if p['name']=='head'); assert h['pivot']==[60,50]; assert h['adjustments']==dict(edge=1,feather=1.0,overlap=4)
 im=Image.open(BytesIO(z.read('full/head.png'))); assert im.mode=='RGBA'; assert im.size==(128,128); assert im.getpixel((60,60))[3]==0; assert im.getpixel((12,12))[3]==0; assert im.getpixel((40,40))[3]>0
 print('PASS: browser ZIP pixel/coordinate verification')`,[path.join(temp,'assets.zip')]);
  await page.screenshot({path:path.join(tmpdir(),'lostfleet-editor-smoke.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS: JPG/PNG upload, zoomed rectangle, polygon, subtract, pivot, adjustments, preview, project save/load, ZIP download, no JS errors');
  console.log(result.trim());
})().catch(async error=>{console.error(error);if(page){console.error('Editor status:',await page.locator('#status').textContent());await page.screenshot({path:path.join(tmpdir(),'lostfleet-editor-failure.png'),fullPage:true});}process.exitCode=1;}).finally(async()=>{
  if(browser)await browser.close();if(server)server.kill();rmSync(temp,{recursive:true,force:true});
});
