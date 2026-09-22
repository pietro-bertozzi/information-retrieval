// Optional end-to-end check using Node 22+ and a local Chromium browser.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const browserPath = process.env.BROWSER_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const dashboard = path.resolve(process.argv[2] || 'visualization/data/index.html');
const profile = await mkdtemp(path.join(tmpdir(), 'ir-dashboard-browser-'));
const browser = spawn(browserPath, ['--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], {windowsHide:true, stdio:'ignore'});
let launchError, socket, nextId=0;
browser.on('error', error => {launchError=error;});
const responses = new Map(), errors=[];
const delay = ms => new Promise(resolve=>setTimeout(resolve,ms));
async function until(callback, label) {
  const deadline=Date.now()+20000;
  while(Date.now()<deadline) {
    if(launchError) throw launchError;
    const result=await callback();
    if(result) return result;
    await delay(100);
  }
  throw new Error(`Timed out: ${label}`);
}
function send(method, params={}) {
  return new Promise((resolve,reject)=>{
    const id=++nextId;
    const timer=setTimeout(()=>{responses.delete(id);reject(new Error(`CDP timeout: ${method}`));},10000);
    responses.set(id,{resolve:result=>{clearTimeout(timer);resolve(result);},reject:error=>{clearTimeout(timer);reject(error);}});
    socket.send(JSON.stringify({id,method,params}));
  });
}
async function evaluate(expression) {
  const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function choose(id,value) {
  await evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(value)};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event('change'))`);
}
try {
  const port=await until(async()=>{
    try {return (await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];} catch {return null;}
  },'browser launch');
  const pages=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket=new WebSocket(pages.find(page=>page.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  socket.onmessage=event=>{
    const message=JSON.parse(event.data);
    if(message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    const pending=responses.get(message.id);
    if(pending) {responses.delete(message.id);if(message.error) pending.reject(new Error(message.error.message));else pending.resolve(message.result);}
  };
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(dashboard).href});
  await until(()=>evaluate(`document.getElementById('ranking')?.children.length===11`),'dashboard rendering');
  assert.equal(await evaluate(`document.querySelectorAll('#dataset option').length`),2);
  assert.equal(await evaluate(`document.getElementById('dataset').value`),'jurifindit');
  assert.equal(await evaluate(`document.querySelector('#cards strong').textContent`),'0.196051');
  assert.equal(await evaluate(`document.querySelector('#ranking tr td').textContent`),'1.4.1-binary-cosine');
  assert.equal(await evaluate(`cache.size`),0,'query payloads should load only on demand');
  await evaluate(`document.querySelector('[data-sort="name"]').click()`);
  assert.equal(await evaluate(`document.querySelector('#ranking tr td').textContent`),'1.1.1-term-overlap');
  await evaluate(`document.querySelector('[data-tab="depth"]').click()`);
  assert.equal(await evaluate(`document.querySelectorAll('#curve polyline').length`),11);
  await choose('metric','R@1000');
  assert.equal(await evaluate(`document.querySelector('#cards strong').textContent`),'0.704684');
  await choose('metric','Bpref');
  assert.match(await evaluate(`document.getElementById('curve').textContent`),/No saved cutoff/);
  await choose('metric','nDCG@10');
  await evaluate(`document.querySelector('[data-tab="queries"]').click()`);
  await until(()=>evaluate(`document.getElementById('query-status').textContent.includes('across 179 queries')`),'query loading from file URL');
  assert.equal(await evaluate(`document.querySelectorAll('#query-rows tr').length`),50);
  await evaluate(`document.getElementById('next').click()`);
  assert.equal(await evaluate(`document.getElementById('page').textContent`),'Page 2 of 4');
  await choose('candidate','1.1.2-term-overlap.trec');
  await until(()=>evaluate(`document.getElementById('query-status').textContent.includes('0 wins · 179 ties · 0 losses')`),'identical model query comparison');
  await choose('outcome','wins');
  assert.equal(await evaluate(`document.querySelectorAll('#query-rows tr').length`),0);
  await choose('outcome','all');
  await choose('dataset','scifact');
  await until(()=>evaluate(`document.getElementById('query-status').textContent.includes('across 300 queries')`),'dataset switch query comparison');
  await evaluate(`document.querySelector('[data-tab="details"]').click()`);
  assert.match(await evaluate(`document.getElementById('metadata').textContent`),/trec_eval/);
  await evaluate(`document.querySelector('[data-tab="overview"]').click()`);
  await evaluate(`for(const checkbox of document.querySelectorAll('#models input')) {checkbox.checked=false;checkbox.dispatchEvent(new Event('change'));}`);
  assert.equal(await evaluate(`document.querySelectorAll('#ranking tr').length`),0);
  await choose('dataset','jurifindit');
  const screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
  const screenshotPath=path.join(tmpdir(),'ir-dashboard-overview.png');
  await writeFile(screenshotPath,Buffer.from(screenshot.data,'base64'));
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate(`document.documentElement.scrollWidth <= innerWidth`),true,'mobile layout must not overflow horizontally');
  assert.deepEqual(errors,[],'no browser runtime exceptions');
  console.log('Browser checks passed: dataset/metric selection, rankings, curves, lazy query loading, pagination, ties, filters, empty selection, details, and mobile layout.');
  console.log(`Screenshot: ${screenshotPath}`);
} finally {
  if(socket?.readyState===WebSocket.OPEN) {
    try {await send('Browser.close');} catch {browser.kill();}
    socket.close();
  } else browser.kill();
  await delay(500);
  // Delete only the unique browser profile created inside the system temp folder.
  if(path.dirname(path.resolve(profile))===path.resolve(tmpdir()) && path.basename(profile).startsWith('ir-dashboard-browser-')) {
    await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:300});
  }
}