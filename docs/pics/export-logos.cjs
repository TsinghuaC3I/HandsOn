const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');

(async () => {
  const browser = await chromium.launch({executablePath:process.env.BROWSER_PATH || '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless:true});
  try {
    for (const file of fs.readdirSync(__dirname).filter(f=>/^handson-.*\.svg$/.test(f))) {
      const source = fs.readFileSync(path.join(__dirname,file),'utf8');
      const [, ,w,h] = source.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
      for (const scale of [1,2]) {
        const p = await browser.newPage({viewport:{width:Math.ceil(w),height:Math.ceil(h)},deviceScaleFactor:scale});
        await p.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block}</style>${source}`);
        await p.screenshot({path:path.join(__dirname,file.replace('.svg',scale===1?'.png':'@2x.png')),omitBackground:true});
        await p.close();
      }
    }
    const p = await browser.newPage({viewport:{width:16,height:16},deviceScaleFactor:1});
    for(const size of [16,32,48,64,128,256]) {
      await p.setViewportSize({width:size,height:size});
      const source = fs.readFileSync(path.join(__dirname,'handson-icon.svg'),'utf8');
      await p.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block;width:100vw;height:100vh}</style>${source}`);
      await p.screenshot({path:path.join(__dirname,`handson-icon-${size}.png`),omitBackground:true});
    }
    await p.close();
    console.log('Exported approved PNG artwork and icon sizes.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
