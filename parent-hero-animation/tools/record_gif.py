import time, os, glob
from playwright.sync_api import sync_playwright
SLOW=10; FPS=15
for f in glob.glob('frames/*.png'): os.remove(f)
INIT="""
(() => { const SLOW=%d; const pn=performance.now.bind(performance); const st=window.setTimeout.bind(window);
  performance.now = () => pn()/SLOW; window.setTimeout = (fn, ms, ...a) => st(fn, (ms||0)*SLOW, ...a);
  const raf=window.requestAnimationFrame.bind(window); window.requestAnimationFrame = cb => raf(() => cb(performance.now()));
  window.__idle=[]; document.addEventListener('DOMContentLoaded',()=>{ const sc=document.getElementById('hero');
    new MutationObserver(()=>{ const on=sc.classList.contains('idle'); if(on && !window.__last) window.__idle.push(performance.now()); window.__last=on; }).observe(sc,{attributes:true}); });
})();""" % SLOW
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':600,'height':380})
    pg.add_init_script(INIT)
    cdp=pg.context.new_cdp_session(pg); cdp.send('Animation.enable')
    pg.goto('file:///home/claude/intellicamp/parent-hero-animation/email-record.html?email', wait_until='domcontentloaded')
    cdp.send('Animation.setPlaybackRate',{'playbackRate':1/SLOW})
    
    box=pg.locator('.eframe').bounding_box(); clip={k:box[k] for k in ('x','y','width','height')}
    while len(pg.evaluate('window.__idle'))<1: time.sleep(0.2)
    T0=pg.evaluate('window.__idle[0]')/1000 + 0.005
    i=0
    while True:
        target=T0+i/FPS
        while pg.evaluate('performance.now()')/1000 < target: time.sleep(0.004)
        idl=pg.evaluate('window.__idle')
        if len(idl)>=2 and target >= idl[1]/1000 + 0.005: break
        pg.screenshot(path=f'frames/f{i:03d}.png',clip=clip); i+=1
    print('frames',i,'cycle',round(idl[1]/1000-idl[0]/1000,3)); b.close()
