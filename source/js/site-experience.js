/* First-party enhancements. Navigation and content remain usable without this file. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k,v) { try { localStorage.setItem(k,v); } catch {} } };
  const root = document.documentElement;
  let noticeTimer;
  function notice(text) {
    let box = $('.site-notice');
    if (!box) { box = document.createElement('div');box.className='site-notice';box.setAttribute('role','status');document.body.append(box); }
    box.textContent=text;box.hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>box.hidden=true,2500);
  }
  const themeButton=document.createElement('button');themeButton.type='button';themeButton.className='site-theme-toggle';
  const themeName=()=>({light:'浅色',dark:'深色',auto:'跟随系统'}[root.getAttribute('data-theme') || 'auto']);
  const labelTheme=()=>{themeButton.textContent='显示主题 · '+themeName()+'  ◐';themeButton.setAttribute('aria-label','当前'+themeName()+'，切换显示主题');};
  $('.l_left .nav-area')?.append(themeButton);labelTheme();
  themeButton.addEventListener('click',()=>{
    const next=({auto:'light',light:'dark',dark:'auto'})[root.getAttribute('data-theme')||'auto'];
    next==='auto'?root.removeAttribute('data-theme'):root.setAttribute('data-theme',next);store.set('Stellar.theme',next);labelTheme();
    const frame=$('iframe.giscus-frame');frame?.contentWindow?.postMessage({giscus:{setConfig:{theme:next==='auto'?'preferred_color_scheme':next}}},'https://giscus.app');
  });
  const article=$('.l_main>article.md-text.content');
  if(article && $('#reader-focus')){
    const estimate=Math.max(1,Math.ceil(article.textContent.replace(/\s/g,'').length/450));$('#reading-estimate').textContent=`约 ${estimate} 分钟 · 阅读设置`;
    const setLarge=large=>{root.classList.toggle('reader-large',large);$('#reader-size').setAttribute('aria-pressed',String(large));};
    setLarge(store.get('caius.reader.large')==='true');
    $('#reader-size').addEventListener('click',()=>{const large=!root.classList.contains('reader-large');setLarge(large);store.set('caius.reader.large',String(large));});
    $('#reader-focus').addEventListener('click',()=>{const focus=root.classList.toggle('reader-focus');$('#reader-focus').setAttribute('aria-pressed',String(focus));$('#reader-focus').textContent=focus?'退出专注':'专注阅读';});
    $('#reader-link').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(location.href);notice('文章链接已复制');}catch{notice('暂时无法自动复制，请从浏览器地址栏复制链接');}});
    const progress=$('#reading-progress');let queued=false;
    const update=()=>{queued=false;const rect=article.getBoundingClientRect();const range=Math.max(1,article.offsetHeight-innerHeight);const percent=Math.max(0,Math.min(100,-rect.top/range*100));progress.firstElementChild.style.width=percent+'%';progress.setAttribute('aria-valuenow',String(Math.round(percent)));};
    const schedule=()=>{if(!queued){queued=true;requestAnimationFrame(update);}};
    addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);if(window.ResizeObserver)new ResizeObserver(schedule).observe(article);schedule();
  }
  // Native image dialog: no image CDN or third-party lightbox is needed.
  if(article && typeof HTMLDialogElement!=='undefined'){
    const dialog=document.createElement('dialog');dialog.className='zoom-dialog';dialog.setAttribute('aria-label','放大查看图示');
    const close=document.createElement('button');close.type='button';close.textContent='关闭 ×';close.addEventListener('click',()=>dialog.close());
    const image=document.createElement('img'),caption=document.createElement('p');dialog.append(close,image,caption);document.body.append(dialog);
    dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
    article.querySelectorAll('img').forEach(img=>{
      if(img.closest('a,button')||img.classList.contains('emoji')||img.hasAttribute('no-lazy'))return;
      const button=document.createElement('button');button.type='button';button.className='image-zoom';button.setAttribute('aria-label','放大图示：'+(img.alt||'文章配图'));
      img.before(button);button.append(img);img.decoding='async';
      button.addEventListener('click',()=>{image.src=img.currentSrc||img.src;image.alt=img.alt;caption.textContent=img.alt;dialog.showModal();});
    });
  }
  // Load the index only when the reader asks to search. Share the pending request.
  const navButton=$('[data-site-nav]');
  navButton?.addEventListener('click',()=>{const open=$('.l_body').toggleAttribute('leftbar');navButton.setAttribute('aria-expanded',String(open));});
  $('[data-site-search]')?.addEventListener('click',()=>{$('.l_body')?.setAttribute('leftbar','');navButton?.setAttribute('aria-expanded','true');$('#search-input')?.focus();});
  const input=$('#search-input'),results=$('#search-result'),wrapper=$('#search-wrapper');
  if(!input||!results||!wrapper)return;
  const hint=document.createElement('div');hint.className='search-shortcut';hint.textContent='⌘ / Ctrl K 搜索 · Esc 清空';wrapper.after(hint);
  const status=document.createElement('div');status.className='search-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');results.before(status);
  input.placeholder='搜索文章、概念、代码…';input.setAttribute('aria-controls','search-result');
  input.closest('form').addEventListener('submit',e=>{e.preventDefault();results.querySelector('a')?.focus();});
  let data,pending,version=0,timer;
  async function load(){
    if(data)return data;
    if(!pending){
      pending=(async()=>{const response=await fetch('/search.json',{signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error('HTTP '+response.status);const rows=await response.json();return rows.filter(r=>r.title&&r.path).map(r=>({...r,lowerTitle:r.title.toLowerCase(),lowerContent:(r.content||'').toLowerCase()}));})();
    }
    try{data=await pending;return data;}catch(e){pending=null;throw e;}
  }
  async function search(){
    const current=++version,query=input.value.trim().toLowerCase();results.replaceChildren();
    if(!query){wrapper.setAttribute('searching','false');status.textContent='';return;}
    wrapper.setAttribute('searching','true');status.textContent='正在搜索…';
    try{
      const rows=await load();if(current!==version)return;
      const terms=query.split(/\s+/).filter(Boolean),filter=input.dataset.filter||'';
      const found=rows.filter(r=>!filter||r.path.includes(filter)).map(r=>{
        if(!terms.every(t=>r.lowerTitle.includes(t)||r.lowerContent.includes(t)))return null;
        const score=terms.reduce((n,t)=>n+(r.lowerTitle.includes(t)?20:1),0)+(r.lowerTitle.includes(query)?30:0);
        return {r,score};
      }).filter(Boolean).sort((a,b)=>b.score-a.score).slice(0,40);
      status.textContent=found.length?`找到 ${found.length===40?'至少 ':''}${found.length} 篇结果，标题匹配优先`:'没有找到相关文章，试试更短的关键词。';
      const list=document.createElement('ul');list.className='search-result-list';
      found.forEach(({r})=>{
        let url;try{url=new URL(r.path,location.origin);if(url.origin!==location.origin)return;}catch{return;}
        const li=document.createElement('li'),a=document.createElement('a'),title=document.createElement('span'),snippet=document.createElement('p');a.href=url.pathname;title.className='search-result-title';title.textContent=r.title;snippet.className='search-result-content';
        const positions=terms.map(t=>r.lowerContent.indexOf(t)).filter(i=>i>=0);const start=Math.max(0,(positions.length?Math.min(...positions):0)-24);
        snippet.textContent=(start?'…':'')+(r.content||'').slice(start,start+125)+'…';a.append(title,snippet);li.append(a);list.append(li);
      });results.append(list);
    }catch{if(current===version){status.textContent='搜索索引加载失败。';const retry=document.createElement('button');retry.type='button';retry.textContent='重试';retry.addEventListener('click',search);status.append(' ',retry);}}
  }
  input.addEventListener('input',()=>{version++;clearTimeout(timer);timer=setTimeout(search,160);});
  input.addEventListener('keydown',e=>{if(e.key==='Escape'){input.value='';search();input.blur();}if(e.key==='ArrowDown'){e.preventDefault();results.querySelector('a')?.focus();}});
  results.addEventListener('keydown',e=>{const links=[...results.querySelectorAll('a')],i=links.indexOf(document.activeElement);if(e.key==='ArrowDown'){e.preventDefault();links[Math.min(i+1,links.length-1)]?.focus();}if(e.key==='ArrowUp'){e.preventDefault();i<=0?input.focus():links[i-1].focus();}if(e.key==='Escape'){input.value='';search();input.focus();}});
  addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();const body=$('.l_body');if(innerWidth<=667)body?.setAttribute('leftbar','');input.focus();}});
})();
