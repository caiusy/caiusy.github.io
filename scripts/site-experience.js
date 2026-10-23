'use strict';
// Keep enhancements in the source repository, independent of the theme checkout.
hexo.extend.filter.register('after_render:html', function (html) {
  if (!html.includes('name="hexo-theme"')) return html;
  const match = html.match(/<meta property="og:url" content="([^"]+)"/);
  const url = match && match[1];
  const path = url ? new URL(url).pathname : '';
  html = html.replace('<body>', '<body><a class="skip-link" href="#main">跳到正文</a><div class="mobile-sitebar"><a href="/">Caius Lu<span>技术札记</span></a><div><button type="button" data-site-search aria-label="打开站内搜索">搜索</button><button type="button" data-site-nav aria-label="打开站点导航" aria-expanded="false">导航 ☰</button></div></div>');
  html = html.replace('id="main"', 'id="main" tabindex="-1"');
  if (url && !html.includes('rel="canonical"')) html = html.replace('</head>', `<link rel="canonical" href="${url}"></head>`);
  html = html.replace(/<nav class="menu[^>]*>[\s\S]*?<\/nav>/g, menu => menu.replace(/<a class="nav-item[^>]*>[\s\S]*?<\/a>/g, link => {
    const href = link.match(/href="([^"]+)"/)[1];
    const title = link.match(/class="nav-label">([^<]+)/)?.[1] || '导航';
    link = link.replace('nav-item active', 'nav-item');
    if (href === path || (href === '/' && /^\/\d{4}\//.test(path))) link = link.replace('nav-item', 'nav-item active').replace('<a ', '<a aria-current="page" ');
    return link.replace(/title="[^"]*"/, `title="${title}"`).replace('<a ', `<a aria-label="${title}" `);
  }));
  html = html.replace('id="search-input"', 'id="search-input" aria-label="搜索文章、主题与笔记" autocomplete="off"');
  html = html.replace("class='laptop-only rightbar-toggle mobile'", "aria-label='打开文章目录' class='laptop-only rightbar-toggle mobile'")
    .replace("class='mobile-only leftbar-toggle mobile'", "aria-label='打开站点导航' class='mobile-only leftbar-toggle mobile'");
  if (path === '/') {
    const hero = `<section class="fieldnotes-hero" aria-label="站点介绍">
      <p class="eyebrow">CAIUS LU / ENGINEERING FIELDNOTES</p>
      <h1>把原理讲透，<br><em>把系统做实。</em></h1>
      <p class="hero-description">从路侧感知到多模态智能，记录计算机视觉、雷视融合与大模型的推导、复现和工程实践。</p>
      <div class="hero-actions"><a href="/projects/">探索项目 <span aria-hidden="true">↗</span></a><a href="/llm-reading-map/">大模型学习路线 <span aria-hidden="true">→</span></a></div>
      <div class="topic-tracks"><a href="/roadside-bev/"><small>01 / PERCEPTION</small><strong>视觉与 BEV</strong><span>相机几何 · 多视角感知</span></a><a href="/radar-camera-fusion/"><small>02 / FUSION</small><strong>雷达 × 相机</strong><span>标定 · 融合 · 工程验证</span></a><a href="/llm-reading-map/"><small>03 / INTELLIGENCE</small><strong>多模态与大模型</strong><span>Attention · 训练 · 交互实验</span></a></div>
      <a class="lab-feature" href="/dsa-lab/"><span class="lab-dot" aria-hidden="true"></span><span><small>交互实验室</small><strong>亲手调整 Gate，看懂 DeepSeek DSA</strong></span><span aria-hidden="true">↗</span></a>
    </section><div class="latest-heading"><h2>最近写下的</h2><a href="/archives/">全部归档 →</a></div>`;
    html = html.replace('<div class="post-list post">', hero + '<div class="post-list post">');
  }
  if (/layout="post"/.test(html)) {
    const toolbar = `<div class="reader-tools" role="group" aria-label="阅读设置"><span id="reading-estimate">长文阅读</span><div><button type="button" id="reader-size" aria-label="切换正文字号" aria-pressed="false">大字</button><button type="button" id="reader-focus" aria-pressed="false">专注阅读</button><button type="button" id="reader-link">复制链接</button></div></div><div id="reading-progress" role="progressbar" aria-label="阅读进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span></span></div>`;
    html = html.replace(/(<article class="md-text content[^>]*>)/, toolbar + '$1');
  }
  return html;
});

// Keep legacy math articles working while avoiding MathJax on ordinary pages.
hexo.extend.filter.register('before_post_render', function(data) {
  if (data.mathjax == null && /\.md$/i.test(data.source || '')) {
    const prose=(data.content || data.raw || '').replace(/```[\s\S]*?```/g,'').replace(/`[^`]*`/g,'');
    if (/\$\$|\\\[|\\\(|\$[^$\n]+\$/.test(prose)) data.mathjax=true;
  }
  return data;
},8);

hexo.extend.generator.register('caius_search', function(locals) {
  const { stripHTML } = require('hexo-util');
  const rows=[];
  for (const collection of [locals.posts.sort('-date'),locals.pages]) {
    collection.each(page=>{
      if (!page.title || page.indexing===false || !['post','page','wiki'].includes(page.layout)) return;
      const content=stripHTML((page.content||'').replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi,'')).replace(/\s+/g,' ').trim();
      rows.push({title:page.title,path:'/'+page.path.replace(/index\.html$/,''),content});
    });
  }
  return {path:'search.json',data:JSON.stringify(rows)};
});
