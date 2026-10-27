(() => {
  const pv = document.getElementById('busuanzi_value_site_pv');
  const uv = document.getElementById('busuanzi_value_site_uv');
  if (!pv || !uv) return;

  let settled = false;
  const finish = data => {
    const sitePv = data && (data.site_pv ?? data.busuanzi_site_pv);
    const siteUv = data && (data.site_uv ?? data.busuanzi_site_uv);
    if (settled || sitePv == null || siteUv == null) return false;
    settled = true;
    pv.textContent = sitePv;
    uv.textContent = siteUv;
    return true;
  };

  const jsonp = endpoint => new Promise((resolve, reject) => {
    const callback = `BusuanziFallback_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const script = document.createElement('script');
    const timer = setTimeout(() => cleanup(reject, new Error('timeout')), 4500);
    const cleanup = (done, value) => {
      clearTimeout(timer);
      script.remove();
      delete window[callback];
      done(value);
    };
    window[callback] = data => cleanup(resolve, data);
    script.onerror = () => cleanup(reject, new Error('load failed'));
    script.referrerPolicy = 'no-referrer-when-downgrade';
    script.src = `${endpoint}${endpoint.includes('?') ? '&' : '?'}jsonpCallback=${callback}`;
    document.head.appendChild(script);
  });

  Promise.any([
    jsonp('https://busuanzi.ibruce.info/busuanzi'),
    jsonp('https://counter.busuanzi.icodeq.com/')
  ]).then(finish).catch(() => fetch('https://cdn.busuanzi.cc/api.php', {
    method: 'POST',
    body: JSON.stringify({ url: location.href, referrer: document.referrer })
  }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(finish)).catch(() => {
    pv.textContent = '暂不可用';
    uv.textContent = '暂不可用';
  });
})();
