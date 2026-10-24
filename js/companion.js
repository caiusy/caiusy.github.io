/* Local, draggable reading companion. */
(() => {
  'use strict';
  const host = document.createElement('aside');
  host.className = 'study-cat study-pikachu';
  host.setAttribute('aria-label', '学习伙伴皮卡丘');
  host.innerHTML = `<div class="cat-note" role="status">皮卡丘陪你学习，拖动我可以换位置。⚡</div>
    <button class="cat-toggle" type="button" aria-label="收起学习伙伴" title="收起学习伙伴">−</button>
    <button class="cat-character" type="button" aria-label="拖动皮卡丘，点击获取学习提示" title="拖动我，点一下有提示">
      <svg viewBox="0 0 160 174" aria-hidden="true">
        <ellipse cx="80" cy="165" rx="47" ry="7" fill="#172b4d" opacity=".13"/>
        <g class="pika-tail"><path d="M114 110 144 83 133 76 155 52 132 58 141 70 115 81Z" fill="#f7c83d" stroke="#664c27" stroke-width="3" stroke-linejoin="round"/></g>
        <g class="pika-body"><path d="M48 111q-10 24-2 42 14 13 35 13t35-13q9-22-3-42Z" fill="#ffda45" stroke="#664c27" stroke-width="3"/><path d="m51 134-14 10q-3 11 11 12l18-8m45-14 13 10q4 11-10 12l-19-8" fill="#ffda45" stroke="#664c27" stroke-width="3" stroke-linecap="round"/>
        <path d="M46 67 25 12q-3-11 5-10 17 7 38 43M94 45q20-36 36-43 9-1 5 10l-21 55" fill="#ffda45" stroke="#664c27" stroke-width="3" stroke-linejoin="round"/><path d="M25 12q-3-11 5-10 7 3 17 12L31 28Zm110 0q4-11-5-10-7 3-17 12l16 14Z" fill="#664c27"/><path d="M30 53q8-25 50-25t50 25v41q-4 40-50 43-45-3-50-43Z" fill="#ffda45" stroke="#664c27" stroke-width="3"/>
        <g class="pika-eyes" fill="#392f2a"><ellipse cx="56" cy="78" rx="7" ry="10"/><ellipse cx="104" cy="78" rx="7" ry="10"/></g><circle cx="58" cy="75" r="2.5" fill="white"/><circle cx="106" cy="75" r="2.5" fill="white"/><circle cx="43" cy="100" r="11" fill="#ed6a51"/><circle cx="117" cy="100" r="11" fill="#ed6a51"/><path d="m76 96 4 3 4-3m-4 4q-4 8-9 3m9-3q4 8 9 3" fill="none" stroke="#664c27" stroke-width="2.5" stroke-linecap="round"/></g>
        <path d="m66 139 14 4 14-4-2 12-12 4-12-4Z" fill="#315f91"/><path d="m74 145 5 4 7-6" fill="none" stroke="white" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </button>`;
  document.body.append(host);
  const toggle = host.querySelector('.cat-toggle');
  const character = host.querySelector('.cat-character');
  const note = host.querySelector('.cat-note');
  const store = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
  const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
  let small = read('caius.pika.small') === 'true';
  const saved = (() => { try { return JSON.parse(read('caius.pika.position')); } catch { return null; } })();
  let position = saved && Number.isFinite(saved.x) && Number.isFinite(saved.y) ? saved : null;
  const clamp = (value, max) => Math.min(Math.max(value, 0), Math.max(0, max));
  function place(x, y, persist = false) {
    const maxX = Math.max(0, innerWidth - host.offsetWidth);
    const maxY = Math.max(0, innerHeight - host.offsetHeight);
    const left = clamp(x, maxX), top = clamp(y, maxY);
    host.style.left = `${left}px`;
    host.style.top = `${top}px`;
    host.style.bottom = 'auto';
    position = { x: maxX ? left / maxX : 0, y: maxY ? top / maxY : 0 };
    if (persist) store('caius.pika.position', JSON.stringify(position));
  }
  function placeSaved() {
    if (position) place(position.x * Math.max(0, innerWidth - host.offsetWidth),
      position.y * Math.max(0, innerHeight - host.offsetHeight));
  }
  function render() {
    host.classList.toggle('is-small', small);
    toggle.textContent = small ? '+' : '−';
    toggle.setAttribute('aria-label', small ? '展开学习伙伴' : '收起学习伙伴');
    toggle.title = toggle.getAttribute('aria-label');
    toggle.setAttribute('aria-expanded', String(!small));
    character.hidden = small;
    note.hidden = small;
    requestAnimationFrame(placeSaved);
  }
  render();
  toggle.addEventListener('click', () => { small = !small; render(); store('caius.pika.small', String(small)); });
  const tips = ['先说清输入和输出，再开始写代码。', '试一个最小例子，看看每一步发生了什么。', '检查边界：空输入、重复值和最大值。', '读公式时，检查每个张量的形状。', '卡住也没关系，先把问题拆小一点。', '休息一下，喝口水再继续。'];
  let tipIndex = 0, drag = null, suppressClick = false;
  character.addEventListener('pointerdown', e => {
    if (e.button !== 0 || small) return;
    const box = host.getBoundingClientRect();
    drag = { id: e.pointerId, startX: e.clientX, startY: e.clientY, left: box.left, top: box.top, moved: false };
    character.setPointerCapture(e.pointerId);
  });
  character.addEventListener('pointermove', e => {
    if (!drag || drag.id !== e.pointerId) return;
    const dx = e.clientX - drag.startX, dy = e.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) < 6) return;
    drag.moved = true;
    host.classList.add('is-dragging');
    place(drag.left + dx, drag.top + dy);
  });
  function endDrag(e) {
    if (!drag || drag.id !== e.pointerId) return;
    suppressClick = drag.moved;
    if (drag.moved) store('caius.pika.position', JSON.stringify(position));
    drag = null;
    host.classList.remove('is-dragging');
  }
  character.addEventListener('pointerup', endDrag);
  character.addEventListener('pointercancel', endDrag);
  character.addEventListener('click', e => {
    if (suppressClick) { e.preventDefault(); suppressClick = false; return; }
    note.textContent = tips[tipIndex++ % tips.length];
  });
  addEventListener('resize', placeSaved);
})();
