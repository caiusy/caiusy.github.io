/* Draggable companion; artwork: PokéAPI sprites/official-artwork/25.png. */
(() => {
  'use strict';
  const host = document.createElement('aside');
  host.className = 'study-cat study-pikachu';
  host.setAttribute('aria-label', '学习伙伴皮卡丘');
  host.innerHTML = `<div class="cat-note" role="status">皮卡丘陪你学习，拖动我可以换位置。⚡</div>
    <button class="cat-toggle" type="button" aria-label="收起学习伙伴" title="收起学习伙伴">−</button>
    <button class="cat-character" type="button" aria-label="拖动皮卡丘，点击获取学习提示" title="拖动我，点一下有提示">
      <span class="pika-sparks" aria-hidden="true"><i>✦</i><i>⚡</i><i>✦</i></span>
      <picture class="pika-art"><source srcset="/images/companion/pikachu-showdown.gif" type="image/gif"><img src="/images/companion/pikachu-official.png" alt="" width="475" height="475" draggable="false" decoding="async"></picture>
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
  const moods = [
    ['pika-happy', '太棒了！保持这个节奏继续推导。⚡'],
    ['pika-think', '让我想想……先从最小例子验证。💡'],
    ['pika-zap', '能量充满！把复杂问题拆成几个小步骤。✨'],
    ['pika-rest', '深呼吸一下，好的答案值得慢慢打磨。🌙']
  ];
  let moodIndex = 0, tipIndex = 0, drag = null, suppressClick = false;
  let moodTimer;
  function setMood(index, announce = true) {
    moodIndex = (index + moods.length) % moods.length;
    host.classList.remove(...moods.map(item => item[0]));
    host.classList.add(moods[moodIndex][0]);
    if (announce) note.textContent = moods[moodIndex][1];
    clearTimeout(moodTimer);
    moodTimer = setTimeout(() => setMood(moodIndex + 1), 7500);
  }
  setMood(0, false);
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
    setMood(moodIndex + 1);
    note.textContent = `${moods[moodIndex][1]} ${tips[tipIndex++ % tips.length]}`;
  });
  character.addEventListener('mouseenter', () => host.classList.add('pika-hover'));
  character.addEventListener('mouseleave', () => host.classList.remove('pika-hover'));
  addEventListener('resize', placeSaved);
})();
