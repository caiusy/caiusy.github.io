const $=s=&gt;document.querySelector(s);
const states={kv:0,rope:0,mla:0}; const timers={};
const fmt=(v,n=4)=&gt;Math.abs(v)&lt;1e-9?'0.0000':v.toFixed(n);
const vec=a=&gt;'['+a.map(x=&gt;fmt(x)).join(', ')+']';
const softmax=a=&gt;{const b=a.map(x=&gt;Math.exp(x-Math.max(...a)));const s=b.reduce((x,y)=&gt;x+y,0);return b.map(x=&gt;x/s)};
function bars(a){return '<div class="bars">'+a.map((x,i)=&gt;`<div class="bar" style="height:${25+100*x}px">${fmt(x)}<br>位置 ${i+1}</div>`).join('')+'</div>'}
const kvCaptions=[
'前两条 K、V 已经存在缓存中。它们来自旧位置，而旧 Q 已完成任务。',
'处理位置 3：计算新的 q₃、k₃、v₃。旧位置的投影不用再算。',
'只把 k₃、v₃ 追加到缓存；本步也可以读取当前 token 自己。',
'当前 q₃ 与所有三个 Key 点积，除以 √2，再用同一个 Softmax 归一化。',
'权重乘三个 Value，得到当前输出。此后不必长期保留 q₃。',
'位置 4 到来：q₄ 接棒，只追加 k₄、v₄。前三条缓存保持不变。',
'新 Query 重新生成整行权重。不能复用上一行权重，也不能只修改旧输出。'];
function renderKV(){const s=states.kv;const t=s&gt;=5?4:3;const weights=t===3?softmax([1,1,2].map(x=&gt;x/Math.sqrt(2))):softmax([2,0,2,2].map(x=&gt;x/Math.sqrt(2)));
const k=['[1, 0]','[0, 1]','[1, 1]','[1, −1]'],v=['[10, 0]','[0, 20]','[6, 6]','[4, 8]'];const count=s&lt;2?2:t;
let h=`<div class="tag">持久缓存 / ${count} 条历史记录</div><div class="tokens">`;
for(let j=0;j<count;j++)h+=`<div class="token ${j===t-1?'new':'old'}">位置 ${j+1}<br>K ${k[j]}<br>V ${v[j]}</count;j++)h+=`<div></div>`;
h+='';
h+=s===0?'<div class="calc ghost">旧 q₁、q₂：无需跨步保存</div>':`<div class="calc">当前 q${t===3?'₃ = [1, 1]':'₄ = [2, 0]'} ${s&gt;=5?'← 新 Query':''}</div>`;
if(s===1)h+='<div class="calc">待追加：k₃ = [1, 1]，v₃ = [6, 6]</div>';
if(s&gt;=3&amp;&amp;s!==5)h+=bars(weights);
if(s===4)h+='<div class="calc">o₃ = [5.503490, 7.986041]</div>';
if(s===6)h+='<div class="calc">o₄ = [6.166907, 5.816113]<br>q₃ 不参与这次计算。</div>';
$('#kv-stage').innerHTML=h;$('#kv-caption').textContent=kvCaptions[s];$('#kv-counter').textContent=`${s+1} / 7`;
}
let rp={m:3,n:1,shift:0};
function renderRope(){let m=rp.m+rp.shift,n=rp.n+rp.shift;const a=m*Math.PI/6,b=n*Math.PI/6;const cx=180,cy=115,r=85;const point=x=&gt;[cx+r*Math.cos(x),cy-r*Math.sin(x)];let q=point(a),k=point(b);let score=Math.cos(a-b);
$('#rope-stage').innerHTML=`<div class="rope-input"><label>Query 位置 <input aria-label="Query 位置" id="qm" type="range" min="1" max="12" value="${rp.m}"><b>${m}</b></label><label>Key 位置 <input aria-label="Key 位置" id="kn" type="range" min="1" max="12" value="${rp.n}"><b>${n}</b></label></div><svg class="diagram" viewBox="0 0 600 235" role="img" aria-label="Query 和 Key 的二维旋转"><circle cx="180" cy="115" r="85" fill="none" stroke="#bdcebd"></circle><path d="M65 115H295M180 12V220" stroke="#c4d1c1"></path><path d="M180 115L${q[0]} ${q[1]}" stroke="#b67537" stroke-width="4"></path><path d="M180 115L${k[0]} ${k[1]}" stroke="#337958" stroke-width="4"></path><circle cx="${q[0]}" cy="${q[1]}" r="6" fill="#b67537"></circle><circle cx="${k[0]}" cy="${k[1]}" r="6" fill="#337958"></circle><g font-family="PingFang SC,sans-serif" font-size="17" fill="#254c3e"><text x="325" y="65">Q 旋转 ${m*30}°</text><text x="325" y="103">K 旋转 ${n*30}°</text><text x="325" y="141">相对位移 n−m = ${n-m}</text><text x="325" y="182">点积 = ${fmt(score)}</text></g></svg><div class="calc">原始 q = k = [1, 0] · 每步 30°<br>cos((${n}−${m}) × 30°) = ${fmt(score)}</div>`;
$('#qm').oninput=e=&gt;{rp.m=+e.target.value;renderRope()};$('#kn').oninput=e=&gt;{rp.n=+e.target.value;renderRope()};
$('#rope-caption').textContent=rp.shift?`两个位置同时加 ${rp.shift}，相对位移和点积保持不变。拖动滑块还可以单独改变一个位置。`:'拖动任意位置，观察点积变化；播放会同时平移两个位置。此例用于演示，真实 RoPE 使用多组频率。';
$('#rope-counter').textContent=`整体平移 +${rp.shift}`;
}
const mlaCaptions=[
'跨步保存三条 cᴷⱽ 与旋转后的 kᴿ。当前 qᶜ、qᴿ 只是本次查询。',
'先把当前内容 Query 乘上 Wᵁᴷ 的转置，得到潜空间 Query [3, 1]。',
'潜空间 Query 直接与 C 点积；不需要展开整张内容 Key。',
'位置 Query 读取缓存中的旋转 Key，得到独立的位置匹配项。',
'内容与位置分数先相加，再除以 √(2+2)=2，然后只做一次 Softmax。',
'先用注意力权重汇总压缩向量：z=αC。此时仍未展开历史 Value。',
'只对汇总向量上投影。结果与逐个展开历史 Value 后汇总完全一致。'];
function renderMLA(){const s=states.mla,a=softmax([1.75,(1+Math.sqrt(3)/2)/2,2.5]);const z=[a[0]+a[2],a[1]+a[2]];let h='<div class="tokens"><div class="token old">内容缓存 C<br>[1,0] · [0,1] · [1,1]</div><div class="token old">位置缓存 Kᴿ<br>30° · 60° · 90° 旋转 Key</div></div>';
const lines=['当前 qᶜ = [1, 1]，qᴿ = [0, 1]','q̂ = qᶜ (Wᵁᴷ)ᵀ = [3, 1]','内容点积 = q̂ Cᵀ = [3, 1, 4]','位置点积 = qᴿ (Kᴿ)ᵀ = [0.5, 0.866025, 1]','总分数 = [1.75, 0.933013, 2.5]','z = αC = '+vec(z),'o = z Wᵁⱽ = '+vec([2*z[0],3*z[1]])];
for(let i=Math.max(0,s-2);i&lt;=s;i++)h+=`<div class="calc ${i===s?'':'ghost'}">${lines[i]}</div>`;
if(s===4)h+=bars(a);if(s===6)h+='<div class="tag">显式展开路径：α(C Wᵁⱽ) = [1.7517, 2.1570] ✓</div>';
$('#mla-stage').innerHTML=h;$('#mla-caption').textContent=mlaCaptions[s];$('#mla-counter').textContent=`${s+1} / 7`;
}
const renders={kv:renderKV,rope:renderRope,mla:renderMLA};
function stop(k){clearInterval(timers[k]);delete timers[k];$(`[data-play="${k}"]`).textContent='播放'}
function next(k){if(k==='rope'){rp.shift=(rp.shift+1)%7}else states[k]=(states[k]+1)%7;renders[k]()}
document.querySelectorAll('[data-play]').forEach(b=&gt;b.onclick=()=&gt;{let k=b.dataset.play;if(timers[k]){stop(k);return}b.textContent='暂停';timers[k]=setInterval(()=&gt;next(k),1800)});
document.querySelectorAll('[data-next]').forEach(b=&gt;b.onclick=()=&gt;{stop(b.dataset.next);next(b.dataset.next)});
document.querySelectorAll('[data-reset]').forEach(b=&gt;b.onclick=()=&gt;{let k=b.dataset.reset;stop(k);states[k]=0;if(k==='rope')rp={m:3,n:1,shift:0};renders[k]()});
Object.values(renders).forEach(f=&gt;f());document.addEventListener('visibilitychange',()=&gt;{if(document.hidden)Object.keys(timers).forEach(stop)});
window.addEventListener('scroll',()=&gt;{let d=document.documentElement;$('.progress').style.width=(100*d.scrollTop/(d.scrollHeight-d.clientHeight))+'%'},{passive:true});
