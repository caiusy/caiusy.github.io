'use strict';
(()=&gt;{
const $d=id=&gt;document.getElementById(id), num=id=&gt;+$d(id).value;
const bar=(label,value,color,scale=1)=&gt;'<div class="deep-bar"><span>'+label+'</span><span class="track"><span class="fill" style="width:'+(Math.max(0,Math.min(100,value/scale*100)))+'%;background:'+color+'"></span></span><output>'+(value*100).toFixed(0)+'%</output></div>';
const chart=(id,series,range,{width=600,height=220,zero=false,marker=null}={})=&gt;{
 const lo=zero?Math.min(0,range[0]):range[0],hi=range[1],left=36,right=width-16,top=20,bottom=height-32;
 const px=(x,n)=&gt;left+x/(n-1)*(right-left),py=y=&gt;bottom-(y-lo)/(hi-lo||1)*(bottom-top);
 let html='<line x1="'+left+'" y1="'+bottom+'" x2="'+right+'" y2="'+bottom+'" stroke="#46617a"><line x1="'+left+'" y1="'+top+'" x2="'+left+'" y2="'+bottom+'" stroke="#46617a"><text x="40" y="16" fill="#abc1d5" font-size="11">'+hi.toFixed(2)+'</text><text x="40" y="'+(bottom-5)+'" fill="#abc1d5" font-size="11">'+lo.toFixed(2)+'</text>';
 for(const s of series){if(!s.values.length)continue;html+='<polyline fill="none" stroke="'+s.color+'" stroke-width="3" points="'+s.values.map((v,i)=>px(i,s.values.length).toFixed(1)+','+py(v).toFixed(1)).join(' ')+'">'}
 if(marker){html+='<circle cx="'+px(marker.index,marker.total)+'" cy="'+py(marker.value)+'" r="6" fill="'+(marker.color||'#ffbf72')+'">'}
 $d(id).innerHTML=html;
};
function attention(){const q=num('query')-1,t=num('temp'),causal=$d('mask').value==='causal',raw=K.map(k=&gt;Q[q][0]*k[0]+Q[q][1]*k[1]),scaled=raw.map(v=&gt;v/Math.SQRT2);
 const probs=mask=&gt;{let z=scaled.map((v,i)=&gt;mask&amp;&amp;i&gt;q?-Infinity:v/t),max=Math.max(...z.filter(Number.isFinite)),ex=z.map(v=&gt;Number.isFinite(v)?Math.exp(v-max):0),sum=ex.reduce((a,b)=&gt;a+b,0);return ex.map(x=&gt;x/sum)};
 const current=probs(causal),c=probs(true),all=probs(false);
 $d('deepAttentionRows').innerHTML=words.map((w,i)=&gt;'<tr'+(i===q?' class="picked" ':'')+'="">K'+(i+1)+'「'+w+'」'+raw[i].toFixed(3)+''+scaled[i].toFixed(3)+''+(causal&amp;&amp;i&gt;q?'−∞':(scaled[i]/t).toFixed(3))+''+(current[i]*100).toFixed(1)+'%').join('');
 $d('deepAttentionFormula').textContent='当前 Q'+(q+1)+'：softmax(['+scaled.map((v,i)=&gt;causal&amp;&amp;i&gt;q?'−∞':(v/t).toFixed(2)).join(', ')+']) = ['+current.map(v=&gt;v.toFixed(3)).join(', ')+']';
 $d('deepAttentionCompare').innerHTML='<b>因果掩码</b>'+words.map((w,i)=&gt;bar(w,c[i],'#58d8b2')).join('')+'<b>无掩码</b>'+words.map((w,i)=&gt;bar(w,all[i],'#72c7ff')).join('');
}
function kv(){const L=num('ctx'),ly=num('layers'),h=num('heads'),bytes=num('bytes'),lat=num('latent'),batch=num('deepBatch'),budget=num('deepBudget');
 const fullPer=ly*2*h*128*bytes*batch/2**30,latPer=ly*lat*bytes*batch/2**30,full=L*fullPer,compact=L*latPer;
 $d('deepBatchV').textContent=batch;$d('deepBudgetV').textContent=budget+' GiB';
 $d('deepKvNumbers').innerHTML='<div class="metric"><strong>'+full.toFixed(2)+' GiB</strong><small>普通 KV · 当前长度</small></div><div class="metric"><strong>'+compact.toFixed(2)+' GiB</strong><small>latent · 当前长度</small></div><div class="metric"><strong>'+Math.floor(budget/fullPer).toLocaleString()+'</strong><small>普通 KV 理论 token 上限 / 序列</small></div>';
 const max=Math.max(full,compact,budget)*1.12,x=Array.from({length:25},(_,i)=&gt;i/24*L);chart('deepKvChart',[{color:'#ffbf72',values:x.map(v=&gt;v*fullPer)},{color:'#58d8b2',values:x.map(v=&gt;v*latPer)},{color:'#ff788d',values:x.map(()=&gt;budget)}],[0,max]);
 $d('deepKvNote').textContent='在 '+batch+' 条同长序列、'+budget+' GiB 的专用预算下，普通 KV 理论每条上限约 '+Math.floor(budget/fullPer).toLocaleString()+' token；简化 latent 缓存约 '+Math.floor(budget/latPer).toLocaleString()+' token。实际可用长度仍受模型上下文上限和其他显存占用限制。';
}
function moe(){let N=num('tokens'),E=num('experts'),top=num('top'),token=Math.min(num('deepToken'),N-1);$d('deepToken').max=N-1;$d('deepToken').value=token;$d('deepTokenV').textContent=token;
 const scores=Array.from({length:E},(_,e)=&gt;{let affinity=2.2/(e+1)+.6*Math.sin(token*1.7+e*2.3)+.25*Math.cos(token*.47-e);return {e,affinity,bias:bias[e]||0,total:affinity+(bias[e]||0)}});const chosen=new Set([...scores].sort((a,b)=&gt;b.total-a.total).slice(0,top).map(x=&gt;x.e));
 $d('deepMoeRows').innerHTML=scores.map(x=&gt;'<tr'+(chosen.has(x.e)?' class="picked" ':'')+'="">Expert '+(x.e+1)+''+x.affinity.toFixed(3)+''+x.bias.toFixed(3)+''+x.total.toFixed(3)+''+(chosen.has(x.e)?'✓ 选中':'—')+'').join('');
 $d('deepMoeChoice').textContent='Token '+token+' 当前选择 Expert '+[...chosen].map(e=&gt;e+1).join('、')+'。偏置并不修改亲和分数本身，只在排序前相加。';
}
let moeHistory=[];function moeChart(){let cv=+$d('cv').textContent,r=+$d('round').textContent;if(r===0)moeHistory=[cv];else if(moeHistory.length&lt;=r)moeHistory.push(cv);else moeHistory[r]=cv;let max=Math.max(.2,...moeHistory)*1.15;chart('deepMoeChart',[{color:'#58d8b2',values:moeHistory.length===1?[cv,cv]:moeHistory}],[0,max],{height:180});}
function rlhf(){let A=num('adv'),r=num('ratio'),eps=num('clip'),beta=num('kl'),pold=num('pold'),pref=num('pref');const val=x=&gt;{let raw=x*A,c=Math.max(1-eps,Math.min(1+eps,x))*A,ppo=Math.min(raw,c),p=pold*x,kl=klBinary(p,pref);return {raw,c,ppo,kl,final:ppo-beta*kl}};
 const points=Array.from({length:131},(_,i)=&gt;.5+i*.01),raw=points.map(x=&gt;val(x).raw),ppo=points.map(x=&gt;val(x).ppo),current=val(r),range=[Math.min(...raw,...ppo)-.2,Math.max(...raw,...ppo)+.2];chart('deepRlhfChart',[{color:'#72c7ff',values:raw},{color:'#58d8b2',values:ppo}],range,{height:240,marker:{index:Math.round((r-.5)*100),total:131,value:current.ppo}});
 $d('deepRlhfLedger').innerHTML=[['未裁剪 r·A',current.raw],['裁剪候选 clip(r)·A',current.c],['PPO min',current.ppo],['二选一精确 KL',current.kl],['β × KL 惩罚',beta*current.kl],['最终示意目标',current.final]].map(([name,x])=&gt;'<div><span>'+name+'</span><b>'+x.toFixed(4)+'</b></div>').join('');
 let active=A&gt;0&amp;&amp;r&gt;1+eps||A&lt;0&amp;&amp;r&lt;1-eps;$d('deepRlhfRegion').textContent=active?'当前处于 PPO 裁剪平台：在这个单动作示意中，PPO 项对 r 的局部斜率为 0。KL 项仍随 r 变化。':'当前 PPO 项仍随 r 改变。区间边界为 ['+(1-eps).toFixed(2)+', '+(1+eps).toFixed(2)+']；是否进入平台还取决于优势符号。';
}
function rope(){const m=num('m'),n=num('n'),theta=num('theta'),shift=num('deepShift'),q=[1,.35],k=[.72,.92];const dot=d=&gt;{let angle=-d*theta,kr=[k[0]*Math.cos(angle)-k[1]*Math.sin(angle),k[0]*Math.sin(angle)+k[1]*Math.cos(angle)];return q[0]*kr[0]+q[1]*kr[1]};let d=m-n,baseline=dot(d),m2=m+shift,n2=n+shift,translated=dot(m2-n2);
 $d('deepShiftV').textContent=shift;$d('deepRopePairs').innerHTML='<div><span>原位置 (m,n)</span><b>('+m+', '+n+')</b></div><div><span>平移后位置</span><b>('+m2+', '+n2+')</b></div><div><span>两组位置差</span><b>'+d+' / '+(m2-n2)+'</b></div><div><span>原点积 / 平移后</span><b>'+baseline.toFixed(6)+' / '+translated.toFixed(6)+'</b></div>';
 $d('deepRopeProof').textContent='(m+'+shift+') − (n+'+shift+') = m−n = '+d+'。在相同 q、k、θ 下，点积差仅为数值舍入误差。';let xs=Array.from({length:129},(_,i)=&gt;i-64),ys=xs.map(dot),min=Math.min(...ys),max=Math.max(...ys);chart('deepRopeChart',[{color:'#58d8b2',values:ys}],[min-.1,max+.1],{marker:d&gt;=-64&amp;&amp;d&lt;=64?{index:d+64,total:129,value:baseline}:null});
}
let sampleCounts=[0,0,0,0,0,0],sampleTotal=0;function sampling(){let t=num('temp'),Kmax=num('topk'),p=num('topp'),max=Math.max(...logits.map(v=&gt;v/t)),ex=logits.map(v=&gt;Math.exp(v/t-max)),z=ex.reduce((a,b)=&gt;a+b,0),base=ex.map(v=&gt;v/z),order=base.map((v,i)=&gt;({i,v})).sort((a,b)=&gt;b.v-a.v),top=new Set(order.slice(0,Kmax).map(x=&gt;x.i)),kept=[],acc=0;for(const item of order.slice(0,Kmax)){kept.push(item.i);acc+=item.v;if(acc&gt;=p)break}const finalSum=kept.reduce((a,i)=&gt;a+base[i],0),keepSet=new Set(kept);
 $d('deepSamplingRows').innerHTML=tokens.map((word,i)=&gt;'<tr'+(keepset.has(i)?' class="picked" ':'')+'="">'+word+''+logits[i].toFixed(1)+''+(base[i]*100).toFixed(1)+'%'+(top.has(i)?(base[i]*100).toFixed(1)+'%':'—')+''+(keepSet.has(i)?(base[i]*100).toFixed(1)+'%':'—')+''+(distribution[i]*100).toFixed(1)+'%').join('');
 $d('deepSamplingNote').textContent='Top-P 前累计保留质量为 '+(finalSum*100).toFixed(1)+'%。末列将保留候选除以这部分质量，使概率重新合计为 100%。';sampleFrequency();
}
function sampleFrequency(){let dist=distribution,total=sampleTotal;$d('deepSamplingFreq').innerHTML=tokens.map((word,i)=&gt;'<div class="deep-bar"><span>'+word+'</span><span class="track"><span class="fill" style="width:'+(total?sampleCounts[i]/total*100:0)+'%;background:#58d8b2"></span></span><output>'+(total?(sampleCounts[i]/total*100).toFixed(0)+'%':'—')+'</output></div>').join('');let tv=total?sampleCounts.reduce((s,v,i)=&gt;s+Math.abs(v/total-dist[i]),0)/2:0;$d('deepSamplingStat').textContent=total?'累计 '+total+' 次独立抽样；经验分布与理论分布的总变差距离约 '+tv.toFixed(3)+'。样本数增大时通常更接近，但单次实验不保证单调。':'尚未进行批量抽样。先观察理论概率，再点击 100 次或 1000 次。';}
function drawSamples(n){for(let j=0;j<n;j++){let u="Math.random(),c=0,i=distribution.findIndex(v=">(c+=v)&gt;=u);sampleCounts[i&lt;0?distribution.length-1:i]++;sampleTotal++}sampleFrequency()}
const slug=location.pathname.split('/').filter(Boolean).at(-1);const watch=(ids,fn)=&gt;ids.forEach(id=&gt;$d(id).addEventListener('input',fn));
if(slug==='attention'){watch(['query','temp','mask'],attention);attention()}
if(slug==='kv-cache'){watch(['ctx','layers','heads','bytes','latent','deepBatch','deepBudget'],kv);kv()}
if(slug==='moe'){watch(['tokens','experts','top','gamma'],()=&gt;{moe();moeHistory=[];moeChart()});$d('deepToken').addEventListener('input',moe);$d('step').addEventListener('click',()=&gt;{moe();moeChart()});$d('reset').addEventListener('click',()=&gt;{moeHistory=[];moe();moeChart()});moe();moeChart()}
if(slug==='rlhf'){watch(['adv','ratio','clip','kl','pold','pref'],rlhf);for(const [id,a,r] of [['deepPositive','1.2','1.5'],['deepNegative','-1.2','0.6'],['deepInside','1.2','1.0']])$d(id).addEventListener('click',()=&gt;{$d('adv').value=a;$d('ratio').value=r;$d('adv').dispatchEvent(new Event('input'));$d('ratio').dispatchEvent(new Event('input'))});rlhf()}
if(slug==='rope'){watch(['m','n','theta','deepShift'],rope);rope()}
if(slug==='sampling'){watch(['temp','topk','topp'],()=&gt;{sampleCounts=[0,0,0,0,0,0];sampleTotal=0;sampling()});$d('deepDraw100').addEventListener('click',()=&gt;drawSamples(100));$d('deepDraw1000').addEventListener('click',()=&gt;drawSamples(1000));$d('deepDrawReset').addEventListener('click',()=&gt;{sampleCounts=[0,0,0,0,0,0];sampleTotal=0;sampleFrequency()});sampling()}
})();
const $d=id=&gt;document.getElementById(id);
function addNotebook(){
 const slug=location.pathname.split('/').filter(Boolean).at(-1);
 const configs={
  attention:{title:'Attention 实验工作台',goal:'验证温度与因果掩码怎样改变注意力分布。',steps:['选择第 2 个 Query，记录未来位置是否为 0。','把温度从 1.0 调到 0.4，比较熵和最大权重。','切换无掩码，解释为什么所有 Key 都重新参与归一化。']},
  'kv-cache':{title:'KV Cache 容量实验工作台',goal:'从单请求扩展到并发请求，找出预算先被哪种缓存耗尽。',steps:['固定 16 GiB 预算，逐渐增加并发序列。','把精度从 BF16 改成 FP8，比较绝对占用。','把上下文长度翻倍，验证缓存近似线性增长。']},
  moe:{title:'MoE 路由实验工作台',goal:'观察亲和分数、偏置和 Top-K 如何共同决定专家负载。',steps:['将 γ 设为 0，运行三轮作为无反馈基线。','恢复 γ=0.30，记录变异系数如何变化。','切换检查 token，说明 bias 只参与排序前加法。']},
  rope:{title:'RoPE 几何实验工作台',goal:'用数值验证旋转后的点积只依赖相对位置。',steps:['记下当前 m−n 和点积。','改变共同平移量，确认两组位置差仍相等。','改变 θ，观察相同距离为何得到不同相对角度。']},
  sampling:{title:'采样策略实验工作台',goal:'拆开温度、Top-K、Top-P 和重新归一化四道门。',steps:['把 Top-K 设为 1，观察输出是否退化为确定选择。','降低 Top-P，记录候选集合和熵的变化。','抽样 1000 次，比较经验频率与理论概率。']}
 };
 const c=configs[slug];if(!c||document.getElementById('lab-notebook'))return;
 const key='caius.lab.notebook.'+slug;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
 const section=document.createElement('section');section.id='lab-notebook';section.className='lab-notebook panel';section.innerHTML='<div class="eyebrow">LAB NOTEBOOK / '+slug.toUpperCase()+'</div><h2>'+c.title+'</h2><p>'+c.goal+'</p><div class="notebook-grid"><div><h3>实验清单</h3><div class="notebook-steps">'+c.steps.map((x,i)=&gt;'<label><input type="checkbox" data-step="'+i+'"><span><b>0'+(i+1)+'</b>'+x+'</span></label>').join('')+'</div><div class="notebook-progress"><span id="notebookProgress">0 / '+c.steps.length+' 完成</span><span id="notebookRuns">运行记录 0 次</span></div></div><div><h3>我的观察</h3><textarea id="notebookText" rows="6" placeholder="记录你观察到的变化、一个疑问，或下一步想验证的假设……"></textarea><div class="notebook-actions"><button class="button primary" id="notebookSave" type="button">保存实验记录</button><button class="button" id="notebookClear" type="button">清空</button><span id="notebookStatus" aria-live="polite"></span></div></div></div>';
 const anchor=document.querySelector('.lab-extra')||document.querySelector('.source');anchor?.before(section);
 const boxes=[...section.querySelectorAll('[data-step]')],text=$d('notebookText'),progress=$d('notebookProgress'),runs=$d('notebookRuns'),status=$d('notebookStatus');boxes.forEach((b,i)=&gt;b.checked=!!saved.steps?.[i]);text.value=saved.text||'';
 const render=()=&gt;{const done=boxes.filter(x=&gt;x.checked).length;progress.textContent=done+' / '+boxes.length+' 完成';runs.textContent='运行记录 '+(saved.runs||0)+' 次';section.classList.toggle('notebook-complete',done===boxes.length)};
 boxes.forEach(b=&gt;b.addEventListener('change',()=&gt;{saved.steps=boxes.map(x=&gt;x.checked);saved.runs=(saved.runs||0)+1;try{localStorage.setItem(key,JSON.stringify(saved))}catch{}render()}));
 $('notebookSave').addEventListener('click',()=&gt;{saved.steps=boxes.map(x=&gt;x.checked);saved.text=text.value;saved.runs=(saved.runs||0)+1;try{localStorage.setItem(key,JSON.stringify(saved))}catch{}status.textContent='已保存 '+new Date().toLocaleTimeString();render()});
 $('notebookClear').addEventListener('click',()=&gt;{saved={};boxes.forEach(x=&gt;x.checked=false);text.value='';try{localStorage.removeItem(key)}catch{}status.textContent='已清空';render()});render();
}
setTimeout(addNotebook,0);
</n;j++){let></tr'+(keepset.has(i)?'></tr'+(chosen.has(x.e)?'></tr'+(i===q?'></circle></polyline></line></line>