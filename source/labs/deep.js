'use strict';
(()=>{
const $d=id=>document.getElementById(id), num=id=>+$d(id).value;
const bar=(label,value,color,scale=1)=>'<div class="deep-bar"><span>'+label+'</span><span class="track"><span class="fill" style="width:'+(Math.max(0,Math.min(100,value/scale*100)))+'%;background:'+color+'"></span></span><output>'+(value*100).toFixed(0)+'%</output></div>';
const chart=(id,series,range,{width=600,height=220,zero=false,marker=null}={})=>{
 const lo=zero?Math.min(0,range[0]):range[0],hi=range[1],left=36,right=width-16,top=20,bottom=height-32;
 const px=(x,n)=>left+x/(n-1)*(right-left),py=y=>bottom-(y-lo)/(hi-lo||1)*(bottom-top);
 let html='<line x1="'+left+'" y1="'+bottom+'" x2="'+right+'" y2="'+bottom+'" stroke="#46617a"/><line x1="'+left+'" y1="'+top+'" x2="'+left+'" y2="'+bottom+'" stroke="#46617a"/><text x="40" y="16" fill="#abc1d5" font-size="11">'+hi.toFixed(2)+'</text><text x="40" y="'+(bottom-5)+'" fill="#abc1d5" font-size="11">'+lo.toFixed(2)+'</text>';
 for(const s of series){if(!s.values.length)continue;html+='<polyline fill="none" stroke="'+s.color+'" stroke-width="3" points="'+s.values.map((v,i)=>px(i,s.values.length).toFixed(1)+','+py(v).toFixed(1)).join(' ')+'"/>'}
 if(marker){html+='<circle cx="'+px(marker.index,marker.total)+'" cy="'+py(marker.value)+'" r="6" fill="'+(marker.color||'#ffbf72')+'"/>'}
 $d(id).innerHTML=html;
};
function attention(){const q=num('query')-1,t=num('temp'),causal=$d('mask').value==='causal',raw=K.map(k=>Q[q][0]*k[0]+Q[q][1]*k[1]),scaled=raw.map(v=>v/Math.SQRT2);
 const probs=mask=>{let z=scaled.map((v,i)=>mask&&i>q?-Infinity:v/t),max=Math.max(...z.filter(Number.isFinite)),ex=z.map(v=>Number.isFinite(v)?Math.exp(v-max):0),sum=ex.reduce((a,b)=>a+b,0);return ex.map(x=>x/sum)};
 const current=probs(causal),c=probs(true),all=probs(false);
 $d('deepAttentionRows').innerHTML=words.map((w,i)=>'<tr'+(i===q?' class="picked"':'')+'><td>K'+(i+1)+'「'+w+'」</td><td>'+raw[i].toFixed(3)+'</td><td>'+scaled[i].toFixed(3)+'</td><td>'+(causal&&i>q?'−∞':(scaled[i]/t).toFixed(3))+'</td><td>'+(current[i]*100).toFixed(1)+'%</td></tr>').join('');
 $d('deepAttentionFormula').textContent='当前 Q'+(q+1)+'：softmax(['+scaled.map((v,i)=>causal&&i>q?'−∞':(v/t).toFixed(2)).join(', ')+']) = ['+current.map(v=>v.toFixed(3)).join(', ')+']';
 $d('deepAttentionCompare').innerHTML='<b>因果掩码</b>'+words.map((w,i)=>bar(w,c[i],'#58d8b2')).join('')+'<b>无掩码</b>'+words.map((w,i)=>bar(w,all[i],'#72c7ff')).join('');
}
function kv(){const L=num('ctx'),ly=num('layers'),h=num('heads'),bytes=num('bytes'),lat=num('latent'),batch=num('deepBatch'),budget=num('deepBudget');
 const fullPer=ly*2*h*128*bytes*batch/2**30,latPer=ly*lat*bytes*batch/2**30,full=L*fullPer,compact=L*latPer;
 $d('deepBatchV').textContent=batch;$d('deepBudgetV').textContent=budget+' GiB';
 $d('deepKvNumbers').innerHTML='<div class="metric"><strong>'+full.toFixed(2)+' GiB</strong><small>普通 KV · 当前长度</small></div><div class="metric"><strong>'+compact.toFixed(2)+' GiB</strong><small>latent · 当前长度</small></div><div class="metric"><strong>'+Math.floor(budget/fullPer).toLocaleString()+'</strong><small>普通 KV 理论 token 上限 / 序列</small></div>';
 const max=Math.max(full,compact,budget)*1.12,x=Array.from({length:25},(_,i)=>i/24*L);chart('deepKvChart',[{color:'#ffbf72',values:x.map(v=>v*fullPer)},{color:'#58d8b2',values:x.map(v=>v*latPer)},{color:'#ff788d',values:x.map(()=>budget)}],[0,max]);
 $d('deepKvNote').textContent='在 '+batch+' 条同长序列、'+budget+' GiB 的专用预算下，普通 KV 理论每条上限约 '+Math.floor(budget/fullPer).toLocaleString()+' token；简化 latent 缓存约 '+Math.floor(budget/latPer).toLocaleString()+' token。实际可用长度仍受模型上下文上限和其他显存占用限制。';
}
function moe(){let N=num('tokens'),E=num('experts'),top=num('top'),token=Math.min(num('deepToken'),N-1);$d('deepToken').max=N-1;$d('deepToken').value=token;$d('deepTokenV').textContent=token;
 const scores=Array.from({length:E},(_,e)=>{let affinity=2.2/(e+1)+.6*Math.sin(token*1.7+e*2.3)+.25*Math.cos(token*.47-e);return {e,affinity,bias:bias[e]||0,total:affinity+(bias[e]||0)}});const chosen=new Set([...scores].sort((a,b)=>b.total-a.total).slice(0,top).map(x=>x.e));
 $d('deepMoeRows').innerHTML=scores.map(x=>'<tr'+(chosen.has(x.e)?' class="picked"':'')+'><td>Expert '+(x.e+1)+'</td><td>'+x.affinity.toFixed(3)+'</td><td>'+x.bias.toFixed(3)+'</td><td>'+x.total.toFixed(3)+'</td><td>'+(chosen.has(x.e)?'✓ 选中':'—')+'</td></tr>').join('');
 $d('deepMoeChoice').textContent='Token '+token+' 当前选择 Expert '+[...chosen].map(e=>e+1).join('、')+'。偏置并不修改亲和分数本身，只在排序前相加。';
}
let moeHistory=[];function moeChart(){let cv=+$d('cv').textContent,r=+$d('round').textContent;if(r===0)moeHistory=[cv];else if(moeHistory.length<=r)moeHistory.push(cv);else moeHistory[r]=cv;let max=Math.max(.2,...moeHistory)*1.15;chart('deepMoeChart',[{color:'#58d8b2',values:moeHistory.length===1?[cv,cv]:moeHistory}],[0,max],{height:180});}
function rlhf(){let A=num('adv'),r=num('ratio'),eps=num('clip'),beta=num('kl');const val=x=>{let raw=x*A,c=Math.max(1-eps,Math.min(1+eps,x))*A,ppo=Math.min(raw,c),proxy=.5*Math.log(x)**2;return {raw,c,ppo,proxy,final:ppo-beta*proxy}};
 const points=Array.from({length:131},(_,i)=>.5+i*.01),raw=points.map(x=>val(x).raw),ppo=points.map(x=>val(x).ppo),current=val(r),range=[Math.min(...raw,...ppo)-.2,Math.max(...raw,...ppo)+.2];chart('deepRlhfChart',[{color:'#72c7ff',values:raw},{color:'#58d8b2',values:ppo}],range,{height:240,marker:{index:Math.round((r-.5)*100),total:131,value:current.ppo}});
 $d('deepRlhfLedger').innerHTML=[['未裁剪 r·A',current.raw],['裁剪候选 clip(r)·A',current.c],['PPO min',current.ppo],['教学偏移代理 ½(log r)²',current.proxy],['β × 代理惩罚',beta*current.proxy],['最终示意目标',current.final]].map(([name,x])=>'<div><span>'+name+'</span><b>'+x.toFixed(4)+'</b></div>').join('');
 let active=A>0&&r>1+eps||A<0&&r<1-eps;$d('deepRlhfRegion').textContent=active?'当前处于 PPO 裁剪平台：在这个单动作示意中，PPO 项对 r 的局部斜率为 0。惩罚项仍随 r 变化。':'当前 PPO 项仍随 r 改变。区间边界为 ['+(1-eps).toFixed(2)+', '+(1+eps).toFixed(2)+']；是否进入平台还取决于优势符号。';
}
function rope(){const m=num('m'),n=num('n'),theta=num('theta'),shift=num('deepShift'),q=[1,.35],k=[.72,.92];const dot=d=>{let angle=-d*theta,kr=[k[0]*Math.cos(angle)-k[1]*Math.sin(angle),k[0]*Math.sin(angle)+k[1]*Math.cos(angle)];return q[0]*kr[0]+q[1]*kr[1]};let d=m-n,baseline=dot(d),m2=m+shift,n2=n+shift,translated=dot(m2-n2);
 $d('deepShiftV').textContent=shift;$d('deepRopePairs').innerHTML='<div><span>原位置 (m,n)</span><b>('+m+', '+n+')</b></div><div><span>平移后位置</span><b>('+m2+', '+n2+')</b></div><div><span>两组位置差</span><b>'+d+' / '+(m2-n2)+'</b></div><div><span>原点积 / 平移后</span><b>'+baseline.toFixed(6)+' / '+translated.toFixed(6)+'</b></div>';
 $d('deepRopeProof').textContent='(m+'+shift+') − (n+'+shift+') = m−n = '+d+'。在相同 q、k、θ 下，点积差仅为数值舍入误差。';let xs=Array.from({length:129},(_,i)=>i-64),ys=xs.map(dot),min=Math.min(...ys),max=Math.max(...ys);chart('deepRopeChart',[{color:'#58d8b2',values:ys}],[min-.1,max+.1],{marker:d>=-64&&d<=64?{index:d+64,total:129,value:baseline}:null});
}
let sampleCounts=[0,0,0,0,0,0],sampleTotal=0;function sampling(){let t=num('temp'),Kmax=num('topk'),p=num('topp'),max=Math.max(...logits.map(v=>v/t)),ex=logits.map(v=>Math.exp(v/t-max)),z=ex.reduce((a,b)=>a+b,0),base=ex.map(v=>v/z),order=base.map((v,i)=>({i,v})).sort((a,b)=>b.v-a.v),top=new Set(order.slice(0,Kmax).map(x=>x.i)),kept=[],acc=0;for(const item of order.slice(0,Kmax)){kept.push(item.i);acc+=item.v;if(acc>=p)break}const finalSum=kept.reduce((a,i)=>a+base[i],0),keepSet=new Set(kept);
 $d('deepSamplingRows').innerHTML=tokens.map((word,i)=>'<tr'+(keepSet.has(i)?' class="picked"':'')+'><td>'+word+'</td><td>'+logits[i].toFixed(1)+'</td><td>'+(base[i]*100).toFixed(1)+'%</td><td>'+(top.has(i)?(base[i]*100).toFixed(1)+'%':'—')+'</td><td>'+(keepSet.has(i)?(base[i]*100).toFixed(1)+'%':'—')+'</td><td>'+(distribution[i]*100).toFixed(1)+'%</td></tr>').join('');
 $d('deepSamplingNote').textContent='Top-P 前累计保留质量为 '+(finalSum*100).toFixed(1)+'%。末列将保留候选除以这部分质量，使概率重新合计为 100%。';sampleFrequency();
}
function sampleFrequency(){let dist=distribution,total=sampleTotal;$d('deepSamplingFreq').innerHTML=tokens.map((word,i)=>'<div class="deep-bar"><span>'+word+'</span><span class="track"><span class="fill" style="width:'+(total?sampleCounts[i]/total*100:0)+'%;background:#58d8b2"></span></span><output>'+(total?(sampleCounts[i]/total*100).toFixed(0)+'%':'—')+'</output></div>').join('');let tv=total?sampleCounts.reduce((s,v,i)=>s+Math.abs(v/total-dist[i]),0)/2:0;$d('deepSamplingStat').textContent=total?'累计 '+total+' 次独立抽样；经验分布与理论分布的总变差距离约 '+tv.toFixed(3)+'。样本数增大时通常更接近，但单次实验不保证单调。':'尚未进行批量抽样。先观察理论概率，再点击 100 次或 1000 次。';}
function drawSamples(n){for(let j=0;j<n;j++){let u=Math.random(),c=0,i=distribution.findIndex(v=>(c+=v)>=u);sampleCounts[i<0?distribution.length-1:i]++;sampleTotal++}sampleFrequency()}
const slug=location.pathname.split('/').filter(Boolean).at(-1);const watch=(ids,fn)=>ids.forEach(id=>$d(id).addEventListener('input',fn));
if(slug==='attention'){watch(['query','temp','mask'],attention);attention()}
if(slug==='kv-cache'){watch(['ctx','layers','heads','bytes','latent','deepBatch','deepBudget'],kv);kv()}
if(slug==='moe'){watch(['tokens','experts','top','gamma'],()=>{moe();moeHistory=[];moeChart()});$d('deepToken').addEventListener('input',moe);$d('step').addEventListener('click',()=>{moe();moeChart()});$d('reset').addEventListener('click',()=>{moeHistory=[];moe();moeChart()});moe();moeChart()}
if(slug==='rlhf'){watch(['adv','ratio','clip','kl'],rlhf);for(const [id,a,r] of [['deepPositive','1.2','1.5'],['deepNegative','-1.2','0.6'],['deepInside','1.2','1.0']])$d(id).addEventListener('click',()=>{$d('adv').value=a;$d('ratio').value=r;$d('adv').dispatchEvent(new Event('input'));$d('ratio').dispatchEvent(new Event('input'))});rlhf()}
if(slug==='rope'){watch(['m','n','theta','deepShift'],rope);rope()}
if(slug==='sampling'){watch(['temp','topk','topp'],()=>{sampleCounts=[0,0,0,0,0,0];sampleTotal=0;sampling()});$d('deepDraw100').addEventListener('click',()=>drawSamples(100));$d('deepDraw1000').addEventListener('click',()=>drawSamples(1000));$d('deepDrawReset').addEventListener('click',()=>{sampleCounts=[0,0,0,0,0,0];sampleTotal=0;sampleFrequency()});sampling()}
})();
