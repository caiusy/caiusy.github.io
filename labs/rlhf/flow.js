'use strict';
(()=&gt;{
const q=id=&gt;document.getElementById(id),val=id=&gt;+q(id).value;
const sigmoid=z=&gt;1/(1+Math.exp(-z)),logit=p=&gt;Math.log(p/(1-p));
let chosen='A',withZ=0,withoutZ=0,steps=0,withHistory=[],withoutHistory=[],timer=null;
function renderPreference(){
 const a=val('scoreA'),b=val('scoreB'),pa=sigmoid(a-b),pc=chosen==='A'?pa:1-pa,loss=-Math.log(pc),reward=chosen==='A'?a:b,baseline=val('value'),advantage=reward-baseline;
 q('scoreAV').textContent=a.toFixed(1);q('scoreBV').textContent=b.toFixed(1);q('valueV').textContent=baseline.toFixed(1);
 q('chooseA').setAttribute('aria-pressed',String(chosen==='A'));q('chooseB').setAttribute('aria-pressed',String(chosen==='B'));
 q('prefProb').textContent=(pc*100).toFixed(1)+'%';q('prefLoss').textContent=loss.toFixed(3);
 q('prefBars').innerHTML='<div class="bar-row"><span>偏好 A</span><span class="track"><span class="fill" style="width:'+(pa*100)+'%;background:#58d8b2"></span></span><output>'+(pa*100).toFixed(1)+'%</output></div><div class="bar-row"><span>偏好 B</span><span class="track"><span class="fill" style="width:'+((1-pa)*100)+'%;background:#ffbf72"></span></span><output>'+((1-pa)*100).toFixed(1)+'%</output></div>';
 q('prefExplain').textContent='你选择了回答 '+chosen+'。'+(pc&gt;=.5?'奖励模型当前更倾向于你的选择；':'奖励模型当前更倾向于另一个回答；')+'交叉熵损失为 '+loss.toFixed(3)+'；对同一标注，正确回答概率越高，损失越低。';
 q('rewardV').textContent=reward.toFixed(2);q('advantageV').textContent=(advantage&gt;=0?'+':'')+advantage.toFixed(2);
 q('advantageExplain').textContent='R = '+reward.toFixed(2)+'，V = '+baseline.toFixed(2)+'，所以 A_demo = '+advantage.toFixed(2)+'。'+(advantage&gt;0?'本次结果高于基线，PPO 会倾向增加该动作概率。':advantage&lt;0?'本次结果低于基线，PPO 会倾向降低该动作概率。':'本次结果等于基线，PPO 项没有偏好方向。');
}
function stop(){if(timer){clearInterval(timer);timer=null}q('policyRun').textContent='自动运行'}
function objective(z,regularized){
 const p=sigmoid(z),old=val('pold'),ref=val('pref'),adv=val('adv'),clip=val('clip'),beta=val('kl'),r=p/old,clipped=Math.min(1+clip,Math.max(1-clip,r)),ppo=Math.min(r*adv,clipped*adv);
 const kl=p*Math.log(p/ref)+(1-p)*Math.log((1-p)/(1-ref));
 return {p,r,ppo,kl,result:ppo-(regularized?beta*kl:0)};
}
function policyChart(){
 const left=40,right=582,top=20,bottom=207,n=Math.max(2,withHistory.length),x=i=&gt;left+i/(n-1)*(right-left),y=v=&gt;bottom-v*(bottom-top);
 const line=(values,color)=&gt;'<polyline fill="none" stroke="'+color+'" stroke-width="3" points="'+values.map((v,i)=>x(i).toFixed(1)+','+y(v).toFixed(1)).join(' ')+'">';
 const ref=val('pref');q('policyChart').innerHTML='<line x1="'+left+'" y1="'+bottom+'" x2="'+right+'" y2="'+bottom+'" stroke="#45617b"><line x1="'+left+'" y1="'+top+'" x2="'+left+'" y2="'+bottom+'" stroke="#45617b"><line x1="'+left+'" y1="'+y(ref)+'" x2="'+right+'" y2="'+y(ref)+'" stroke="#72c7ff" stroke-dasharray="5 5"><text x="40" y="17" fill="#abc1d5" font-size="11">1.0</text><text x="40" y="220" fill="#abc1d5" font-size="11">0</text><text x="528" y="225" fill="#abc1d5" font-size="11">step '+steps+'</text>'+line(withHistory.length===1?[withHistory[0],withHistory[0]]:withHistory,'#58d8b2')+line(withoutHistory.length===1?[withoutHistory[0],withoutHistory[0]]:withoutHistory,'#ffbf72');
}
function renderPolicy(){const a=objective(withZ,true),b=objective(withoutZ,false);q('learningRateV').textContent=val('learningRate').toFixed(2);q('policyStepV').textContent=steps;q('policyWithV').textContent=(a.p*100).toFixed(1)+'%';q('policyWithoutV').textContent=(b.p*100).toFixed(1)+'%';policyChart();
 q('policyNote').textContent=steps===0?'两条轨迹都从 π_old 开始；点击训练 1 步后才发生更新。':`第 ${steps} 步：有 KL 的策略对参考分布 KL=${a.kl.toFixed(4)}；无 KL 的对参考分布 KL=${b.kl.toFixed(4)}。当前两条曲线使用同一优势和 PPO clip。`;
}
function reset(){stop();withZ=withoutZ=logit(val('pold'));steps=0;withHistory=[val('pold')];withoutHistory=[val('pold')];renderPolicy()}
function step(){if(steps&gt;=120){stop();return}const h=.0001,eta=val('learningRate'),grad=(z,regularized)=&gt;(objective(z+h,regularized).result-objective(z-h,regularized).result)/(2*h);const ga=grad(withZ,true),gb=grad(withoutZ,false);withZ=Math.max(-8,Math.min(8,withZ+eta*ga));withoutZ=Math.max(-8,Math.min(8,withoutZ+eta*gb));steps++;withHistory.push(sigmoid(withZ));withoutHistory.push(sigmoid(withoutZ));renderPolicy();if(steps&gt;=120)stop()}
for(const id of ['scoreA','scoreB','value'])q(id).addEventListener('input',renderPreference);
q('chooseA').addEventListener('click',()=&gt;{chosen='A';renderPreference()});q('chooseB').addEventListener('click',()=&gt;{chosen='B';renderPreference()});
q('sendAdv').addEventListener('click',()=&gt;{const reward=chosen==='A'?val('scoreA'):val('scoreB');q('adv').value=(reward-val('value')).toFixed(1);q('adv').dispatchEvent(new Event('input'));q('policyNote').textContent='已把教学优势送入 PPO，并重置策略轨迹。';q('adv').scrollIntoView({behavior:'smooth',block:'center'})});
q('policyStep').addEventListener('click',step);q('policyReset').addEventListener('click',reset);q('policyRun').addEventListener('click',()=&gt;{if(timer){stop();return}timer=setInterval(step,90);q('policyRun').textContent='暂停'});
for(const id of ['adv','clip','kl','pold','pref'])q(id).addEventListener('input',reset);
q('learningRate').addEventListener('input',renderPolicy);
renderPreference();reset();
})();
</line></line></line></polyline>