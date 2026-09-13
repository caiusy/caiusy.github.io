"""Reproduce the article's synthetic examples and check algebraic equivalence."""
import json
from pathlib import Path
import numpy as np

def softmax(x):
    e = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)

def rope(x, pos):
    # Row vectors: multiply by R(pos).T, pairwise, with distinct frequencies.
    y = np.array(x, dtype=float, copy=True)
    for r in range(x.shape[-1]//2):
        angle = pos * (0.31/(r+1))
        a, b = x[..., 2*r], x[..., 2*r+1]
        y[..., 2*r] = a*np.cos(angle)-b*np.sin(angle)
        y[..., 2*r+1] = a*np.sin(angle)+b*np.cos(angle)
    return y

K = np.array([[1,0],[0,1],[1,1],[1,-1]], float)
V = np.array([[10,0],[0,20],[6,6],[4,8]], float)
results = {}
for t, q in [(3,[1,1]),(4,[2,0])]:
    scores = np.array(q)@K[:t].T/np.sqrt(2)
    a=softmax(scores)
    results[f'kv{t}']={'scores':scores.tolist(),'weights':a.tolist(),'output':(a@V[:t]).tolist()}
C=np.array([[1,0],[0,1],[1,1],[2,1]],float)
UK=np.array([[1,2],[0,1]],float)
UV=np.diag([2.,3.])
KR=np.array([[np.cos(j*np.pi/6),np.sin(j*np.pi/6)] for j in range(1,5)])
for t,q in [(3,[1,1]),(4,[0,1])]:
    q=np.array(q,float); qr=KR[t-1]
    explicit_content=q@(C[:t]@UK).T
    latent_content=(q@UK.T)@C[:t].T
    np.testing.assert_allclose(explicit_content,latent_content,atol=1e-12)
    position=qr@KR[:t].T
    scores=(latent_content+position)/2
    a=softmax(scores)
    z=a@C[:t]
    out=z@UV
    np.testing.assert_allclose(a@(C[:t]@UV),out,atol=1e-12)
    results[f'mla{t}']={'content':latent_content.tolist(),'position':position.tolist(),'scores':scores.tolist(),'weights':a.tolist(),'latent':z.tolist(),'output':out.tolist()}
# Random multihead, multistep check, including output projection and different value dimension.
rng=np.random.default_rng(1313)
maxerr=0.
for _ in range(20):
    T,H,D,dc,dh,dr,dv=7,3,9,4,6,4,5
    c=rng.normal(size=(T,dc)); prekr=rng.normal(size=(T,dr))
    kr=np.stack([rope(prekr[j],j) for j in range(T)])
    uk=rng.normal(size=(H,dc,dh)); uv=rng.normal(size=(H,dc,dv))
    wo=rng.normal(size=(H,dv,D))
    for t in range(T):
        qc=rng.normal(size=(H,dh)); qr=rope(rng.normal(size=(H,dr)),t)
        explicit=[]; absorbed=[]
        for i in range(H):
            k=np.concatenate([c[:t+1]@uk[i],kr[:t+1]],axis=-1)
            q=np.concatenate([qc[i],qr[i]])
            a=softmax(q@k.T/np.sqrt(dh+dr))
            b=softmax(((qc[i]@uk[i].T)@c[:t+1].T+qr[i]@kr[:t+1].T)/np.sqrt(dh+dr))
            explicit.append((a@(c[:t+1]@uv[i]))@wo[i])
            absorbed.append((b@c[:t+1])@(uv[i]@wo[i]))
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
        a=np.sum(explicit,axis=0); b=np.sum(absorbed,axis=0)
        maxerr=max(maxerr,float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-11)
# Joint shifts preserve RoPE inner products with pre-rotation vectors fixed.
for _ in range(30):
    q,k=rng.normal(size=(2,6)); m,n,shift=2,8,5
    np.testing.assert_allclose(rope(q,m)@rope(k,n),rope(q,m+shift)@rope(k,n+shift),atol=1e-12)
results['verification']={'random_multihead_cases':140,'rope_shift_cases':30,'max_output_abs_error':maxerr,'status':'PASS'}
Path(__file__).with_name('results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
