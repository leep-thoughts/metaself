import numpy as np, pandas as pd
from real.tune_accuracy import *
df=add_covariates(load(),np.random.default_rng(0))
rows=[];mix=[]
for rep in range(20):
    rng=np.random.default_rng(rep)
    probs=df[['task','problem_idx']].drop_duplicates()
    probs['fold']=rng.integers(2,size=len(probs))
    d=df.merge(probs,on=['task','problem_idx'])
    for judge,g in d.groupby('judge'):
        for cov in COVARIATES:
            h=g.dropna(subset=[cov])
            tr,te=h[h.fold==0].reset_index(drop=True),h[h.fold==1].reset_index(drop=True)
            edges=np.quantile(tr[cov],np.linspace(0,1,N_BINS+1)); edges=np.unique(edges)
            if len(edges)!=N_BINS+1: continue
            btr=np.clip(np.searchsorted(edges,tr[cov],side='right')-1,0,N_BINS-1)
            bte=np.clip(np.searchsorted(edges,te[cov],side='right')-1,0,N_BINS-1)
            gp=tr.groupby(btr); bp,ba=gp.a_correct.mean().to_numpy(),gp.acc.mean().to_numpy()
            nte=np.bincount(bte,minlength=N_BINS)
            if (nte==0).any() or len(bp)!=N_BINS: continue
            tp=tr.a_correct.mean()
            for name,w in (('low',extreme(bp,ba,tp,1)),('high',extreme(bp,ba,tp,-1))):
                if w is None: continue
                W=(w/nte)[bte]
                s=weighted_stats(te,W); W=W/W.sum()
                rows.append(dict(rep=rep,judge=judge,cov=cov,point=name,**s))
                mix.append(dict(cov=cov,point=name,**{t:W[te.task==t].sum() for t in ['math500','mmlu','mbpp-plus']}))
r=pd.DataFrame(rows)
m=r.groupby(['cov','judge','point'])[['p','actual_accuracy']].mean().unstack('point')
out=pd.DataFrame({'acc_range':m['actual_accuracy']['high']-m['actual_accuracy']['low'],'p_gap':m['p']['high']-m['p']['low']})
print(out.round(3).unstack('cov').to_string())
print(out.groupby('cov').mean().round(3))
print(r.groupby(['cov','point'])[['p','actual_accuracy','panickssery','equal_opportunity','hspp','roytburg']].mean().round(3))
print(pd.DataFrame(mix).groupby(['cov','point']).mean().round(2))
