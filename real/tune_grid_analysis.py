import numpy as np, pandas as pd
r=pd.read_csv('real/tune_grid_results.csv'); M=['panickssery','equal_opportunity','hspp','roytburg']
print('all',len(r),'feasible>=.8',(r.feasible>=.8).sum(),'reach',r.reachable.sum())
f=r[r.feasible>=.8]
print('held-out error among feasible: p',(f.p-f.target_p).abs().describe().round(3).to_dict())
print('acc',(f.actual_accuracy-f.target_acc).abs().describe().round(3).to_dict())
print('fail reasons', ((f.p-f.target_p).abs()>.02).sum(), ((f.actual_accuracy-f.target_acc).abs()>.02).sum(), (f.ess<100).sum())
a=r[r.reachable]
print(a[['p_sd','actual_accuracy_sd','ess','l1_from_natural']].describe().round(3).loc[['min','50%','max']])
# does each judge have a rectangle: at each p, acc range
for j,g in a.groupby('judge'):
    print(j,'nat p %.2f acc %.2f'%(g.natural_p.iloc[0],g.natural_acc.iloc[0]))
    print('  ',' '.join(f"{tp:.2f}:[{x.target_acc.min():.2f},{x.target_acc.max():.2f}]" for tp,x in g.groupby('target_p')))
# slopes: within judge OLS using achieved p, acc
rows=[]
for j,g in a.groupby('judge'):
    if g.target_p.nunique()<3 or g.acc_offset.nunique()<3: rows.append(dict(judge=j,n=len(g)));continue
    X=np.c_[np.ones(len(g)),g.p,g.actual_accuracy]
    d=dict(judge=j,n=len(g),sp=g.p.max()-g.p.min(),sa=g.actual_accuracy.max()-g.actual_accuracy.min(),rpa=np.corrcoef(g.p,g.actual_accuracy)[0,1])
    for m in M:
        b=np.linalg.lstsq(X,g[m],rcond=None)[0]; d[m+'_p'],d[m+'_a']=b[1],b[2]
    rows.append(d)
t=pd.DataFrame(rows); print(t.round(2).to_string()); print(t.drop(columns='judge').median().round(2).to_string())
