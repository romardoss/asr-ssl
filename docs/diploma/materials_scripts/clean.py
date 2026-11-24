import re,csv,glob,pandas as pd
from pathlib import Path
P=re.compile(r"[^\w\s']+",re.U);S=re.compile(r"\s+")
def n(t): t=str(t).lower().replace("’","'"); return S.sub(" ",P.sub(" ",t)).strip()
def ed(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        c=[i]
        for j,y in enumerate(b,1): c.append(min(p[j]+1,c[j-1]+1,p[j-1]+(x!=y)))
        p=c
    return p[-1]
def rates(R,H):
    we=wn=ce=cn=0
    for r,h in zip(R,H):
        r=n(r);h=n(h)
        if not r: continue
        we+=ed(r.split(),h.split()); wn+=len(r.split()); ce+=ed(r,h); cn+=len(r)
    return we/wn*100, ce/cn*100
cleaned=set(pd.read_csv('datasets/common-voice/raw/test_cleaned.tsv',sep='\t',dtype=str,quoting=csv.QUOTE_NONE,keep_default_na=False)['path'])
print(len(cleaned))
for f in sorted(glob.glob('evaluation/**/predictions.csv',recursive=True)):
    d=pd.read_csv(f,dtype=str,keep_default_na=False); v=d[d.dataset=='common_voice']
    c=v[v.audio_path.map(lambda p:Path(p).name).isin(cleaned)]
    fw,fc=rates(v.reference,v.hypothesis); cw,cc=rates(c.reference,c.hypothesis)
    print(f"{f[11:-16]}|{len(v)}|{len(c)}|{fw:.2f}|{cw:.2f}|{cw-fw:+.2f}|{fc:.2f}|{cc:.2f}|{cc-fc:+.2f}")
