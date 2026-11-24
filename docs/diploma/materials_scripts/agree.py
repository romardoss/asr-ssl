import csv
from collections import Counter
def load(p):
    out={}
    for r in csv.DictReader(l.replace("\0","") for l in open(p,encoding="utf-8",errors="replace")):
        a=(r.get("index") or "").strip(); m=(r.get("mark") or "").strip()
        if a.isdigit() and m.isdigit(): out[int(a)]=int(m)
    return out
g=load('google_gemma-4-26b-a4b-it-02/marks.csv'); j=load('typesafe_jev-1.14-02/marks.csv')
k=sorted(set(g)&set(j)); n=len(k)
ex=sum(g[i]==j[i] for i in k); w1=sum(abs(g[i]-j[i])<=1 for i in k)
pg=Counter(g[i] for i in k); pj=Counter(j[i] for i in k)
pe=sum(pg[m]*pj[m] for m in range(6))/n/n; po=ex/n
# quadratic weighted kappa
num=sum((g[i]-j[i])**2 for i in k); den=sum(pg[a]*pj[b]*(a-b)**2 for a in range(6) for b in range(6))/n
bg=sum(g[i]<=2 for i in k); bj=sum(j[i]<=2 for i in k); both=sum(g[i]<=2 and j[i]<=2 for i in k)
print('n',n,'exact',po,'within1',w1/n,'kappa',(po-pe)/(1-pe),'qwk',1-num/den,'poor g',bg,'poor j',bj,'both',both)
