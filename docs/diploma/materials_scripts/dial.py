import csv, re
csv.field_size_limit(10**9)
P=re.compile(r"[^\w\s']+"); S=re.compile(r"\s+")
norm=lambda t: S.sub(" ",P.sub(" ",t.lower().replace("’","'"))).strip()
top=[r['item'] for r in csv.DictReader(open('analysis/unique-tokens/words.csv',encoding='utf-8'))][:200]; top=set(top)
for run in ['model-whisper-uk-epoch00-01','model-whisper-m-fixed-leaks-5-epochs-epoch04-01']:
    base='output/pseudo-labelling/'+run+'/'
    m={r['ID']:int(r['mark']) for r in csv.DictReader(open(base+'label_marks.csv',encoding='utf-8'))}
    g={True:[],False:[]}
    for r in csv.DictReader(open(base+'labels.csv',encoding='utf-8')):
        if r['ID'] not in m: continue
        g[bool(set(norm(r['text']).split())&top)].append(m[r['ID']])
    for k,v in g.items():
        n=len(v); print(run,'dialect' if k else 'other',n,'mean %.2f'%(sum(v)/n),'4-5 %.1f%%'%(100*sum(x>=4 for x in v)/n),'5 %.1f%%'%(100*sum(x==5 for x in v)/n))
