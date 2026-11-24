import csv, glob, statistics, sys
csv.field_size_limit(10**9)
dur={}
with open('datasets/voa-ukr/ukrainian_30s.csv',encoding='utf-8',newline='') as h:
    for r in csv.DictReader(h): dur[r['ID']]=float(r['duration_seconds'])
for d in sorted(glob.glob('output/pseudo-labelling/*/')):
    rows=list(csv.DictReader(open(d+'labels.csv',encoding='utf-8',newline='')))
    ids=[r['ID'] for r in rows]; u=set(ids)
    empty=sum(1 for r in rows if not r['text'].strip())
    words=[len(r['text'].split()) for r in rows]
    chars=[len(r['text']) for r in rows]
    hrs=sum(dur.get(i,0) for i in u)/3600
    missing=sum(1 for i in u if i not in dur)
    print(d,'rows',len(rows),'unique',len(u),'empty',empty,'paths',len({r['path'] for r in rows}),'hours',round(hrs,2),'notin30s',missing)
    print('  words mean %.1f median %d max %d; chars max %d; >200 words %d'%(statistics.mean(words),statistics.median(words),max(words),max(chars),sum(w>200 for w in words)))
