import csv, glob
from collections import Counter
csv.field_size_limit(10**9)
dur={}
with open('datasets/voa-ukr/ukrainian_30s.csv',encoding='utf-8',newline='') as h:
    for r in csv.DictReader(h): dur[r['ID']]=float(r['duration_seconds'])
for f in sorted(glob.glob('output/pseudo-labelling/*/label_marks.csv')):
    raw=0; bad=0; latest={}
    with open(f,encoding='utf-8',errors='replace',newline='') as h:
        for r in csv.DictReader(l.replace('\0','') for l in h):
            raw+=1
            m=(r.get('mark') or '').strip()
            if not m.isdigit() or int(m)>5 or not r.get('ID'): bad+=1; continue
            latest[r['ID']]=int(m)
    c=Counter(latest.values()); n=len(latest)
    print(f,'raw rows',raw,'bad',bad,'unique IDs',n,'dups',raw-bad-n)
    H={k:0.0 for k in range(6)}
    for i,m in latest.items(): H[m]+=dur.get(i,0)
    tot=sum(H.values())/3600
    for k in range(6): print(f'  mark {k}: {c[k]} ({100*c[k]/n:.2f}%), {H[k]/3600:.1f} h')
    for s in [(2,3,4,5),(3,4,5),(4,5),(5,)]:
        cnt=sum(c[k] for k in s); h=sum(H[k] for k in s)/3600
        print('  ',s,cnt,f'{100*cnt/n:.2f}%',f'{h:.1f} h', f'{100*h/tot:.1f}% hours')
    print('  total hours',round(tot,1),'mean mark',sum(k*c[k] for k in c)/n)
