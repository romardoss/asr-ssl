import re,pandas as pd,numpy as np
def ed(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        c=[i]
        for j,y in enumerate(b,1): c.append(min(p[j]+1,c[j-1]+1,p[j-1]+(x!=y)))
        p=c
    return p[-1]
class J:
    @staticmethod
    def wer(a,b): a=a.split(); return ed(a,b.split())/len(a)
jiwer=J
P=re.compile(r"[^\w\s']+",re.U);S=re.compile(r"\s+")
def n(t): t=str(t).lower().replace("’","'"); return S.sub(" ",P.sub(" ",t)).strip()
runs=["baseline/original-whisper-medium","baseline/common-voice-only-baseline","student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01","student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01","student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01","student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01","whisper-m-fixed-leaks-5-epochs-epoch04-01","whisper-m-fixed-leaks-10-epochs-epoch09-01"]
for r in runs:
    d=pd.read_csv(f"evaluation/{r}/predictions.csv",dtype=str,keep_default_na=False)
    sc=[];ds=[]
    for a,b,s in zip(d.reference,d.hypothesis,d.dataset):
        a=n(a)
        if not a: continue
        sc.append(jiwer.wer(a,n(b))*100); ds.append(s)
    sc=np.array(sc); ds=np.array(ds)
    print(r, len(sc), "median %.1f mean %.1f  =0: %.1f%%  >=100: %.1f%%  >200: %d"%(np.median(sc),sc.mean(),(sc==0).mean()*100,(sc>=100).mean()*100,(sc>200).sum()))
    for s in ["common_voice","fleurs","ukr_dialects","cv10_test"]:
        x=sc[ds==s]; print("   ",s,len(x),"median %.1f =0 %.1f%% >=100 %.1f%%"%(np.median(x),(x==0).mean()*100,(x>=100).mean()*100))
    print("   empty refs dropped:", len(d)-len(sc))
