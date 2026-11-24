import pandas as pd, yaml, glob, os, sys
os.chdir('/home/romardoss/asr/asr_ssl/output/fine-tune')
for m in sorted(glob.glob('*/*/metrics.csv')):
    d=os.path.dirname(m)
    print('\n#####', d)
    for f in ('meta.yaml','hparams.yaml','config.yaml'):
        p=os.path.join(d,f)
        if os.path.exists(p):
            y=yaml.safe_load(open(p)) or {}
            if f=='config.yaml':
                t=y.get('training',{}); dd=y.get('data',{}); b=y.get('batch_size',{})
                print(' config: train',dd.get('train'),'dev',dd.get('dev'),'prec',t.get('precision'),'ep',t.get('max_epochs'),'lr',t.get('learning_rate'),'bs',b.get('train'),'acc',t.get('accumulate_grad_batches'),'init',t.get('init'),'val_every',t.get('val_every'))
            else:
                print(' ',f,{k:v for k,v in y.items() if k in ('init','precision','max_epochs','learning_rate','batch_size','accumulate_grad_batches','train_data','cv_train_rows','accepted_mark','note','epoch','warmup_steps','weight_decay','resumed_from','labels')})
    df=pd.read_csv(m)
    v=df[df.val_wer.notna()] if 'val_wer' in df else df.iloc[0:0]
    print(' rows',len(df),'max step',df.step.max(),'max epoch',df.epoch.max())
    if len(v):
        print(' val evals:',len(v))
        print(v[['epoch','step','val_loss','val_wer','val_cer']].to_string(index=False))
        b=v.loc[v.val_wer.idxmin()]; print(' best val_wer',b.val_wer,'at epoch',b.epoch,'step',b.step)
    if 'train_loss_epoch' in df:
        te=df[df.train_loss_epoch.notna()][['epoch','step','train_loss_epoch']]
        print(' train_loss_epoch:\n',te.to_string(index=False))
    tc=[c for c in df.columns if c.startswith('test_') and c.endswith('wer')]
    if tc:
        t=df[df[tc[0]].notna()]
        print(' test:',t[['epoch']+tc].to_string(index=False))
