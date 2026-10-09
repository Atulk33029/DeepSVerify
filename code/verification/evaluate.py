"""Validation threshold is written once; test uses its frozen value and model hash."""
import argparse,json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve,precision_recall_curve
from .data import PairDataset,validate_pairs,load_canonical
from .model import SiameseResNet18
from .metrics import select_threshold,metrics
from .runtime import sha256,write_json


def load_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=SiameseResNet18(False,ck['config']['normalize_embeddings']).to(device)
    model.load_state_dict(ck['model']);model.eval()
    return model,ck


def score(model,frame,images,device,batch_size=64):
    # Encode every image once, then form distances. This equals Siamese eval()
    # and avoids thousands of repeated forwards for shared reference images.
    ids=sorted(set(frame.image_1_id)|set(frame.image_2_id));vectors={}
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            group=ids[start:start+batch_size]
            batch=torch.stack([load_canonical(Path(images)/f'{i}.png') for i in group]).to(device)
            z=model.encode(batch).cpu()
            vectors.update(zip(group,z))
    return np.array([float(torch.linalg.vector_norm(vectors[a]-vectors[b])) for a,b in zip(frame.image_1_id,frame.image_2_id)])


def figures(frame,threshold,destination):
    y=frame.pair_label.to_numpy();d=frame.distance.to_numpy();m=metrics(y,d,threshold)
    fig,ax=plt.subplots();im=ax.imshow(m['confusion_matrix'],cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(m['confusion_matrix'][i][j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Forged','Genuine'],yticklabels=['Forged','Genuine'],xlabel='Predicted',ylabel='Actual');fig.colorbar(im);fig.tight_layout();fig.savefig(destination/'confusion_matrix.png');plt.close(fig)
    x,tpr,_=roc_curve(y,-d);fig,ax=plt.subplots();ax.plot(x,tpr);ax.set(xlabel='False acceptance rate',ylabel='True genuine acceptance rate',title='ROC');fig.savefig(destination/'roc.png');plt.close(fig)
    precision,recall,_=precision_recall_curve(y,-d);fig,ax=plt.subplots();ax.plot(recall,precision);ax.set(xlabel='Recall (genuine)',ylabel='Precision (genuine)');fig.savefig(destination/'precision_recall.png');plt.close(fig)
    fig,ax=plt.subplots()
    for label,name in [(1,'Genuine'),(0,'Forged')]:ax.hist(d[y==label],bins=50,alpha=.5,label=name,density=True)
    ax.axvline(threshold,color='black',linestyle='--',label='Frozen threshold');ax.set(xlabel='Euclidean distance',ylabel='Density');ax.legend();fig.savefig(destination/'distance_distributions.png');plt.close(fig)


def evaluate(root,images,run,split):
    root,run=Path(root),Path(run);checkpoint=run/'best.pt';frozen=run/'threshold.json'
    dest=run/('validation' if split=='val' else 'test')
    if (dest/'COMPLETE').exists():raise FileExistsError('Evaluation is complete; preserve it and use a new run for a new experiment')
    frame=pd.read_csv(root/f'pairs/{split}_pairs.csv');image_map=pd.read_csv(root/'reports/phase3a_image_split_map.csv')
    validate_pairs(frame,image_map,split)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,ck=load_model(checkpoint,device)
    for relative,digest in ck['provenance'].items():
        if sha256(root/relative)!=digest:raise ValueError(f'Training provenance changed: {relative}')
    if split=='test' or frozen.exists():
        record=json.loads(frozen.read_text())
        if record['selected_on']!='val' or record['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Threshold/checkpoint mismatch')
        threshold=record['threshold']
    frame['distance']=score(model,frame,images,device)
    if split=='val' and not frozen.exists():
        threshold=select_threshold(frame.pair_label,frame.distance)
        write_json(frozen,dict(threshold=threshold,selected_on='val',method='nearest empirical FAR/FRR crossing',checkpoint_sha256=sha256(checkpoint),validation_pairs_sha256=sha256(root/'pairs/val_pairs.csv')))
    if dest.exists():
        dest.rename(dest.with_name(dest.name+'.interrupted_'+str(time.time_ns())))
    dest.mkdir(parents=True)
    frame['threshold']=threshold;frame['prediction']=(frame.distance<=threshold).astype(int)
    for side in [1,2]:frame[f'image_{side}_path']=frame[f'image_{side}_id'].map(lambda i:str(Path(images)/f'{i}.png'))
    frame.to_csv(dest/'scored_pairs.csv',index=False)
    report=dict(status='MEASURED',positive_class='genuine',overall=metrics(frame.pair_label,frame.distance,threshold),datasets={name:metrics(g.pair_label,g.distance,threshold) for name,g in frame.groupby('dataset')},checkpoint_sha256=sha256(checkpoint),pairs_sha256=sha256(root/f'pairs/{split}_pairs.csv'))
    write_json(dest/'metrics.json',report);figures(frame,threshold,dest)
    errors=frame[frame.prediction!=frame.pair_label].copy();errors['error_type']=np.where(errors.prediction==1,'false_acceptance','false_rejection');errors.to_csv(dest/'errors.csv',index=False)
    writers=frame.assign(error=frame.prediction!=frame.pair_label).groupby(['dataset','global_writer_id']).agg(n=('pair_id','size'),errors=('error','sum'));writers['error_rate']=writers.errors/writers.n;writers.sort_values('error_rate',ascending=False).to_csv(dest/'difficult_writers.csv')
    if not errors.empty:
        fig,axes=plt.subplots(min(8,len(errors)),2,figsize=(10,2*min(8,len(errors))),squeeze=False)
        for axes_row,(_,r) in zip(axes,errors.head(8).iterrows()):
            for j,ax in enumerate(axes_row):
                ax.imshow(plt.imread(r[f'image_{j+1}_path']),cmap='gray');ax.axis('off');ax.set_title(f'{r.global_writer_id}: {r.error_type}\nd={r.distance:.4f}, t={threshold:.4f}',fontsize=8)
        fig.tight_layout();fig.savefig(dest/'error_examples.png');plt.close(fig)
    write_json(dest/'error_analysis_report.json',dict(status='MEASURED',errors=len(errors),false_acceptances=int((errors.prediction==1).sum()),false_rejections=int((errors.prediction==0).sum())))
    (dest/'COMPLETE').write_text('Evaluation and all figures completed\n')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--images');p.add_argument('--run',required=True);p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args();root=Path(a.root).resolve();evaluate(root,a.images or root/'preprocessed/images',a.run,a.split)
if __name__=='__main__':main()
