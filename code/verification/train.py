"""Train on locked pairs. Restart with --resume at the last completed epoch."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from torch.utils.data import DataLoader
from .data import PairDataset,validate_pairs
from .model import SiameseResNet18,contrastive_loss
from .runtime import seed_all,environment,sha256,write_json,save_checkpoint


def train(root,images,out,cfg,resume=False):
    root,images,out=Path(root),Path(images),Path(out)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError('Run directory exists: use a new --out or --resume')
    if cfg['augmentation']!='none' or cfg['normalization']!='divide_by_255':
        raise ValueError('Only locked baseline transforms are implemented')
    seed_all(cfg['seed']);out.mkdir(parents=True,exist_ok=True)
    image_map=pd.read_csv(root/'reports/phase3a_image_split_map.csv')
    manifest=pd.read_csv(root/'reports/phase3a_split_manifest.csv')
    if manifest.global_writer_id.duplicated().any():raise ValueError('Writer split overlap')
    if image_map.groupby('global_writer_id').split.nunique().max()!=1:
        raise ValueError('An image-map writer occurs in multiple splits')
    expected=image_map.global_writer_id.map(manifest.set_index('global_writer_id').split)
    if not expected.eq(image_map.split).all():raise ValueError('Image map disagrees with locked manifest')
    splits={s:pd.read_csv(root/f'pairs/{s}_pairs.csv') for s in ['train','val']}
    datasets={}
    for s,frame in splits.items():
        validate_pairs(frame,image_map,s)
        datasets[s]=PairDataset(frame,images)
    provenance={str(p.relative_to(root)):sha256(p) for p in [root/'reports/phase3a_image_split_map.csv',root/'reports/phase3a_split_manifest.csv',root/'pairs/train_pairs.csv',root/'pairs/val_pairs.csv',root/'configs/preprocessing_config.yaml',root/'code/preprocessing/transforms.py',root/'code/preprocessing/eda.py']}
    pre_cfg=yaml.safe_load((root/'configs/preprocessing_config.yaml').read_text())
    if (pre_cfg['target']['width'],pre_cfg['target']['height'])!=(384,128):raise ValueError('Wrong image dimensions')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SiameseResNet18(cfg['pretrained'] and not resume,cfg['normalize_embeddings']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    history=[];best=float('inf');start=0
    if resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False) # only load your own run checkpoints
        if ck['config']!=cfg or ck['provenance']!=provenance:raise ValueError('Resume configuration/data changed')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        history=ck['history'];best=ck['best_loss'];start=ck['epoch']+1
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write_json(out/'config.json',cfg);write_json(out/'environment.json',environment());write_json(out/'provenance.json',provenance)
    for epoch in range(start,cfg['epochs']):
        before=time.perf_counter();losses={}
        for split in ['train','val']:
            training=split=='train';model.train(training)
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(datasets[split],batch_size=cfg['batch_size'],shuffle=training,generator=generator,
                num_workers=cfg['num_workers'],pin_memory=device.type=='cuda')
            total=0.;count=0
            with torch.set_grad_enabled(training):
                for step,(a,b,y) in enumerate(loader):
                    a,b,y=a.to(device),b.to(device),y.to(device)
                    if training:optimizer.zero_grad(set_to_none=True)
                    d=model(a,b);loss=contrastive_loss(d,y,cfg['margin'])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    if training:loss.backward();optimizer.step()
                    total+=float(loss.detach())*len(y);count+=len(y)
                    if step%100==0:print(f'epoch {epoch+1}/{cfg["epochs"]} {split} batch {step+1}/{len(loader)} loss={float(loss.detach()):.5f}',flush=True)
            losses[split]=total/count
        improved=losses['val']<best;best=min(best,losses['val'])
        row=dict(epoch=epoch+1,train_loss=losses['train'],val_loss=losses['val'],learning_rate=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-before)
        history.append(row)
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,preprocessing=pre_cfg,
            provenance=provenance,epoch=epoch,best_loss=best,history=history,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        # Save best first: last never references an unsaved new best checkpoint.
        if improved:save_checkpoint(out/'best.pt',state)
        save_checkpoint(out/'last.pt',state)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        write_json(out/'training_report.json',dict(status='MEASURED',completed_epochs=epoch+1,best_validation_loss=best,history=history))
        print(row,flush=True)
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--images');p.add_argument('--out',required=True);p.add_argument('--config',default='configs/training_config.yaml');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();cfg=yaml.safe_load((root/a.config).read_text())
    train(root,a.images or root/'preprocessed/images',a.out,cfg,a.resume)
if __name__=='__main__':main()
