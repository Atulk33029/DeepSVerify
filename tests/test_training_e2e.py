"""Synthetic integration test. Outputs must never be reported as dataset results."""
import json,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from PIL import Image
from verification.train import train
from verification.evaluate import evaluate


def test_train_resume_freeze_test_pipeline(tmp_path):
    torch.set_num_threads(2)
    source=Path(__file__).resolve().parents[1];root=tmp_path/'project'
    for relative in ['configs/preprocessing_config.yaml','code/preprocessing/transforms.py','code/preprocessing/eda.py']:
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
    (root/'pairs').mkdir();(root/'reports').mkdir();images=root/'preprocessed/images';images.mkdir(parents=True)
    rows=[];manifest=[];rng=np.random.default_rng(42)
    for s,w in [('train','w1'),('val','w2'),('test','w3')]:
        ids=[w+'a',w+'b',w+'c']
        for i,identity in enumerate(ids):
            Image.fromarray(rng.integers(0,256,(128,384),dtype=np.uint8)).save(images/f'{identity}.png')
            rows.append(dict(image_id=identity,global_writer_id=w,dataset='CEDAR',split=s,signature_type='genuine' if i<2 else 'forgery'))
        manifest.append(dict(global_writer_id=w,split=s))
        pd.DataFrame([dict(pair_id=w+'pos',image_1_id=ids[0],image_2_id=ids[1],global_writer_id=w,dataset='CEDAR',split=s,pair_label=1,pair_type='genuine_genuine'),dict(pair_id=w+'neg',image_1_id=ids[0],image_2_id=ids[2],global_writer_id=w,dataset='CEDAR',split=s,pair_label=0,pair_type='genuine_forgery')]).to_csv(root/f'pairs/{s}_pairs.csv',index=False)
    pd.DataFrame(rows).to_csv(root/'reports/phase3a_image_split_map.csv',index=False)
    pd.DataFrame(manifest).to_csv(root/'reports/phase3a_split_manifest.csv',index=False)
    cfg=yaml.safe_load((source/'configs/training_config.yaml').read_text());cfg.update(epochs=1,batch_size=2)
    run=root/'checkpoints/integration'
    train(root,images,run,cfg);before=(run/'best.pt').read_bytes()
    train(root,images,run,cfg,resume=True);assert (run/'best.pt').read_bytes()==before
    evaluate(root,images,run,'val');frozen=(run/'threshold.json').read_bytes();evaluate(root,images,run,'test')
    assert (run/'threshold.json').read_bytes()==frozen
    report=json.loads((run/'test/metrics.json').read_text());assert report['overall']['n']==2
    assert (run/'test/roc.png').exists() and (run/'test/errors.csv').exists()
