import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch,yaml
from PIL import Image
from verification.model import SiameseResNet18,contrastive_loss
from verification.runtime import seed_all,save_checkpoint,sha256
from verification.metrics import metrics,select_threshold
from verification.data import load_canonical,validate_pairs
from verification.inference import Verifier
from preprocessing.transforms import preprocess_image

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_shapes_shared_weights_and_symmetric_distance():
    seed_all();m=SiameseResNet18();m.eval();x=torch.rand(2,1,128,384);y=torch.rand_like(x)
    with torch.no_grad():
        assert m.encode(x).shape==(2,512)
        assert torch.equal(m(x,x),torch.zeros(2))
        assert torch.allclose(m(x,y),m(y,x))
    assert len([n for n,_ in m.named_modules() if n=='backbone'])==1
    with pytest.raises(ValueError):m.encode(torch.rand(2,3,128,384))


def test_contrastive_label_semantics():
    assert contrastive_loss(torch.tensor([0.,1.]),torch.tensor([1.,0.])).item()==0
    assert contrastive_loss(torch.tensor([1.,0.]),torch.tensor([1.,0.])).item()==1


def test_backward_optimizer_checkpoint_and_seed(tmp_path):
    seed_all();m=SiameseResNet18();initial=m.backbone.conv1.weight.detach().clone()
    x=torch.rand(2,1,128,384);y=torch.rand_like(x);optimizer=torch.optim.Adam(m.parameters(),lr=1e-4)
    loss=contrastive_loss(m(x,y),torch.tensor([1.,0.]));loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    optimizer.step();assert not torch.equal(initial,m.backbone.conv1.weight)
    p=tmp_path/'checkpoint.pt';save_checkpoint(p,dict(model=m.state_dict()));n=SiameseResNet18();n.load_state_dict(torch.load(p,weights_only=True)['model']);m.eval();n.eval()
    with torch.no_grad():assert torch.equal(m(x,y),n(x,y))
    seed_all();r=SiameseResNet18();assert torch.equal(initial,r.backbone.conv1.weight)


def test_metrics_and_threshold_ties():
    y=np.array([1,1,0,0]);d=np.array([.1,.2,.8,.9]);t=select_threshold(y,d);v=metrics(y,d,t)
    assert v['accuracy']==1 and v['FAR']==0 and v['FRR']==0 and v['ROC_AUC']==1
    assert metrics(y,np.ones(4),1)['FAR']==1
    with pytest.raises(ValueError):metrics([1,1],[.1,.2],.2)


def test_inference_reuses_training_preprocessing(tmp_path):
    run=tmp_path/'run';run.mkdir();cfg=yaml.safe_load((ROOT/'configs/preprocessing_config.yaml').read_text())
    m=SiameseResNet18();checkpoint=dict(model=m.state_dict(),config={'normalize_embeddings':False},preprocessing=cfg,
        provenance={s:sha256(ROOT/s) for s in ['code/preprocessing/transforms.py','code/preprocessing/eda.py']})
    save_checkpoint(run/'best.pt',checkpoint)
    (run/'threshold.json').write_text(json.dumps(dict(threshold=.5,selected_on='val',checkpoint_sha256=sha256(run/'best.pt'))))
    v=Verifier(run);path=tmp_path/'raw.png';arr=np.full((70,150),235,np.uint8);arr[25:30,20:125]=20;Image.fromarray(arr).save(path)
    canvas,_=preprocess_image(path,cfg,.1,'flatten_bg');assert np.array_equal(v.preprocess(path).numpy()[0,0],canvas.astype('float32')/255)
    result=v.verify(path,path);assert result==dict(prediction='Genuine',distance=0.,threshold=.5)
    Image.new('L',(100,100),255).save(path)
    with pytest.raises(ValueError):v.verify(path,path)
    (run/'threshold.json').write_text(json.dumps(dict(threshold=.5,selected_on='test',checkpoint_sha256=sha256(run/'best.pt'))))
    with pytest.raises(ValueError):Verifier(run)


def test_canonical_loading(tmp_path):
    p=tmp_path/'image.png';Image.new('L',(384,128),255).save(p);x=load_canonical(p)
    assert x.shape==(1,128,384) and x.min()==1 and x.max()==1
    Image.new('RGB',(384,128)).save(p)
    with pytest.raises(ValueError):load_canonical(p)


def test_pair_validation_rejects_bad_labels_and_cross_writer():
    images=pd.DataFrame([['a','w','CEDAR','train','genuine'],['b','w','CEDAR','train','forgery']],columns=['image_id','global_writer_id','dataset','split','signature_type'])
    pairs=pd.DataFrame([['ab','a','b','w','CEDAR','train',0,'genuine_forgery']],columns=['pair_id','image_1_id','image_2_id','global_writer_id','dataset','split','pair_label','pair_type'])
    validate_pairs(pairs,images,'train')
    broken=pairs.copy();broken.loc[0,'pair_label']=1
    with pytest.raises(ValueError):validate_pairs(broken,images,'train')
    images.loc[1,'global_writer_id']='other'
    with pytest.raises(ValueError):validate_pairs(pairs,images,'train')
