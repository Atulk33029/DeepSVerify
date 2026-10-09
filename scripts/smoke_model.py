"""Real training-data smoke test. Does not modify pairs or produce a usable model."""
import sys,json,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import pandas as pd
import torch,yaml
from verification.model import SiameseResNet18,contrastive_loss
from verification.data import load_canonical
from verification.runtime import seed_all,save_checkpoint,write_json,environment

root=Path(__file__).resolve().parents[1];torch.set_num_threads(2);seed_all()
images=root/'preprocessed/images';mapping=pd.read_csv(root/'reports/phase3a_image_split_map.csv')
# Prefer original train pairs when present. Otherwise form a clearly identified
# smoke-only pair from real train images; never save it into the locked pairs.
pair_source='original train_pairs.csv'
pairs_path=root/'pairs/train_pairs.csv'
if pairs_path.exists() and pairs_path.stat().st_size>100:
    pairs=pd.read_csv(pairs_path)
    rows=[]
    for label in [1,0]:
        eligible=pairs[pairs.pair_label==label]
        for _,row in eligible.iterrows():
            if all((images/f'{row[k]}.png').exists() for k in ['image_1_id','image_2_id']):rows.append((row.image_1_id,row.image_2_id,label));break
else:
    pair_source='smoke-only pair from verified training image map; not a replacement for locked pairs'
    available=mapping[(mapping.split=='train') & mapping.image_id.map(lambda x:(images/f'{x}.png').exists())]
    rows=[]
    for writer,g in available.groupby('global_writer_id'):
        genuine=g[g.signature_type=='genuine'].image_id.tolist();forged=g[g.signature_type=='forgery'].image_id.tolist()
        if len(genuine)>=2 and forged:
            rows=[(genuine[0],genuine[1],1),(genuine[0],forged[0],0)];break
if len(rows)!=2:raise RuntimeError('Need real training images for one genuine and one forgery pair')
a=torch.stack([load_canonical(images/f'{x}.png') for x,_,_ in rows]);b=torch.stack([load_canonical(images/f'{y}.png') for _,y,_ in rows]);labels=torch.tensor([float(v) for _,_,v in rows])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a,b,labels=a.to(device),b.to(device),labels.to(device)
model=SiameseResNet18().to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
losses=[];start=time.perf_counter()
for _ in range(3):
    opt.zero_grad();loss=contrastive_loss(model(a,b),labels);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();losses.append(float(loss.detach()))
model.eval()
with tempfile.TemporaryDirectory() as d:
    checkpoint=Path(d)/'smoke.pt';save_checkpoint(checkpoint,{'model':model.state_dict()});other=SiameseResNet18().to(device);other.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model']);other.eval()
    with torch.no_grad():
        assert torch.equal(model(a,b),other(a,b));assert model.encode(a).shape==(2,512)
report=dict(status='MEASURED',purpose='pipeline smoke test; NOT baseline training or accuracy evaluation',source=pair_source,pairs=rows,steps=3,losses=losses,seconds=time.perf_counter()-start,parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_reload_identical=True,environment=environment())
write_json(root/'reports/takeover_20261008/phase4_model_report.json',report);print(json.dumps(report,indent=2))
