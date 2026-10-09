"""Read-only verification of existing canonical images, metadata and pairs."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import argparse,hashlib
import pandas as pd,numpy as np,yaml
from PIL import Image
from verification.data import validate_pairs
from verification.runtime import write_json
p=argparse.ArgumentParser();p.add_argument('--images',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];images=Path(a.images)
m=pd.read_csv(root/'preprocessed/preprocessing_metadata.csv');mapping=pd.read_csv(root/'reports/phase3a_image_split_map.csv');cfg=yaml.safe_load((root/'configs/preprocessing_config.yaml').read_text())
assert not m.image_id.duplicated().any()
assert set(m.image_id)==set(mapping.image_id),'Image metadata/map mismatch'
assert m.preprocessing_status.isin(['ok','flagged']).all()
assert np.allclose(m.margin_fraction,cfg['locked']['margin_fraction'])
assert m.intensity_variant.eq(cfg['locked']['intensity_variant']).all()
assert set(p.stem for p in images.glob('*.png'))==set(m.image_id),'Image cache incomplete or contains extras'
for i,row in enumerate(m.itertuples(index=False)):
    with Image.open(images/f'{row.image_id}.png') as im:
        assert im.mode=='L' and im.size==(384,128),row.image_id
        pixels=np.asarray(im)
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==row.output_sha256,row.image_id
    if i%1000==0:print(f'Checked {i}/{len(m)} existing images',flush=True)
counts={}
for split in ['train','val','test']:
    frame=pd.read_csv(root/f'pairs/{split}_pairs.csv');validate_pairs(frame,mapping,split);counts[split]=len(frame)
write_json(root/'reports/takeover_20261008/preflight_report.json',dict(status='MEASURED',result='PASS',canonical_images=len(m),pair_counts=counts))
print('Preflight PASS: original pixels and pairs preserved')
