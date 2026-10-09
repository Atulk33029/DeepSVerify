from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_canonical(path):
    with Image.open(path) as im:
        if im.mode != 'L' or im.size != (384,128):
            raise ValueError(f'Noncanonical image: {path}: {im.mode}, {im.size}')
        return torch.from_numpy(np.array(im, dtype=np.float32)/255.0).unsqueeze(0)


def validate_pairs(frame, image_map, split):
    required = {'pair_id','image_1_id','image_2_id','global_writer_id','dataset','split','pair_label','pair_type'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Missing pair columns or empty pairs')
    if frame.pair_id.duplicated().any() or not frame.split.eq(split).all():
        raise ValueError('Duplicate pair IDs or incorrect split')
    if not frame.pair_label.isin([0,1]).all():
        raise ValueError('Pair labels must be 0/1')
    canonical = frame.apply(lambda r: tuple(sorted((r.image_1_id,r.image_2_id))),axis=1)
    if canonical.duplicated().any():
        raise ValueError('Duplicate or reversed pair')
    lookup = image_map.set_index('image_id',verify_integrity=True)
    for side in [1,2]:
        ids=frame[f'image_{side}_id']
        if not ids.isin(lookup.index).all():
            raise ValueError('Unknown image ID')
        rows=lookup.loc[ids].reset_index(drop=True)
        for column in ['global_writer_id','dataset','split']:
            if not rows[column].eq(frame[column].reset_index(drop=True)).all():
                raise ValueError(f'Pair {column} mismatch')
        expected = pd.Series('genuine',index=range(len(frame))) if side==1 else frame.pair_label.reset_index(drop=True).map({1:'genuine',0:'forgery'})
        if not rows.signature_type.eq(expected).all():
            raise ValueError('Pair signature labels mismatch')
    if not frame.pair_type.eq(frame.pair_label.map({1:'genuine_genuine',0:'genuine_forgery'})).all():
        raise ValueError('Pair type mismatch')

class PairDataset(Dataset):
    def __init__(self, frame, images_dir):
        self.frame=frame.reset_index(drop=True)
        self.images_dir=Path(images_dir)
        for image_id in set(frame.image_1_id)|set(frame.image_2_id):
            if Path(image_id).name!=image_id or not (self.images_dir/f'{image_id}.png').is_file():
                raise FileNotFoundError(f'Missing canonical image: {image_id}')
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        return (load_canonical(self.images_dir/f'{row.image_1_id}.png'),
                load_canonical(self.images_dir/f'{row.image_2_id}.png'),
                torch.tensor(float(row.pair_label)))
