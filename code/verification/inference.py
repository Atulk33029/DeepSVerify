import argparse,json
from pathlib import Path
import torch
from preprocessing.transforms import preprocess_image
from .evaluate import load_model
from .runtime import sha256

class Verifier:
    def __init__(self,run):
        run=Path(run);self.device=torch.device('cpu')
        record=json.loads((run/'threshold.json').read_text())
        if record['selected_on']!='val' or record['checkpoint_sha256']!=sha256(run/'best.pt'):
            raise ValueError('Frozen threshold does not match checkpoint')
        self.model,self.checkpoint=load_model(run/'best.pt',self.device)
        self.threshold=float(record['threshold']);self.cfg=self.checkpoint['preprocessing']
        root=Path(__file__).resolve().parents[2]
        for relative in ['code/preprocessing/transforms.py','code/preprocessing/eda.py']:
            if sha256(root/relative)!=self.checkpoint['provenance'][relative]:
                raise ValueError('Preprocessing code differs from training')

    def preprocess(self,path):
        locked=self.cfg['locked'];canvas,record=preprocess_image(path,self.cfg,locked['margin_fraction'],locked['intensity_variant'])
        if canvas is None or canvas.min()==canvas.max():raise ValueError('Upload a readable, nonblank signature image')
        if 'polarity_suspect' in record['flags']:raise ValueError('Use dark ink on a light background')
        return torch.from_numpy(canvas.astype('float32')/255).unsqueeze(0).unsqueeze(0)

    def verify(self,reference,query):
        a,b=self.preprocess(reference),self.preprocess(query)
        with torch.inference_mode():distance=float(self.model(a,b)[0])
        return dict(prediction='Genuine' if distance<=self.threshold else 'Forged',distance=distance,threshold=self.threshold)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--query',required=True);a=p.parse_args();print(json.dumps(Verifier(a.run).verify(a.reference,a.query),indent=2))
