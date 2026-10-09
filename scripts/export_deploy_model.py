"""Export trusted, locally trained checkpoint without optimizer or RNG state."""
import argparse,hashlib,json
from pathlib import Path
import torch

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export(source,destination,project):
    source,destination,project=Path(source),Path(destination),Path(project)
    record=json.loads((source/'threshold.json').read_text())
    original_hash=digest(source/'best.pt')
    if record['selected_on']!='val' or record['checkpoint_sha256']!=original_hash:raise ValueError('Source threshold/checkpoint mismatch')
    checkpoint=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
    for relative in ['code/preprocessing/transforms.py','code/preprocessing/eda.py']:
        if digest(project/relative)!=checkpoint['provenance'][relative]:raise ValueError('Preprocessing source differs: '+relative)
    if (destination/'best.pt').exists():raise FileExistsError('Export already exists; preserve it or choose another directory')
    destination.mkdir(parents=True,exist_ok=True)
    slim={k:checkpoint[k] for k in ['model','config','preprocessing','provenance']}
    torch.save(slim,destination/'best.pt')
    reloaded=torch.load(destination/'best.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(t,reloaded['model'][k]) for k,t in checkpoint['model'].items())
    record['source_checkpoint_sha256']=original_hash
    record['checkpoint_sha256']=digest(destination/'best.pt')
    (destination/'threshold.json').write_text(json.dumps(record,indent=2))
    (destination/'export.json').write_text(json.dumps({'weights_bit_identical':True,'source_checkpoint_sha256':original_hash,'checkpoint_sha256':record['checkpoint_sha256'],'bytes':(destination/'best.pt').stat().st_size},indent=2))
    print('Export complete: identical trained weights; threshold unchanged. Size MiB:',(destination/'best.pt').stat().st_size/1024**2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',default='models');a=p.parse_args()
    export(a.source,a.destination,Path(__file__).resolve().parents[1])
