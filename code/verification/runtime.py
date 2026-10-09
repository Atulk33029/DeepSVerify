import hashlib,json,os,platform,random
from pathlib import Path
import numpy as np
import torch,torchvision


def seed_all(seed=42):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def save_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu=platform.processor() or platform.machine(),os=platform.platform())
