import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import streamlit as st
from PIL import Image
st.set_page_config(page_title='DeepSVerify · Signature Intelligence',page_icon='✍️',layout='wide')
st.markdown('''<style>
.block-container{max-width:1120px;padding-top:2.3rem}.hero{padding:32px;border:1px solid #253951;border-radius:20px;background:linear-gradient(120deg,#142b3e,#0b182b);margin-bottom:26px}.eyebrow{color:#35dac0;letter-spacing:3px;font-size:12px;font-weight:700}.hero h1{font-size:46px;letter-spacing:-2px;margin:8px 0}.hero p{color:#b6c5d8;max-width:690px;line-height:1.7}.stButton button{border-radius:10px}div[data-testid="stMetric"]{background:#142136;padding:16px;border-radius:12px}footer{visibility:hidden}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">COMPUTER VISION · RESEARCH DEMO</div><h1>DeepSVerify<span style="color:#35dac0">.</span></h1><p>Two signatures. One measured comparison.<br>Explore offline signature verification with a shared Siamese neural network.</p></div>',unsafe_allow_html=True)
verify,results,method=st.tabs(['Verify signatures','Measured results','How it works'])
run=ROOT/'models'
@st.cache_resource
def verifier(checkpoint_mtime,threshold_mtime):
    import torch
    torch.set_num_threads(2)
    from verification.inference import Verifier
    return Verifier(run)
with verify:
    st.caption('Upload a genuine reference and a query from the same claimed writer. Use clear, similarly captured images with dark ink on light paper.')
    left,right=st.columns(2)
    with left:reference=st.file_uploader('01 / Genuine reference',type=['png','jpg','jpeg','tif','tiff','bmp'],key='reference')
    with right:query=st.file_uploader('02 / Query signature',type=['png','jpg','jpeg','tif','tiff','bmp'],key='query')
    ready=(run/'best.pt').is_file() and (run/'threshold.json').is_file()
    if not ready:st.info('Model setup is incomplete. The trained deployment checkpoint must be added before live verification is available.')
    if st.button('Compare signatures →',type='primary',disabled=not(reference and query and ready),width='stretch'):
        try:
            with tempfile.TemporaryDirectory() as folder:
                paths=[]
                for name,upload,column in [('reference',reference,left),('query',query,right)]:
                    if upload.size>10*1024*1024:raise ValueError('Use an image smaller than 10 MB.')
                    path=Path(folder)/name;path.write_bytes(upload.getvalue())
                    with Image.open(path) as im:
                        if im.width*im.height>20_000_000:raise ValueError('Use an image below 20 megapixels.')
                        im.load();column.image(im.copy(),width='stretch')
                    paths.append(path)
                with st.spinner('Comparing signature embeddings…'):
                    engine=verifier((run/'best.pt').stat().st_mtime_ns,(run/'threshold.json').stat().st_mtime_ns)
                    result=engine.verify(*paths)
            if result['prediction']=='Genuine':st.success('MODEL PREDICTION: GENUINE')
            else:st.warning('MODEL PREDICTION: FORGED')
            a,b=st.columns(2);a.metric('Euclidean distance',f"{result['distance']:.5f}");b.metric('Validation threshold',f"{result['threshold']:.5f}")
            st.caption('A distance at or below the frozen threshold is classified as genuine. Distance is not a probability or confidence percentage.')
        except Exception as exc:st.error(f'Unable to compare these images: {exc}')
    st.caption('Student research prototype. Genuine signatures can be rejected and forgeries can be accepted. Uploads are processed in temporary files that this app deletes after comparison; it does not add them to the dataset.')
with results:
    report=json.loads((ROOT/'results/test_metrics.json').read_text());m=report['overall']
    a,b,c,d=st.columns(4);a.metric('Test accuracy',f"{m['accuracy']:.2%}");b.metric('ROC-AUC',f"{m['ROC_AUC']:.4f}");c.metric('False acceptance',f"{m['FAR']:.2%}");d.metric('False rejection',f"{m['FRR']:.2%}")
    st.caption('Measured on 25,768 writer-disjoint test pairs. Pair counts are not counts of independent people. Positive class: genuine.')
    st.dataframe([{'Dataset':name,'Pairs':v['n'],'Accuracy':f"{v['accuracy']:.2%}",'False acceptance':f"{v['FAR']:.2%}",'False rejection':f"{v['FRR']:.2%}"} for name,v in report['datasets'].items()],hide_index=True,width='stretch')
    st.image(str(ROOT/'assets/training.png'),caption='Recorded training and validation contrastive loss',width='stretch')
with method:
    st.subheader('One shared encoder. A reproducible decision.')
    st.markdown('1. Composite transparency onto white and convert to grayscale.\n2. Detect foreground, crop with a 10% margin, flatten the background, and resize with padding to 384 × 128.\n3. Apply the same ResNet-18 to both images to produce 512-dimensional embeddings.\n4. Compare Euclidean distance with the threshold selected on validation data only.')
    st.info('The baseline uses contrastive loss, seed 42, and no augmentation. The best checkpoint is selected by validation loss. The published test threshold remains unchanged.')
    st.markdown('**Known limitations:** sensitivity to different capture conditions, a single reference image, and uneven error rates across datasets. Identical-image acceptance is expected and is not evidence of handwriting generalisation.')
st.divider();st.caption('DeepSVerify · Offline Signature Forgery Detection & Verification')
