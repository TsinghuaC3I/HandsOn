import json
from pathlib import Path

def require(value, message):
    if not value: raise ValueError(message)

class Preprocessor:
    """Literal training-reader math, including un-clipped state bins and padding."""
    cameras={'front':'base_0_rgb','wrist_left':'left_wrist_0_rgb','wrist_right':'right_wrist_0_rgb'}
    def __init__(self,norm_file,tokenizer_file):
        import numpy as np
        import sentencepiece as spm
        self.norm=json.loads(Path(norm_file).read_text())['norm_stats']
        self.tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_file))
        for name in ('state','actions'):
            require(np.asarray(self.norm[name]['q01']).shape==(52,),'Invalid normalization dimension')
    def observation(self,state,prompt,images):
        import numpy as np
        from PIL import Image
        state=np.asarray(state,np.float32)
        require(state.shape==(52,) and np.isfinite(state).all(),'Invalid measured EEF52 state')
        require(isinstance(prompt,str) and 0<len(prompt)<4000,'Invalid task prompt')
        n=self.norm['state'];lo=np.asarray(n['q01'],np.float32);hi=np.asarray(n['q99'],np.float32)
        normalized=(2*(state-lo)/(hi-lo+1e-6)-1).astype(np.float32)
        discrete=np.digitize(normalized,bins=np.linspace(-1,1,257)[:-1])-1
        cleaned=prompt.strip().replace('_',' ').replace('\n',' ')
        text=f"Task: {cleaned}, State: {' '.join(map(str,discrete))};\nAction: "
        tokens=self.tokenizer.encode(text,add_bos=True)
        require(len(tokens)<=512,f'Prompt/state truncation forbidden: {len(tokens)}')
        ids=np.zeros(512,np.int32);ids[:len(tokens)]=tokens
        result={};padding={}
        require(set(images)==set(self.cameras),'Three camera images required')
        for camera,key in self.cameras.items():
            im=np.asarray(images[camera]);require(im.dtype==np.uint8 and im.ndim==3 and im.shape[-1]==3,'Invalid RGB image')
            h,w=im.shape[:2];require(h>0 and w>0,'Empty image')
            ratio=max(h/224,w/224);rh=int(h/ratio);rw=int(w/ratio);top=(224-rh)//2;left=(224-rw)//2
            resized=np.asarray(Image.fromarray(im).resize((rw,rh),Image.Resampling.BILINEAR))
            dst=np.zeros((224,224,3),np.uint8);dst[top:top+rh,left:left+rw]=resized
            mask=np.zeros((224,224),bool);mask[top:top+rh,left:left+rw]=True
            result[key]=dst.astype(np.float32)/127.5-1;padding[key]=mask
        return dict(image=result,image_mask={k:np.bool_(True) for k in result},image_padding_mask=padding,
            state=normalized,tokenized_prompt=ids,tokenized_prompt_mask=np.arange(512)<len(tokens))
    def denormalize(self,actions):
        import numpy as np
        n=self.norm['actions'];lo=np.array(n['q01']);hi=np.array(n['q99'])
        return ((actions+1)*.5*(hi-lo+1e-6)+lo).astype(np.float32)
