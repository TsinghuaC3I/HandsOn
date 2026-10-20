"""Selected FastWAM checkpoint loader, copied from the paper deployment."""
import hashlib,json
from ..common import asset,path_option,vendor

class FastWAMEvalPolicy:
 def __init__(self,options):
  device=0
  vendor("fastwam")
  import torch,numpy as np,yaml
  from safetensors import safe_open
  from fastwam.models.wan22.fastwam import FastWAM
  from fastwam.models.wan22.wan_video_dit import WanVideoDiT
  from fastwam.models.wan22.action_dit import ActionDiT
  from fastwam.models.wan22.mot import MoT
  from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
  self.torch=torch;self.np=np;torch.cuda.set_device(device)
  c=yaml.safe_load(path_option(options, 'model_config', asset('fastwam', 'model.yaml')).read_text());self.config=c
  # Full module archive contains every model/VAE parameter. Construct exact native
  # classes without loading unrelated pretrained initialization, then strict-load.
  video=WanVideoDiT(**c['video_dit_config']).to(dtype=torch.bfloat16)
  action=ActionDiT(**c['action_dit_config']).to(dtype=torch.bfloat16)
  vae=WanVideoVAE38().to(dtype=torch.bfloat16)
  model=FastWAM(video,action,MoT({'video':video,'action':action}),vae,
   text_dim=4096,proprio_dim=52,device='cpu',torch_dtype=torch.bfloat16,
   video_train_shift=c['video_scheduler']['train_shift'],video_infer_shift=c['video_scheduler']['infer_shift'],
   video_num_train_timesteps=c['video_scheduler']['num_train_timesteps'],
   action_train_shift=c['action_scheduler']['train_shift'],action_infer_shift=c['action_scheduler']['infer_shift'],
   action_num_train_timesteps=c['action_scheduler']['num_train_timesteps'])
  p=options.checkpoint
  ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
  assert ck['optimizer_step']==ck['global_steps']==100000 and ck['prompt_patch']=='03'
  assert ck['data_fingerprint']=='17692771a13cd04a94568df00d2e97edc2bc1a90a2d54c6f1569e2147c914589'
  assert ck['excluded_tasks']==['h01_chinese_chess']
  loaded=model.load_state_dict(ck['module'],strict=True)
  assert not loaded.missing_keys and not loaded.unexpected_keys
  self.loaded_keys=len(ck['module']);del ck
  model.device=torch.device('cuda',device);model.to(device=model.device).eval().requires_grad_(False)
  self.model=model
  self.norm=json.loads(path_option(options, 'normalization', asset('fastwam', 'norm_stats.json')).read_text())['norm_stats']
  self.text=safe_open(str(path_option(options, 'text_cache')),framework='pt',device='cpu')
  self.text_keys=set(self.text.keys())
 def image(self,images):
  torch=self.torch
  from torchvision.transforms.functional import resize
  from torchvision.transforms import InterpolationMode
  cams=[]
  for cam,size in [('front',[256,320]),('wrist_left',[128,160]),('wrist_right',[128,160])]:
   im=self.np.asarray(images[cam]);assert im.dtype==self.np.uint8 and im.ndim==3 and im.shape[-1]==3
   x=torch.from_numpy(im.copy()).permute(2,0,1).float()/255.
   cams.append(resize(x,size,InterpolationMode.BILINEAR,antialias=True))
  return torch.cat([cams[0],torch.cat(cams[1:],dim=-1)],dim=-2)*2-1
 def state(self,x):
  np=self.np;x=np.asarray(x,dtype=np.float32)
  assert x.shape==(52,) and np.isfinite(x).all()
  n=self.norm['state']
  return self.torch.from_numpy(((x-np.array(n['mean'],np.float32))/np.maximum(np.array(n['std'],np.float32),1e-6)).astype(np.float32))
 def context(self,prompt):
  assert isinstance(prompt,str) and prompt.strip() and len(prompt)<4000
  k=hashlib.sha256(prompt.encode()).hexdigest()
  if k not in self.text_keys:raise ValueError('Prompt missing from native text cache; prepare it with handson.policies.encode_text')
  emb=self.text.get_tensor(k);context=self.torch.zeros(128,4096,dtype=self.torch.bfloat16);context[:len(emb)]=emb
  return context
 def predict(self,images,state,prompt,seed):
  image=self.image(images);proprio=self.state(state);context=self.context(prompt)
  with self.torch.inference_mode():
   prediction=self.model.infer_action(prompt=None,input_image=image,action_horizon=32,proprio=proprio,
    context=context,context_mask=self.torch.ones(128,dtype=self.torch.bool),num_inference_steps=10,
    seed=int(seed),rand_device='cpu')['action'].numpy()
  n=self.norm['actions'];actions=prediction*self.np.maximum(self.np.array(n['std']),1e-6)+self.np.array(n['mean'])
  assert actions.shape==(32,52) and self.np.isfinite(actions).all()
  return actions

class Policy(FastWAMEvalPolicy):
 def predict(self,state,images,prompt,seed):
  return super().predict(images,state,prompt,seed)
