"""Selected Motus checkpoint loader, including the native SDPA boundary."""
import json
from ..common import asset,path_option,vendor

def install_sdpa():
 """Length-aware native attention boundary on RTX5090 (no FA2 wheel)."""
 import torch
 import wan.modules.attention as attention
 def sdpa(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,dtype=torch.bfloat16,version=None):
  assert window_size==(-1,-1) and not causal and dropout_p==0
  out_dtype=q.dtype
  if q_scale is not None:q=q*q_scale
  result=[]
  for i in range(q.shape[0]):
   nq=q.shape[1] if q_lens is None else int(q_lens[i]);nk=k.shape[1] if k_lens is None else int(k_lens[i])
   assert nq==q.shape[1] and 0<nk<=k.shape[1]
   qq=q[i:i+1,:nq].transpose(1,2).to(dtype);kk=k[i:i+1,:nk].transpose(1,2).to(dtype);vv=v[i:i+1,:nk].transpose(1,2).to(dtype)
   result.append(torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,scale=softmax_scale).transpose(1,2).to(out_dtype))
  return torch.cat(result,dim=0)
 attention.flash_attention=sdpa
 import wan.modules.model as wan_model
 wan_model.flash_attention=sdpa
 return sdpa

class MotusEvalPolicy:
 def __init__(self,options):
  device=0
  vendor("motus")
  import torch
  from torch import nn
  torch.cuda.set_device(device);torch.set_num_threads(2)
  install_sdpa()
  from ..native import wam_models as models
  from ..native.wam_interfaces import PolicyNormalizer
  from models.motus import Motus,MotusConfig
  from transformers import AutoProcessor
  from utils.vlm_utils import preprocess_vlm_messages
  self.torch=torch
  contract=json.loads(path_option(options, 'model_config', asset('motus', 'data_contract.json')).read_text())
  self.normalizer=PolicyNormalizer(contract['normalization'],'motus')
  self.text_cache=torch.load(path_option(options, 'text_cache'),map_location='cpu',weights_only=True,mmap=True)
  # Create the identical trained architecture; skip historical 14D pretrain migration.
  # The full exact 52D checkpoint is subsequently loaded strictly, including all frozen weights.
  adapter=models.MotusAdapter.__new__(models.MotusAdapter);nn.Module.__init__(adapter)
  adapter.preprocess_vlm=preprocess_vlm_messages
  modelroot=path_option(options, 'foundation')
  adapter.processor=AutoProcessor.from_pretrained(str(modelroot/'Qwen3-VL-2B-Instruct'),local_files_only=True)
  adapter.text_cache=self.text_cache
  config=MotusConfig(wan_checkpoint_path='',wan_config_path=str(modelroot/'Wan2.2-TI2V-5B'),vae_path=str(modelroot/'Wan2.2-TI2V-5B/Wan2.2_VAE.pth'),vlm_checkpoint_path=str(modelroot/'Qwen3-VL-2B-Instruct'),action_state_dim=52,action_dim=52,action_expert_dim=1024,action_expert_norm_eps=1e-5,und_expert_norm_eps=1e-5,global_downsample_rate=1,video_action_freq_ratio=6,num_video_frames=8,video_height=384,video_width=320,batch_size=1,training_mode='finetune',load_pretrained_backbones=False)
  from transformers.modeling_utils import no_init_weights
  with no_init_weights():
   adapter.policy=Motus(config)
  for owner in (adapter.policy.action_expert,adapter.policy.video_model.wan_model):
   for name in ('time_embedding','time_projection'):
    for module in getattr(owner,name).modules():
     if isinstance(module,nn.Linear):module.__class__=models.Float32Linear
  adapter.to(dtype=torch.bfloat16,device='cuda')
  ckpt=options.checkpoint
  state=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
  assert state['step']==state['global_steps']==100000
  assert state['data_fingerprint']==contract['fingerprint']
  loaded=adapter.load_state_dict(state['module'],strict=True)
  self.loaded_keys=len(state['module']);del state
  adapter.eval();self.adapter=adapter
  self.initial_decode=adapter.decode_action
  self.optimized=False
 def predict_mosaic(self,image,state,prompt,seed):
  import numpy as np
  assert image.size==(320,384)
  state=np.asarray(state,np.float32)
  assert state.shape==(52,) and np.isfinite(state).all()
  if prompt not in self.text_cache:raise ValueError('Prompt missing from native text cache; prepare it with handson.policies.encode_text')
  condition={'images':[image],'state':self.normalizer.normalize(state,'state'),'prompt':prompt,'model_mask':self.normalizer.valid_mask}
  with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
   out=self.adapter.decode_action(condition,10,int(seed))[0].float().cpu().numpy()
  assert out.shape==(48,52) and np.isfinite(out).all()
  return self.normalizer.denormalize(out)
 def predict(self,front,left,right,state,prompt,seed):
  from PIL import Image
  from ..native.wam_images import mosaic
  return self.predict_mosaic(mosaic(*[Image.fromarray(x).convert('RGB') for x in (front,left,right)]),state,prompt,seed)
 def enable_verified_vlm_reuse(self):
  from ..native.motus_runtime import install_validation_vlm_reuse
  install_validation_vlm_reuse(self.adapter);self.optimized=True

class Policy(MotusEvalPolicy):
 def predict(self,state,images,prompt,seed):
  return super().predict(images['front'],images['wrist_left'],images['wrist_right'],state,prompt,seed)
