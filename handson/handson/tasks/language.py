"""Paper language sampler: 20% canonical, 80% among 32 training paraphrases.

The independent draw is made once per episode after native goal readback.
Public task identifiers give these draws a new identity; historical model
specific goal distributions and frozen draws are not relabeled.
"""
import hashlib,json
from functools import lru_cache
from .registry import ASSETS
SAMPLING_REVISION='handson-language'
LANGUAGE_SEED=20260924
BANK_SHA256="5d6d52c55c037c7eaa396e5a490b5d65bf8cda778f6d801ad8b62c46b8b83a2d"
@lru_cache(maxsize=1)
def bank():
    data=(ASSETS / 'language_bank.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=BANK_SHA256:
        raise ValueError('Packaged language bank changed')
    return json.loads(data)

def choose(task,values,bank,seed,language_seed):
 row=bank['tasks'][task];matches=[g for g in row['goals'] if g['values']==values];assert len(matches)==1,('Unsupported native goal',task,values)
 goal=matches[0];episode_uid=f'handson:{task}:seed:{int(seed)}'
 key=json.dumps([SAMPLING_REVISION,int(language_seed),episode_uid,goal['goal_id'],'train',0],separators=(',',':')).encode();raw=hashlib.sha256(key).digest()
 canonical=int.from_bytes(raw[:8],'big')/2**64 < .2
 index=None if canonical else int.from_bytes(raw[8:16],'big')%32
 prompt=goal['canonical'] if canonical else goal['train'][index]
 assert prompt not in goal['heldout']
 return dict(prompt=prompt,canonical_prompt=goal['canonical'],goal_id=goal['goal_id'],prompt_variant='canonical' if canonical else f'train_{index:02d}',prompt_variant_index=index,prompt_split='train',language_seed=language_seed,episode_uid=episode_uid,draw_sha256=hashlib.sha256(key).hexdigest(),draw_index=0)

def sample(task, values, seed):
    return dict(choose(task,values,bank(),seed,LANGUAGE_SEED),
                language_bank_sha256=BANK_SHA256, constant_within_episode=True,
                heldout_in_production=False)


@lru_cache(maxsize=None)
def allowed_prompts(task):
    return frozenset(text for goal in bank()["tasks"][task]["goals"]
                     for text in [goal["canonical"], *goal["train"]])
