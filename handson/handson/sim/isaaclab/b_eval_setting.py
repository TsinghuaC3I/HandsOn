"""Explicit task-scoped setting selected before scene construction."""
import json,os
from functools import lru_cache
@lru_cache(maxsize=1)
def selected():
    path=os.environ.get('HANDSON_B_EVAL_SELECTION')
    return json.load(open(path)) if path else None
def setting_for(task):
    value=selected()
    return value if value and value['task']==task else None
