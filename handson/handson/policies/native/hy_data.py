import json
import numpy as np
NORM_FILE = None

class NumericSpace:
    def __init__(self):
        self.stats = json.loads(NORM_FILE.read_text())["norm_stats"]

    def normalize(self, values, kind="actions"):
        s = self.stats[kind]
        return ((values - np.asarray(s["mean"], dtype=np.float32)) /
                np.maximum(np.asarray(s["std"], dtype=np.float32), 1e-6)).astype(np.float32)

    def denormalize(self, values):
        s = self.stats["actions"]
        return (values * np.maximum(np.asarray(s["std"], dtype=np.float32), 1e-6)
                + np.asarray(s["mean"], dtype=np.float32))

def model_inputs(batch, device, inference=False):
    result = {}
    for k, v in batch.items():
        if k.startswith("observation.images."):
            result[k] = v.to(device, non_blocking=True).float().div_(255)
        elif k == "observation.state" or (not inference and k in ("action", "action_is_pad")):
            result[k] = v.to(device, non_blocking=True)
        elif k == "task":
            result[k] = v
    if inference:
        assert not any(k in result for k in ("action", "action_is_pad", "target_physical"))
    return result
