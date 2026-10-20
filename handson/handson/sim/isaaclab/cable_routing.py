"""Preserve the measured cable anchors and length across wrist poses."""
import json
import os
from pathlib import Path
import numpy as np


def make_solver(original, adjustments, diagnostic_path=None):
    numerical_cases=[]

    def length(points):
        return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())

    def solve(profile,palm,forearm,*,solve_length=True):
        target=float(profile['length_m'])
        reason='overlong_original_bridge'
        overrun=None
        try:
            result=original(profile,palm,forearm,solve_length=solve_length)
            if not solve_length:return result
            overrun=length(result[0])-target
            if abs(overrun)<=5e-7:return result
            # The original source permits a base curve up to 20 microns
            # longer than its target. Its nonnegative slack search cannot
            # shorten that curve. Preserve the downstream 2-micron check.
            assert 0<overrun<=2.01e-5,('Unexpected native cable solver error',overrun)
            reason='native_positive_length_deadband'
        except ValueError as error:
            if str(error)!='Requested cable length cannot span the current anchors':raise
        base,_=original(profile,palm,forearm,solve_length=False)
        for prefix_end in (24,12,1):
            suffix=base[prefix_end:]
            u=np.linspace(0,1,len(suffix))[:,None]
            chord=(1-u)*suffix[0]+u*suffix[-1]
            def curve(scale):return np.vstack([base[:prefix_end],chord+scale*(suffix-chord)])
            minimum=length(curve(0))
            if minimum<=target:break
        assert minimum<=target,('Frozen cable physically cannot span anchors',profile['palm_link'],minimum,target)
        low,high=0.,1.
        for _ in range(35):
            mid=(low+high)/2
            if length(curve(mid))<target:low=mid
            else:high=mid
        scale=(low+high)/2;points=curve(scale)
        assert abs(length(points)-target)<5e-8
        assert np.array_equal(points[:prefix_end],base[:prefix_end])
        assert np.array_equal(points[-1],base[-1])
        item={'side':profile['palm_link'],'reason':reason,'original_bridge_length_m':length(base),
              'target_length_m':target,'bridge_contraction':scale,'fixed_prefix_end_index':prefix_end,
              'corrected_length_m':length(points),'native_length_overrun_m':overrun}
        adjustments.append(item)
        if reason=='native_positive_length_deadband' and diagnostic_path is not None and len(numerical_cases)<8:
            numerical_cases.append({**item,'profile':profile,'palm':np.asarray(palm).tolist(),
                                    'forearm':np.asarray(forearm).tolist()})
            path=Path(diagnostic_path);temporary=path.with_suffix('.next')
            temporary.write_text(json.dumps({'version':1,'cases':numerical_cases},indent=2)+'\n')
            os.replace(temporary,path)
        return points,{'length_m':length(points),'slack_amplitude_m':0.,'bridge_contraction':scale}

    return solve
