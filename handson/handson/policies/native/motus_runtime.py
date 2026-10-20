def install_validation_vlm_reuse(adapter):
    """Reuse deterministic condition features within ONE decoded action window.

    No training cache, no disk cache, no reuse across examples or weight updates.
    Must compare the entire decoded result against the original before accepting.
    """
    from .wam_models import MotusAdapter
    assert isinstance(adapter,MotusAdapter)
    original_decode=adapter.decode_action
    def decode(condition,steps,seed):
        assert not adapter.training,'Condition reuse is evaluation-only'
        module=adapter.policy.und_module
        original_extract=module.extract_und_features
        cached={}
        def extract(vlm_inputs):
            if not cached:
                cached['inputs']=vlm_inputs
                cached['value']=original_extract(vlm_inputs)
            else:
                assert vlm_inputs is cached['inputs'],'Changed VLM inputs within decode'
            # The joint denoiser can mutate its working understanding tokens.
            # Always give it an independent tensor, including on the first call.
            return cached['value'].clone()
        module.extract_und_features=extract
        try:return original_decode(condition,steps,seed)
        finally:module.extract_und_features=original_extract
    adapter.decode_action=decode
