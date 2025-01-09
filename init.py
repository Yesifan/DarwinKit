from darkit.core.lib.options import save_models_metadata

try:
    from darkit.lm.models import Metadata as lm_metadata

    save_models_metadata("lm", lm_metadata)
    print("Updated models options for lm.")
except Exception as e:
    print(f"Error loading lm.models: {e}")
