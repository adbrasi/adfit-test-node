from .nodes.batchcrop_nodes import ADBatchCropFromMaskAdvanced, ADBatchUncropAdvanced

NODE_CLASS_MAPPINGS = {
    "ADBatchCropFromMaskAdvanced": ADBatchCropFromMaskAdvanced,
    "ADBatchUncropAdvanced": ADBatchUncropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ADBatchCropFromMaskAdvanced": "AD CropFit Advanced",
    "ADBatchUncropAdvanced": "AD UncropFit Advanced",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
