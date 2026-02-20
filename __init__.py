from .nodes.batchcrop_nodes import BatchCropFromMaskAdvanced, BatchUncropAdvanced

NODE_CLASS_MAPPINGS = {
    "BatchCropFromMaskAdvanced": BatchCropFromMaskAdvanced,
    "BatchUncropAdvanced": BatchUncropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchCropFromMaskAdvanced": "Batch Crop From Mask Advanced",
    "BatchUncropAdvanced": "Batch Uncrop Advanced",
}

WEB_DIRECTORY = "./web"

from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):
    try:
        PromptServer.instance.app.add_routes(
            [
                web.static(
                    "/kjweb_async",
                    (Path(__file__).parent.absolute() / "kjweb_async").as_posix(),
                )
            ]
        )
    except Exception:
        pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
