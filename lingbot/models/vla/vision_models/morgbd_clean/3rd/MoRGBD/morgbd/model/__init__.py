import importlib
from typing import *

if TYPE_CHECKING:
    from .v1 import MoRGBDModel as MoRGBDModelV1
    from .v2 import MoRGBDModel as MoRGBDModelV2


def import_model_class_by_version(version: str) -> Type[Union['MoRGBDModelV1', 'MoRGBDModelV2']]:
    assert version in ['v1', 'v2'], f'Unsupported model version: {version}'
    
    try:
        module = importlib.import_module(f'.{version}', __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')

    cls = getattr(module, 'MoRGBDModel')
    return cls
