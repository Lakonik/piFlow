import warnings

# suppress warnings from MMCV about optional dependencies
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=r'^Fail to import ``MultiScaleDeformableAttention`` from ``mmcv\.ops\.multi_scale_deform_attn``.*',
    module=r'^mmcv\.cnn\.bricks\.transformer$',
)

# import all modules for registration
from .apis import *
from .datasets import *
from .models import *
from .ops import *
from .runner import *
from .evaluation import *
from .utils import *

from .version import __version__
