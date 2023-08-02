from ..hook_module.register_hook import initialize_hook
from .dump import pretest_hook
from .info_dump import initialize_output_json
from .utils import set_dump_switch


initialize_hook(pretest_hook)
initialize_output_json()

__all__ = ['set_dump_switch']