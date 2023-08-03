from api_accuracy_checker.hook_module.register_hook import initialize_hook
from api_accuracy_checker.dump.dump import pretest_hook
from api_accuracy_checker.dump.info_dump import initialize_output_json
from api_accuracy_checker.dump.utils import set_dump_switch


initialize_hook(pretest_hook)
initialize_output_json()

__all__ = ['set_dump_switch', 'msCheckerConfig']