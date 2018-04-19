import sys
import importlib
from pathlib import Path

dir_path = Path(__file__).parent.absolute()
myself = sys.modules[__name__]
for module_iter in dir_path.glob('*.py'):
  module_name = module_iter.stem
  if module_name == '__init__':
    continue
  module = importlib.import_module(f'{__name__}.{module_name}')
  for mod in module.__dict__.keys():
    if not mod in ['__builtins__', '__doc__', '__file__', '__name__', '__package__']:
      myself.__dict__[mod] = module.__dict__[mod]
