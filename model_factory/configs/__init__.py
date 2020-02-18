import os
import yaml

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = 'config.yml'
CONFIG_PATH = os.path.join(CURR_DIR, CONFIG_FILE)

CONFIG_ENV = os.getenv('CONFIG_ENV', 'datadumpling')

__baseconfig__ = yaml.safe_load(open(CONFIG_PATH))
__config__ = __baseconfig__[CONFIG_ENV]
