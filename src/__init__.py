'''
Example module for template project.
Pylint will check code in the src directory only!
'''

import logging
import os
from datetime import datetime

__all__ = [
]

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
if not os.path.exists('log'):
    os.makedirs('log')
current_datetime = datetime.now()

fh = logging.FileHandler("log/{}_{}_{}_{}.log".format(current_datetime.year,
                                                      current_datetime.month,
                                                      current_datetime.day,
                                                      current_datetime.hour))
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
