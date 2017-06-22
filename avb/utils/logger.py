import logging
import datetime
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# log_path = os.path.join('output', 'log')
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# filename = os.path.join(log_path, 'AVB_{}.log'.format(datetime.datetime.now().isoformat()))
# fh = logging.FileHandler(filename)
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
