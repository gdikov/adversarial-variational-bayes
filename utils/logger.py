import logging
import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# fh = logging.FileHandler('/Users/admin/Documents/University/TUM/Master_Info/SS17/DLRW/'
#                          'AdversarialVariationalBayes/logs/AVB_{}.log'.format(datetime.datetime.now().isoformat()))
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
