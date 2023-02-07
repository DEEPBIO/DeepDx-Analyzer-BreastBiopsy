import os
'''
Some values used in multiple files are defined here to avoid circular reference error
'''

PROGRESS_PRE_PROCESS = 0.15 # pre processing takes around 20% of analyze()
PROGRESS_INFER = 0.7 # inference takes around 70% of analyze()
MPP_SCALE_VAL = 8

BATCH_SIZE = os.environ.get("BREAST_ANALYZER_BATCH_SIZE", 8)
WORKER_NUM = os.environ.get("BREAST_ANALYZER_WORKER", 8)