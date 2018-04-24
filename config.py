import os, sys
import logging
import threading
import time

ResultFileLock = threading.RLock()

log_file = "./FstLogProcess.log"
log_level = logging.INFO
log_format = "%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"
TimeBegin = 1504195200000
Interval_time = 1000 * 60
CacheHitTime = 5
FstLogSize = 2000
csv_file = "./csv/EOS.csv"