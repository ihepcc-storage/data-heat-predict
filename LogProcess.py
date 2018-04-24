#-*-coding:utf-8-*-
#!/bin/env python

import os, sys
import time
import threading
import subprocess
import logging
import logging.handlers
import traceback
import config as config
import MyThread
import ESReceive


TimeBegin = 1513612800*1000 #在这里修改日志采集起始时间
Interval_time = 1000 * 60
LogSize = 300
RA = ESReceive.ReceiveAPI(200,TimeBegin, 1) # 在这里修改日志采集起始时间
MAX_THREADNUM = 1



def ThreadsControl(logger=None):
    """Threads Control"""
    MaxThread = 100
    MinThread = 1
    mt = MyThread.MyThread()
    ThreadNum = 0
    g_func_list = []
    g_arg_list = []
    g_arg_list.append({"TimeBegin": RA.GetTimeBegin(), "Interval_time": RA.Interval_time, "LogSize": RA.GetLogSize()})
    g_func_list.append({"func": RA.run, "args": (RA.GetTimeBegin(), RA.Interval_time, RA.GetLogSize(),)})
    logger.info(g_func_list)
    while True:
        for num in range(1,ThreadNum+1):
            print "add a thread"
            RA.UpdateTimeBegin()
            g_arg_list.append({"TimeBegin":RA.GetTimeBegin(),"Interval_time":RA.Interval_time,"LogSize":RA.GetLogSize()})
            g_func_list.append({"func":RA.run,"args":(RA.GetTimeBegin(),RA.Interval_time,RA.GetLogSize(),)})
        logger.info(g_func_list)
        mt.set_thread_func_list(g_func_list)
        mt.start()
        g_func_list = []
        #g_arg_list = []
        index = -1
        for i in mt.ret_value():
            index += 1
            if i == 1:
                g_arg_list[index] = None
                if (ThreadNum+len(g_func_list))<MAX_THREADNUM:
                    ThreadNum += 1
            elif i == 0:
                g_arg_list[index] = None
                if ThreadNum>0:
                    ThreadNum -= 1
            else:
                try:
                    g_func_list.append({"func":RA.run,"args":(g_arg_list[index]["TimeBegin"],
                                                        g_arg_list[index]["Interval_time"],g_arg_list[index]["LogSize"])})
                except Exception as e:
                    print e
                    traceback.print_exc()
        index = -1
        for val in g_arg_list:
            index += 1
            if val==None:
                del g_arg_list[index]
        #break


def main():
    logger = logging.getLogger("Mainlog")
    logger.setLevel(config.log_level)
    fh = None
    try:
        fh = logging.handlers.WatchedFileHandler(config.log_file)
    except AttributeError:
        fh = logging.handlers.TimedRotatingFileHandler(config,'D',1,14)
        fh.suffix = "%Y%m%d.log"
    except:
        return
    fh.setFormatter(logging.Formatter(config.log_format))
    logger.addHandler(fh)


    ThreadsControl(logger)

    #mt = MyThread.MyThread()
    #g_func_list = []




if __name__ == '__main__':
    main()
