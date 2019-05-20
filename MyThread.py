#!/bin/env python
#-*-coding:utf-8-*-

import threading

class MyThread(object):
    def __init__(self, func_list=None):
    #�����̺߳����ķ���ֵ���ܣ�������Ϊ0��˵��ȫ���ɹ�
        self.ret_flag = []
        self.func_list = func_list
        self.threads = []


    def set_thread_func_list(self, func_list):
        """
        :param func_list: a list. Each member is a dict which contains two params: func and args
        :return:
        """
        self.func_list = func_list


    def trace_func(self, func, *args, **kwargs):
        #���profile_func���µĸ����̷߳���ֵ�ĺ�����������ִ�е��̺߳����������һ�㺯�����Ի�ȡ����ֵ
        """
        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        ret = func(*args, **kwargs)
        self.ret_flag.append(ret)


    def start(self):
        """
        :start multiple threads and wait
        :return:
        """
        self.threads = []
        self.ret_flag = []
        for func_dict in self.func_list:
            if func_dict["args"]:
                new_arg_list = []
                new_arg_list.append(func_dict["func"])
                for arg in func_dict["args"]:
                    new_arg_list.append(arg)
                new_arg_tuple = tuple(new_arg_list)
                t = threading.Thread(target=self.trace_func, args=new_arg_tuple)
            else:
                t = threading.Thread(target=self.trace_func, args=(func_dict["func"],))
            self.threads.append(t)

        for thread_obj in self.threads:
            thread_obj.start()

        for thread_obj in self.threads:
            thread_obj.join()


    def ret_value(self):
        """
        :return: self.ret_flag
        """
        return self.ret_flag

    def clear_ret_flag(self):
        """
        clear ret flag
        :return:
        """
        self.ret_flag = []
