# -*-coding:UTF-8-*-
import sys,os
import traceback
import mysql.connector
import time

class MysqlCon:
    """
    Create a long mysql connection.
    """
    def __init__(self, host="",user="",passwd="",db="",port=3306,charset='utf8',retryNum=228800):
        """
        :param host:
        :param user:
        :param passwd:
        :param db:
        :param port:
        :param charset:
        """
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
        self.port = port
        self.charset = charset
        self.retryNum = retryNum
        self.sql_list = []
        self.conn = None
        self._conn()


    def _conn(self):
        try:
            print 'connect ', self.host 
            self.conn = mysql.connector.connect(host=self.host,user=self.user,
                                                passwd=self.passwd,db=self.db,port=self.port,charset=self.charset)
            return True
        except Exception as e:
            print e
            return False
        except:
            return False


    def _reConn(self,retryNum=28800,stime=3):#重试连接总次数为1天，假如服务器1天宕机都没发现，就...
        """
        :param retryNum:
        :param stime:
        :return:
        """
        _number = 0
        _status = True
        while _status and _number<=retryNum:
            try:
                self.conn.ping() #检查连接是否异常
                _status = False
            except:
                if self._conn()==True: #重新连接，成功退出
                    _status = False
                    break
                _number += 1
                time.sleep(stime) #连接不成功，休眠3分钟，继续循环，直到成功或重试次数结束


    def select(self,sql=''):
        """
        :param sql:
        :return:
        """
        try:
            self._reConn()
            self.cursor = self.conn.cursor()
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.cursor.close()
            return result
        except mysql.connector.Error as e:
            print(e)
            return False


    def insert(self,sql=''):
        """
        :param sql:
        :return:
        """
        self.sql_list.append(sql)
        #if len(self.sql_list)<10:
        if len(self.sql_list) < 10:
            return True
        self._reConn()
        self.cursor = self.conn.cursor()
        #for sql in self.sql_list:
        index = -1
        while(index<len(self.sql_list)-1):
            index += 1
            sql = self.sql_list[index]
            try:
                out = self.cursor.execute(sql)
            except mysql.connector.Error as err:
                print(err.msg)
                continue
                #return False
            del self.sql_list[index]
            index -= 1
        self.conn.commit()
        self.cursor.close()
        self.sql_list = []
        return True




    def select_limit(self,sql='',offset=0,length=20):
        """
        :param sql:
        :return:
        """
        sql = '%s limit %d , %d;' %(sql,offset,length)
        return self.select(sql)


    def query(self,sql=''):
        """
        :param sql:
        :return:
        """
        try:
            self._reConn()
            self.cursor = self.conn.cursor()
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.conn.commit()
            self.cursor.close()
            return (True,result)
        except mysql.connector.Error as err:
            print(err.msg)
            return False


    def close(self):
        self.conn.close()
