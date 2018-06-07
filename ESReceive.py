#!/bin/env python

import os, sys
import time
import threading
import logging
import logging.handlers
import subprocess
import json
import config
import traceback
import happybase
import FeatureExtraction


MysqlConfig = {
    'user':'chengzj',
    'passwd':'chengzj@2016',
    'host':'localhost',
    'db':'PopularDB'
}

MyClient = None


class ReceiveAPI(object):
    '''*****************************************************'''
    def __init__(self, LogSize, TimeBegin, CacheHit=1):
        #threading.Thread.__init__(self)

        self.TimeBegin = TimeBegin
        self.Interval_time = config.Interval_time
        if LogSize>config.FstLogSize:
            self.LogSize = LogSize
        else:
            self.LogSize = config.FstLogSize
        self.flag = 0
        self.CacheHit = CacheHit
        self.ResultFile = ""
        self.logger = logging.getLogger("Mainlog")
        self.pool = happybase.ConnectionPool(size=3,host='192.168.60.64',autoconnect=True)
        self.connectHbase()


    def connectHbase(self):
        try:
            self.connection = happybase.Connection('192.168.60.64', autoconnect=True)
            #self.connection = self.pool.connection()
            self.table = self.connection.table("eos_log")
            self.IndexTable = self.connection.table("eos_log_index")
            self.UidOtsPathTable = self.connection.table("eoslog_UidOtsPathMap")
            self.UidPathOtsTable = self.connection.table("eoslog_UidPathOtsMap")
            self.PathOtsUidTable = self.connection.table("eoslog_PathOtsUidMap")
            self.OtsUidPathTable = self.connection.table("eoslog_OtsUidPathMap")

            self.bat = self.table.batch(batch_size=10000)
            self.IndexBat = self.IndexTable.batch(batch_size=2000)
            self.UidOtsPathBat = self.UidOtsPathTable.batch(batch_size=1000)
            self.UidPathOtsBat = self.UidPathOtsTable.batch(batch_size=1000)
            self.PathOtsUidBat = self.PathOtsUidTable.batch(batch_size=1000)
            self.OtsUidPathBat = self.OtsUidPathTable.batch(batch_size=1000)
        except Exception as e:
            print "Hbase connect error!"
            print e
            traceback.print_exc()

        #self.MyClient = MysqlCon.MysqlCon(**MysqlConfig)


    def __del__(self):
        self.MyClient.close()

    def GetLogSize(self):
        return self.LogSize


    def SetLogSize(self, LogSize):
        self.LogSize = LogSize


    def GetTimeBegin(self):
        return self.TimeBegin


    def UpdateTimeBegin(self):
        if (self.TimeBegin+self.Interval_time)/1000 < (time.time()-3600):
            self.TimeBegin = self.TimeBegin + self.Interval_time
        else:
            time.sleep(600)
            self.connection.close()
            self.connectHbase()


    def GetCacheHit(self):
        return self.CacheHit



    def GetCmd(self,TimeBegin=-1,Interval_time=-1,LogSize=-1):
        if TimeBegin==-1:
            TimeBegin = self.TimeBegin
        if Interval_time==-1:
            Interval_time = self.Interval_time

        if LogSize==-1:
            LogSize = self.LogSize

        self.cmd = 'curl -XGET \'http://elastic:mine09443@logger01.ihep.ac.cn:9200/eos_fst_log-*/eos_fst_log/_search \
                                   \' -H "Content-Type:application/json" -d \'{"query":{"bool":{"must":[{"term":{"func.keyword":"Report"}},{"range":{"@timestamp": \
                                   {"gt":"%d","lte":"%d"}}},{"query_string":{"default_field":"log","query":\
                                   "sec.app=fuse"}}],"must_not":[],"should":[]}},"from":0,"size":%d,"sort":[],"aggs":{}}\'' % \
                   (TimeBegin, TimeBegin + Interval_time, LogSize)

        #print "CMD:", self.cmd

        print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(TimeBegin/1000)),"   ",\
            time.strftime('%Y-%m-%d %H:%M:%S',time.localtime((TimeBegin+Interval_time)/1000))
        return self.cmd



    def JSONParse(self, InputStr=""):
        try:
            jsonObj = json.loads(InputStr)
            return jsonObj
        except:
            #self.logger.error(e)
            #print e
            traceback.print_exc()
            return -1
        pass



    def HTTP_GET(self,TimeBegin=-1,Interval_time=-1,LogSize=-1):
        """create a http get request."""
        try:
            cmd = self.GetCmd(TimeBegin,Interval_time,LogSize)
            #self.logger.info(cmd)
            t_begin = time.time()
            print 'begin time is', t_begin
            sp = subprocess.Popen(cmd, shell=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = sp.communicate(input=None)
            t_end = time.time()
            print 'end time is', t_end
            if sp.returncode != 0:
                self.logger.error("Error when fetching fst log from es.")
                self.logger.error(err)
                self.CacheHit = -1
                return
            else:
                if (t_end - t_begin) > config.CacheHitTime:
                    self.CacheHit = 0
                else:
                    self.CacheHit = 1
                return out
        except Exception as e:
            print e
            traceback.print_exc()
            return



    def judgedict(self,dict):
        if not dict.has_key('hits'):
            print 0
            return False
        if not dict['hits'].has_key('total'):
            print 1
            return False
        if not dict['hits'].has_key('hits'):
            print 2
            return False
        for i in dict['hits']['hits']:
            if not i.has_key('_index'):
                print 3
                return False
            if not i.has_key('_source'):
                print 4
                return False
            if not i['_source'].has_key('log'):
                print 5
                return False
        if len(dict['hits']['hits'])<1:
            print -1
            return True

        return True


    def run(self,TimeBegin=-1,Interval_time=-1,LogSize=-1,):
        if TimeBegin == -1:
            TimeBegin = self.TimeBegin
        response = self.HTTP_GET(TimeBegin,Interval_time,LogSize)
        responseDict = self.JSONParse(response)
        resultDict = []
        if responseDict == -1:
            self.logger.error("")
            return -1
        if not self.judgedict(responseDict):
            #print resultDict
            print 'responseDict Error!'
            print response
            '''
            if str(response).find('Result window is too large')>0:
                #config.Interval_time = config.Interval_time/2
                print 'Result window is too large'
                print self.GetCmd(TimeBegin, Interval_time/2, LogSize=10000)
                self.CacheHit = self.run(TimeBegin,Interval_time/2,LogSize=10000)
                print self.GetCmd(TimeBegin+Interval_time/2,Interval_time/2,LogSize=10000)
                self.CacheHit = self.run(TimeBegin+Interval_time/2,Interval_time/2,LogSize=10000)
            return self.CacheHit
            '''
            return -1

        try:
            validLogNum = int(responseDict['hits']['total'])
            print 'validLogNum ',validLogNum
            if validLogNum > LogSize:
                if validLogNum<=10000:
                    self.LogSize = validLogNum
                    self.CacheHit = self.run(TimeBegin, Interval_time, self.LogSize)
                elif validLogNum>10000:
                    print "Log size window is too large"
                    print TimeBegin, Interval_time / 2,TimeBegin + Interval_time / 2, Interval_time / 2
                    self.LogSize = 10000
                    print self.GetCmd(TimeBegin, Interval_time / 2, LogSize=10000)
                    self.CacheHit = self.run(TimeBegin, Interval_time / 2, LogSize=10000)
                    print self.GetCmd(TimeBegin + Interval_time / 2, Interval_time / 2, LogSize=10000)
                    self.CacheHit = self.run(TimeBegin + Interval_time / 2, Interval_time / 2, LogSize=10000)

            else:
                #self.logger.info("Received %d records!" % validLogNum)
                print "Received %d records!" % validLogNum
                if len(responseDict['hits']['hits'])>1:
                    self.ResultFile = responseDict['hits']['hits'][0]['_index']
                #print self.ResultFile

                i = 0
                """
                while(i<len(responseDict['hits']['hits'])):
                    if responseDict['hits']['hits'][i]['_source']['log'].find('path=/#curl#/') != -1:
                        del responseDict['hits']['hits'][i]
                        i -= 1
                    i += 1
                """

                i = -1
                log = ""
                print "Selected %d records!" % len(responseDict['hits']['hits'])
                while 1:
                    while(i<len(responseDict['hits']['hits'])-1):
                        i += 1
                        #print responseDict['hits']['hits'][i]['_source']['log']
                        #print "log is", log[:100]
                        if responseDict['hits']['hits'][i]['_source']['log']==log:
                            #print 'Same log'
                            del responseDict['hits']['hits'][i]
                            i -= 1
                            continue

                        #print 'Not Same log'
                        if(i<len(responseDict['hits']['hits'])):
                            log = responseDict['hits']['hits'][i]['_source']['log']
                            # little modification
                            # re = FeatureExtraction.FeatureExtraction(log,self.bat,self.IndexBat,self.UidOtsPathBat,self.UidPathOtsBat,
                               #                                        self.PathOtsUidBat,self.OtsUidPathBat)
                            re = FeatureExtraction.FeatureExtractionSimple(log,self.bat,self.IndexBat,self.UidOtsPathBat,self.UidPathOtsBat,
                                                                       self.PathOtsUidBat,self.OtsUidPathBat)
                            ErrorNum = 0
                            while(re == -2): # Hbase Connection failed. Try to reConnect with hbase.
                                ErrorNum += 1
                                print 'This is ', ErrorNum ,'error!'
                                #with self.pool.connection() as connection1:
                                connection1 = happybase.Connection('192.168.60.64', autoconnect=True)
                                table = connection1.table("eos_log")
                                IndexTable = connection1.table("eos_log")
                                UidOtsPathTable = connection1.table("eoslog_UidOtsPathMap")
                                UidPathOtsTable = connection1.table("eoslog_UidPathOtsMap")
                                PathOtsUidTable = connection1.table("eoslog_PathOtsUidMap")
                                OtsUidPathTable = connection1.table("eoslog_OtsUidPathMap")
                                bat = table.batch(batch_size=1)
                                IndexBat = IndexTable.batch(batch_size=1)
                                UidOtsPathBat = UidOtsPathTable.batch(batch_size=1)
                                UidPathOtsBat = UidPathOtsTable.batch(batch_size=1)
                                PathOtsUidBat = PathOtsUidTable.batch(batch_size=1)
                                OtsUidPathBat = OtsUidPathTable.batch(batch_size=1)
                                    # little modificaiton
                                    # re = FeatureExtraction.FeatureExtraction(log, bat, IndexBat, UidOtsPathBat,
                                                                          #   UidPathOtsBat, PathOtsUidBat,
                                                                          #  OtsUidPathBat)

                                re = FeatureExtraction.FeatureExtractionSimple(log, bat, IndexBat,
                                                                               UidOtsPathBat, UidPathOtsBat,
                                                                               PathOtsUidBat, OtsUidPathBat)

                                try:
                                    bat.send()
                                    IndexBat.send()
                                    UidOtsPathBat.send()
                                    UidPathOtsBat.send()
                                    PathOtsUidBat.send()
                                    OtsUidPathBat.send()
                                except:
                                    re = -2

                                if re == 0:
                                    self.bat = bat
                                    self.IndexBat = IndexBat
                                    self.UidOtsPathBat = UidOtsPathBat
                                    self.UidPathOtsBat = UidPathOtsBat
                                    self.PathOtsUidBat = PathOtsUidBat
                                    self.OtsUidPathBat = OtsUidPathBat


                            if ErrorNum > 0:
                                try:
                                    print 'Create new self.connection'
                                    self.connection.close()
                                    self.connectHbase()
                                except Exception as e:
                                    print e
                                    traceback.print_exc()
                    #bat.send()
                    #IndexBat.send()
                            #resultDict.append(responseDict['hits']['hits'][i]['_source']['message'])

                #config.ResultFileLock.acquire()
                #with open("./fst_log/"+self.ResultFile, 'a') as f:
                    #for l in resultDict:
                        #f.write(l)
                        #f.write("\n")
                #config.ResultFileLock.release()
                    break
        except Exception as e:
            print e
            traceback.print_exc()
            return -1

        print "exit run"
        return self.CacheHit



RA = ReceiveAPI(200,1527782400000, 1)


def main():
    while True:
        CacheHit = RA.run(LogSize=config.FstLogSize,Interval_time=config.Interval_time)

        print "CacheHit", CacheHit
        if CacheHit==-1:
            continue
        elif CacheHit==0:
            RA.UpdateTimeBegin()
        elif CacheHit==1:
            print 'UpdateTimeBegin',RA.TimeBegin,RA.Interval_time
            RA.UpdateTimeBegin()
            print 'UpdateTimeBegin', RA.TimeBegin, RA.Interval_time
        else:
            break

if __name__ == '__main__':
    main()


