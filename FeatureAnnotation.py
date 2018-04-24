import sys,os
import threading
import math
from datetime import datetime
import MysqlCon
import config
import utilities

popularPath = []
nreadList = []
rbList = []
rankList = []
MysqlConfig = {
    'user':'chengzj',
    'passwd':'chengzj@2016',
    'host':'localhost',
    'db':'PopularDB'
}
MyClient = None


def createSQL(path="",nread=0,rb=0.0,rank=-1,):
    """
    :param path:
    :param nread:
    :param rb:
    :param rank:
    :return:
    """
    SQL = "insert into PopuarDir (PopuarDirPath,nread,nb,rank,date) values('%s',%d,%f,%d,'%s')" \
          % (path,nread,rb,rank,datetime.now())
    return SQL
    pass


def FindPopularFile(path="",nread=0,rb=0,rank=-1,):
    """
    :param path:
    :return:
    """
    global nreadList
    global rbList
    global rankList
    global popularPath

    for i in range(0,len(popularPath)):
        if path.startswith(popularPath[i][0]) and (len(path.split('/'))-len(popularPath[i][0].split('/')))==1:
            flag = 1
            if path not in popularPath[i]:
                popularPath[i].append(path)
                nreadList[i] -= nread
                rbList[i] -= rb

    tmplist = []
    tmplist.append(path)
    popularPath.append(tmplist)
    nreadList.append(nread)
    rbList.append(rb)
    rankList.append(rank)

    _max = len(popularPath)
    t = 0
    while t<_max:
        if nreadList[t]==0:
            del popularPath[t]
            del nreadList[t]
            del rbList[t]
            del rankList[t]
            t -= 1
            _max -= 1
        t += 1


def FeatureAnnotation(filename=''):
    global MyClient
    line = None
    with open(name=filename, mode='r') as file:
        for line in file:
            print line
            if line.find('nread=')!=-1:
                nread_begin = line.find('nread=')
                nread_end = line.find('rb=')
                nb_end = line.find('/')
                nread = int(line[nread_begin+6:nread_end].strip())
                rb_str = line[nread_end+3:nb_end].strip()
                rb = 0
                if rb_str.split(' ')[1]=='TB':
                    rb = float(rb_str.split(' ')[0])*math.pow(10,12)
                if rb_str.split(' ')[1]=='GB':
                    rb = float(rb_str.split(' ')[0])*math.pow(10,9)
                if rb_str.split(' ')[1]=='MB':
                    rb = float(rb_str.split(' ')[0])*math.pow(10,6)
                if rb_str.split(' ')[1]=='kB':
                    rb = float(rb_str.split(' ')[0])*math.pow(10,3)
                rank = int(line[:6])
                t = line.find('/#curl#/')
                if t!=-1:
                    path = line[t:].strip()
                    FindPopularFile(path, nread, rb, rank)
                else:
                    t = line.find('/eos')
                    if t!=-1:
                        path = line[t:].strip()
                        FindPopularFile(path,nread,rb,rank)
        FindPopularFile("")

    for i in range(0,len(popularPath)):
        SQL = createSQL(popularPath[i][0],nreadList[i],rbList[i],rankList[i])
        MyClient.insert(SQL)
    #for i in range(0, len(popularPath)):
        #print popularPath[i][0],nreadList[i],rbList[i],rankList[i]
    pass


def main():
    global MyClient
    MyClient = MysqlCon.MysqlCon(**MysqlConfig)
    FeatureAnnotation("./eos_io_ns/out")
    MyClient.close()
    pass


if __name__ == '__main__':
    main()