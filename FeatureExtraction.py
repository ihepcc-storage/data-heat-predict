

#-*-coding:utf-8-*-

import sys, os

import logging

import traceback

import logging.handlers

import threading

import config



import hashlib

import happybase





logger = logging.getLogger('CSVLog')

logger.setLevel(config.log_level)

fh = logging.handlers.RotatingFileHandler(config.csv_file,mode='a',maxBytes=1024*1024*10,backupCount=1024*10)

logger.addHandler(fh)



def FeatureExtraction(log_str='', LogTableBatch=None, IndexTableBatch=None, UidOtsPathTableBatch=None,

                      UidPathOtsTableBacth=None, PathOtsUidTableBatch=None, OtsUidPathTableBatch=None):

    #global MyClient

    try:

        feature_list = []

        logid= ""

        td = ""

        username,path,ruid = "","",""

        isuser,path_depth = 0,0

        AccessProgress = 0.0

        path_suffix = 'NaN'

        alpha,digit,alpha01,digit01,alpha02,digit02,isuser= 0,0,0,0,0,0,0

        ots,cts,sfwdb,sbwdb,sxlbwdb,sxlfwdb = 0.0,0.0,0.0,0.0,0.0,0.0

        rb,rb_min,rb_max,rb_sigma,wb,wb_min,wb_max,wb_sigma = 0,0,0,0,0,0,0,0

        nrc,nwc,nbwds,nfwds,nxlbwds,nxlfwds,osize,csize=0,0,0,0,0,0,0,0

        rt,wt=0.0,0.0

        log_list = log_str.split('&')

        logid = log_list[0]

        for val in log_list:

            if val.find('td=') != -1 and len(val.split('=')) > 1:

                td = val.split('=')[1]

            if val.find('path=')!=-1 and len(val.split('='))>1:

                path = val.split('=')[1]

                if path.find('/eos/')==-1:

                    return -1

                index = path.find('/#curl#/')

                if index != -1:

                    path = path[7:]

                feature_list.append(path)

                """

                    _t = path.split('/')

                    if len(_t)>4:

                        username = _t[4]

                

                if path.find('.')>0 and len(path.split('.'))==2:

                    if len(path.split('.')[1])<5:

                        path_suffix = path.split('.')[1]

                _path = path.split('/')

                

                if isuser:

                    for i in range(0,4):

                        try:

                            del _path[1]

                        except:

                            break

                else:

                    for i in range(0,2):

                        try:

                            del _path[1]

                        except:

                            break

                path_depth = len(_path)

                if path_depth>1:

                    for i in _path[-1].split('.')[0]:

                        if i.isalpha():

                            alpha +=1

                        if i.isdigit():

                            digit +=1

                if path_depth>2:

                    for i in _path[-2].split('.')[0]:

                        if i.isalpha():

                            alpha01 += 1

                        if i.isdigit():

                            digit01 += 1

                if path_depth>3:

                    for i in _path[-3].split('.')[0]:

                        if i.isalpha():

                            alpha02 += 1

                        if i.isdigit():

                            digit02 += 1

                """

                """

                feature_list.append(str(isuser))

                feature_list.append(str(alpha))

                feature_list.append(str(digit))

                feature_list.append(str(alpha01))

                feature_list.append(str(digit01))

                feature_list.append(str(alpha02))

                feature_list.append(str(digit02))

                feature_list.append(path_suffix)

                feature_list.append(str(path_depth-1))

                """

            elif val.find('ruid=')!=-1 and len(val.split('='))>1:

                ruid = str(val.split('=')[1])

                feature_list.append(ruid)

            elif val.find('ots=')!=-1 and len(val.split('='))>1:

                ots = float(val.split('=')[1])

            elif val.find('otms=')!=-1 and len(val.split('=')) > 1:

                ots += float(val.split('=')[1])/1000.0

                ots = 9999999999.99 - ots

                ots = '%.2f' % ots

            elif val.find('cts=')!=-1 and len(val.split('='))>1:

                cts = float(val.split('=')[1])

            elif val.find('ctms=')!=-1 and len(val.split('='))>1:

                cts += float(val.split('=')[1])/1000.0

                cts = '%.2f' % cts

                #feature_list.append(str(cts-ots))

            elif val.find('rb=')!=-1 and len(val.split('='))>1:

                rb = int(val.split('=')[1])

                feature_list.append(str(rb))

            elif val.find('rb_min=') != -1 and len(val.split('=')) > 1:

                rb_min = int(val.split('=')[1])

                feature_list.append(str(rb_min))

            elif val.find('rb_max=') != -1 and len(val.split('=')) > 1:

                rb_max = int(val.split('=')[1])

                feature_list.append(str(rb_max))

            elif val.find('rb_sigma=') != -1 and len(val.split('=')) > 1:

                rb_sigma = float(val.split('=')[1])

                feature_list.append(str(rb_sigma))

                '''

                if rb>0:

                    feature_list.append(str(1))

                else:

                    feature_list.append(str(0))

                '''

            elif val.find('wb=')!=-1 and len(val.split('='))>1:

                wb = int(val.split('=')[1])

                feature_list.append(str(wb))

            elif val.find('wb_min=') != -1 and len(val.split('=')) > 1:

                wb_min = int(val.split('=')[1])

                feature_list.append(str(wb_min))

            elif val.find('wb_max=') != -1 and len(val.split('=')) > 1:

                wb_max = int(val.split('=')[1])

                feature_list.append(str(wb_max))

            elif val.find('wb_sigma=') != -1 and len(val.split('=')) > 1:

                wb_sigma = float(val.split('=')[1])

                feature_list.append(str(wb_sigma))

                '''

                if wb>0:

                    feature_list.append(str(1))

                else:

                    feature_list.append(str(0))

                '''

            elif val.find('sfwdb=')!=-1 and len(val.split('='))>1:

                sfwdb = float(val.split('=')[1])

                feature_list.append(str(sfwdb))

            elif val.find('sbwdb=')!=-1 and len(val.split('='))>1:

                sbwdb = float(val.split('=')[1])

                feature_list.append(str(sbwdb))

            elif val.find('sxlfwdb=')!=-1 and len(val.split('='))>1:

                sxlfwdb = float(val.split('=')[1])

                feature_list.append(str(sxlfwdb))

            elif val.find('sxlbwdb=')!=-1 and len(val.split('='))>1:

                sxlbwdb = float(val.split('=')[1])

                feature_list.append(str(sxlbwdb))

            elif val.find('nrc=')!=-1 and len(val.split('='))>1:

                nrc = int(val.split('=')[1])

                feature_list.append(str(nrc))

            elif val.find('nwc=')!=-1 and len(val.split('='))>1:

                nwc = int(val.split('=')[1])

                feature_list.append(str(nwc))

            elif val.find('nfwds=')!=-1 and len(val.split('='))>1:

                nfwds = int(val.split('=')[1])

                feature_list.append(str(nfwds))

            elif val.find('nbwds=')!=-1 and len(val.split('='))>1:

                nbwds = int(val.split('=')[1])

                feature_list.append(str(nbwds))

            elif val.find('nxlfwds=')!=-1 and len(val.split('='))>1:

                nxlfwds = int(val.split('=')[1])

                feature_list.append(str(nxlfwds))

            elif val.find('nxlbwds=')!=-1 and len(val.split('='))>1:

                nxlbwds = int(val.split('=')[1])

                feature_list.append(str(nxlbwds))

            elif val.find('rt=')!=-1 and len(val.split('='))>1:

                rt = float(val.split('=')[1])

                feature_list.append(str(rt))

            elif val.find('wt=')!=-1 and len(val.split('='))>1:

                wt = float(val.split('=')[1])

                feature_list.append(str(wt))

            elif val.find('osize=')!=-1 and len(val.split('='))>1:

                osize = int(val.split('=')[1])

                feature_list.append(str(osize))

            elif val.find('csize=')!=-1 and len(val.split('='))>1:

                csize = int(val.split('=')[1])

                feature_list.append(str(csize))


        if (rb+wb==0):
            return 0

        size = max(osize, csize)

        if size!=0:

            if rb>0:

                AccessProgress = round(-1*rb / size, 3)

            if wb>0:

                AccessProgress = round(wb / size, 3)

        """

        #Edited in 2017

        username_hash = hashlib.md5(username).hexdigest()

        path_hash = hashlib.md5(path).hexdigest()

        row_key = username_hash+path_hash+str(AccessProgress)+AccessTime

        print row_key

        """


        #Edited in 2018

        logid_hash = hashlib.md5(logid).hexdigest()

        path_hash = hashlib.md5(path).hexdigest()


        try:

            IndexTableBatch.put(path_hash,{'name:filename':path})


            UidOtsPathTableBatch.put(str(ruid+str(ots)+path_hash),{'cf:logid':logid_hash})

            UidPathOtsTableBacth.put(ruid+path_hash+str(ots),{'cf:logid':logid_hash})

            PathOtsUidTableBatch.put(path_hash+str(ots)+ruid,{'cf:logid':logid_hash})

            OtsUidPathTableBatch.put(str(ots)+ruid+path_hash,{'cf:logid':logid_hash})


            LogTableBatch.put(logid_hash,{'Id:uid':ruid})

            LogTableBatch.put(logid_hash,{'size:osize':str(osize),'size:csize':str(csize),'size:ots':str(ots),'size:cts':str(cts)})

            LogTableBatch.put(logid_hash,{'cf_r:rb':str(rb),'cf_r:rb_min':str(rb_min),'cf_r:rb_max':str(rb_max),'cf_r:rb_sigma':str(rb_sigma),

                                    'cf_r:nrc':str(nrc),'cf_r:rt':str(rt)})

            LogTableBatch.put(logid_hash,{'cf_w:wb':str(wb),'cf_w:wb_min':str(wb_min),'cf_w:wb_max':str(wb_max),'cf_w:wb_sigma':str(wb_sigma),

                                    'cf_w:nwc':str(nwc),'cf_w:wt':str(wt)})

            LogTableBatch.put(logid_hash,{'seek:sfwdb':str(sfwdb),'seek:sbwdb':str(sbwdb),'seek:sxlfwdb':str(sxlfwdb),'seek:sxlbwdb':str(sxlbwdb),'seek:nfwds':str(nfwds),'seek:nbwds':str(nbwds),'seek:nxlfwds':str(nxlfwds),'seek:nxlbwds':str(nxlbwds)})

        except Exception as e:

            print "hbase insert failed"

            print e

            traceback.print_exc()

            return -2



        """

        sql = "insert into FileFeature values('%s','%s','%s',%d,%d,%d,%d,%d,%d,%d,'%s',%d,'%s',%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f," \

              "%f,%f,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,FROM_UNIXTIME(%f))" % \

              (logid, td, path, isuser, alpha, digit, alpha01, digit01, alpha02, digit02, path_suffix, path_depth, ruid,

               cts - ots, rb, rb_min

               , rb_max, rb_sigma, wb, wb_min, wb_max, wb_sigma, sfwdb, sbwdb, sxlfwdb, sxlbwdb, nrc, nwc, nfwds,

               nbwds, nxlfwds, nxlbwds, rt, wt, osize, csize,ots)

        print sql

        print 'insert begin'

        while True:

            try:

                MyClient.insert(sql)

                break

            except Exception as e:

                print "Error occured when trying to insert a row into mysql"

                print e

        """

    except Exception as e:

        print e

        traceback.print_exc()

        return -2

    return 0

    pass


def FeatureExtractionStr():

    pass



def FeatureExtractionSimple(log_str='', LogTableBatch=None, IndexTableBatch=None, UidOtsPathTableBatch=None,

                      UidPathOtsTableBacth=None, PathOtsUidTableBatch=None, OtsUidPathTableBatch=None):

    ots = 0.0

    ruid = "00000"

    path = ""


    log_list = log_str.split('&')

    for val in log_list:

        if val.find('ruid=') != -1 and len(val.split('=')) > 1:

            ruid = str(val.split('=')[1])

            if len(ruid)<5:

                ruid += ('0'*(5-len(ruid)))

        if val.find('ots=') != -1 and len(val.split('=')) > 1:

            ots = float(val.split('=')[1])

        if val.find('otms=') != -1 and len(val.split('=')) > 1:

            ots += float(val.split('=')[1]) / 1000.0

            ots = 9999999999.99 - ots

            ots = '%.2f' % ots

        if val.find('path=') != -1 and len(val.split('=')) > 1:

            path = val.split('=')[1]

            if path.find('/eos/') == -1:

                return -1

            index = path.find('/#curl#/')

            if index != -1:

                path = path[7:]

    try:

        path_hash = hashlib.md5(path).hexdigest()

    except Exception as e:

        print path

        print e

        return 0


    try:

        OtsUidPathTableBatch.put(str(ots) + ruid + path_hash, {'cf:log': log_str})

        PathOtsUidTableBatch.put(path_hash + str(ots) + ruid, {'cf:log': log_str})

        UidOtsPathTableBatch.put(ruid + str(ots) + path_hash, {'cf:log': log_str})

        UidPathOtsTableBacth.put(ruid + path_hash + str(ots), {'cf:log': log_str})

    except Exception as e:

        print e

        traceback.print_exc()

        return -2


    return 0


    pass




def main():

    #MyClient = MysqlCon.MysqlCon(**MysqlConfig)

    #sql = 'delete from FileFeature;'

    #MyClient.query(sql)

    #MyClient.close()

    connection = happybase.Connection('localhost', autoconnect=True)

    table = connection.table('eos_log:eos_log')


    pass



if __name__ == '__main__':

    main()


