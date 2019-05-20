#!/bin/env python

import happybase
import csv

connection = happybase.Connection('192.168.60.64', autoconnect=True)
table = connection.table('eos_log')
tableIndex = connection.table('eos_log_index')
UidOtsPathTable = connection.table("eoslog_UidOtsPathMap")
UidPathOtsTable = connection.table("eoslog_UidPathOtsMap")
PathOtsUidTable = connection.table("eoslog_PathOtsUidMap")
OtsUidPathTable = connection.table("eoslog_OtsUidPathMap")

def statistics(TimeBegin=0.0, TimeEnd=0.0, csv_Batchwriter=None, csv_Userwriter=None, limit=None):

    rowStart = str(TimeBegin)+'00000'+'0'*32
    rowEnd = str(TimeEnd)+'99999'+'f'*32
    LoginNodeFile = 0
    ComputeNodeFile = 0
    DuplicateFeature = 0
    feature_set = set()
    for key, data in OtsUidPathTable.scan(row_start=rowStart, row_stop=rowEnd, reverse=True):
        if limit!= None:
            if len(feature_set)>limit:
                break

        try:
            log = data['cf:log']
        except:
            #print 'No log, continue'
            continue

        path_hash = key[-32:]
        log_list = log.split('&')
        rb = float(log_list[13].split('=')[1])
        wb = float(log_list[17].split('=')[1])

        osize = float(log_list[34].split('=')[1])
        host = log_list[-6].split('=')[1]
        if rb == 0.0 and wb > 0.0 and osize == 0.0:
            if host.startswith('vm'):
                ComputeNodeFile += 1
            else:
                LoginNodeFile += 1
            print 'LoginNodeFile:', LoginNodeFile, 'ComputeNodeFile', ComputeNodeFile

            filename_suffix = 'none'
            filename_part2 = 'none'
            filename_part3 = 'none'
            filename_part4 = 'none'
            filename_part5 = 'none'
            filename_part6 = 'none'
            filename_part7 = 'none'
            filename_part8 = 'none'
            filename_part9 = 'none'
            r_list = []
            BatchFileNum = 0
            UserFileNum = 0
            ruid = log_list[2].split('=')[1]
            rgid = log_list[3].split('=')[1]
            filename = log_list[1].split('=')[1]
            if filename.find('/eos/') == -1:
                continue
            index = filename.find('/#curl#/')
            if index != -1:
                filename = filename[7:]

            _filename = filename.split('/')
            if _filename[-1].find('.') > 0:
                filename_suffix = _filename[-1].split('.', 1)[-1]
                _suffix_list = filename_suffix.split('.')
                if len(_suffix_list)>1:
                    _suffixlen_list = []
                    for i in range(0,len(_suffix_list)):
                        _suffixlen_list.append(str(len(_suffix_list[i])))
                    filename_suffix = '-'.join(_suffixlen_list)

            filename_depth = len(_filename)
            if filename_depth > 2:
                filename_part2 = _filename[-2]
            if filename_depth > 3:
                filename_part3 = _filename[-3]
            if filename_depth > 4:
                filename_part4 = _filename[-4]
            if filename_depth > 5:
                filename_part5 = _filename[-5]
            if filename_depth > 6:
                filename_part6 = _filename[-6]
            if filename_depth > 7:
                filename_part7 = _filename[-7]
            if filename_depth > 8:
                filename_part8 = _filename[-8]
            if filename_depth > 9:
                filename_part9 = _filename[-9]
            for k, d in PathOtsUidTable.scan(row_prefix=path_hash):
                try:
                    _log = d['cf:log']
                except:
                    continue
                _log_list = _log.split('&')
                _rb = float(_log_list[13].split('=')[1])
                _wb = float(_log_list[17].split('=')[1])
                _host = _log_list[-6].split('=')[1]
                if _rb > 0.0 and _wb == 0.0:
                    if _host.startswith('vm'):
                        BatchFileNum += 1
                    else:
                        UserFileNum += 1
            r_list.extend([ruid, rgid, filename_suffix, str(filename_depth), host, filename_part2, filename_part3, filename_part4,
                           filename_part5, filename_part6, filename_part7, filename_part8, filename_part9,
                           BatchFileNum > UserFileNum])
            feature_str = ','.join(r_list[:-1])
            old_len = len(feature_set)
            feature_set.add(feature_str)
            new_len = len(feature_set)
            if new_len>old_len:
                if BatchFileNum>UserFileNum:
                    csv_Batchwriter.writerow(r_list)
                else:
                    csv_Userwriter.writerow(r_list)
            else:
                DuplicateFeature += 1
                print 'DuplicateFeature:', DuplicateFeature
    return key

    pass



def ScanHbase(rowStart=None, csv_Batchwriter=None, csv_Userwriter=None, limit=100, reverse=False, ForceBalance=True):

    #rowStart = TimeBegin+'00000'+'0'*32
    #rowEnd = TimeEnd+'99999'+'f'*32

    ct = 0
    old_path_hash = ''
    BatchFile = 0
    UserFile = 0
    NewFileFlag = 0
    ruid = 'none'
    rgid = 'none'
    filename_suffix = 'none'
    filename_depth = 'none'
    filename_part2 = 'none'
    filename_part3 = 'none'
    filename_part4 = 'none'
    filename_part5 = 'none'
    filename_part6 = 'none'
    filename_part7 = 'none'
    filename_part8 = 'none'
    filename_part9 = 'none'
    create_host = 'none'
    last_create_host = 'none'

    BatchFileNum = 0
    UserFileNum = 0
    looptoken = 0
    last_log = ""
    log = ""
    r_dict = {}

    for key, data in PathOtsUidTable.scan(row_start=rowStart, reverse=reverse):
        print BatchFileNum, UserFileNum


        if BatchFileNum+UserFileNum>limit-1:
            print BatchFileNum, UserFileNum
            return key

        r_list = []
        try:
            last_log = log
            log = data['cf:log']
        except:
            log = ""
            print 'No log, continue'
            continue

        log_list = log.split('&')
        path_hash = key[:32]
        if path_hash not in r_dict.keys():
            r_dict[path_hash] = {}
            r_dict[path_hash]['BatchFile'] = 0
            r_dict[path_hash]['UserFile'] = 0
            r_dict[path_hash]['create_host'] = 'none'
            r_dict[path_hash]['ruid'] = 'none'
            r_dict[path_hash]['rgid'] = 'none'
            r_dict[path_hash]['filename_suffix'] = 'none'
            r_dict[path_hash]['filename_depth'] = 0
            r_dict[path_hash]['filename_part2'] = 'none'
            r_dict[path_hash]['filename_part3'] = 'none'
            r_dict[path_hash]['filename_part4'] = 'none'
            r_dict[path_hash]['filename_part5'] = 'none'
            r_dict[path_hash]['filename_part6'] = 'none'
            r_dict[path_hash]['filename_part7'] = 'none'
            r_dict[path_hash]['filename_part8'] = 'none'
            r_dict[path_hash]['filename_part9'] = 'none'


        rb = float(log_list[13].split('=')[1])
        wb = float(log_list[17].split('=')[1])
        osize = float(log_list[34].split('=')[1])
        if rb == 0.0 and wb>0.0 and osize>0.0 :
            continue

        host = log_list[-6].split('=')[1]
        if host.startswith('vm'):
            #BatchFile += 1
            if rb>0.0:
                r_dict[path_hash]['BatchFile'] += 1
        else:
            #if wb>0.0:
                #BatchFile += 1
            #UserFile += 1
            if rb>0.0:
                r_dict[path_hash]['UserFile'] += 1
        #last_create_host = create_host
        #create_host = host
        if osize==0.0 and wb>0.0:
            r_dict[path_hash]['create_host'] = host

        #if osize == 0.0:
            #NewFileFlag += 1

        #path_hash = key[:32]
        if path_hash == old_path_hash:
            print 'Same file:', log_list[1].split('=')[1],' , continue', r_dict[path_hash]['BatchFile'], r_dict[path_hash]['UserFile']
            continue



        #if host.startswith('vm'):
            #BatchFile -= 1
        #else:
            #UserFile -= 1

        #if osize == 0.0:
            #NewFileFlag -= 1

        filename = log_list[1].split('=')[1]
        if filename.find('/eos/') == -1:
            continue
        index = filename.find('/#curl#/')
        if index != -1:
            filename = filename[7:]

        _filename = filename.split('/')
        #if _filename[2] != 'user':
            #continue

        """
        if NewFileFlag>0:
            r_list.extend([ruid, rgid, filename_suffix, filename_depth, filename_part2, filename_part3, filename_part4,
                            filename_part5, filename_part6, filename_part7, filename_part8, filename_part9, BatchFile>UserFile])
            csv_writer.writerow(r_list)
        """

        #r_list.extend([ruid, rgid, filename_suffix, filename_depth, last_create_host, filename_part2, filename_part3, filename_part4,
                       #filename_part5, filename_part6, filename_part7, filename_part8, filename_part9, BatchFile>UserFile])



        ruid = log_list[2].split('=')[1]
        r_dict[path_hash]['ruid'] = ruid

        rgid = log_list[3].split('=')[1]
        r_dict[path_hash]['rgid'] = rgid

        #csize = float(log_list[35].split('=')[1])

        if _filename[-1].find('.') > 0:
            filename_suffix = _filename[-1].split('.',1)[-1]
            r_dict[path_hash]['filename_suffix'] = filename_suffix
        filename_depth = len(_filename)
        r_dict[path_hash]['filename_depth'] = filename_depth

        #filename_part2, filename_part3, filename_part4, filename_part5, filename_part6, filename_part7, filename_part8, \
           #filename_part9 = 'none','none','none','none','none','none','none','none'
        if filename_depth>2:
            r_dict[path_hash]['filename_part2'] = _filename[-2]
        if filename_depth>3:
            r_dict[path_hash]['filename_part3'] = _filename[-3]
        if filename_depth>4:
            r_dict[path_hash]['filename_part4'] = _filename[-4]
        if filename_depth>5:
            r_dict[path_hash]['filename_part5'] = _filename[-5]
        if filename_depth>6:
            r_dict[path_hash]['filename_part6'] = _filename[-6]
        if filename_depth>7:
            r_dict[path_hash]['filename_part7'] = _filename[-7]
        if filename_depth>8:
            r_dict[path_hash]['filename_part8'] = _filename[-8]
        if filename_depth>9:
            r_dict[path_hash]['filename_part9'] = _filename[-9]

        #r_list.extend(
            #[ruid, rgid, filename_suffix, filename_depth, last_create_host, filename_part2, filename_part3,
            #filename_part4,filename_part5, filename_part6, filename_part7, filename_part8, filename_part9, BatchFile > UserFile])
        if old_path_hash=='':
            old_path_hash = path_hash
            continue
        r_list.extend([r_dict[old_path_hash]['ruid'],
                       r_dict[old_path_hash]['rgid'],
                       r_dict[old_path_hash]['filename_suffix'],
                       r_dict[old_path_hash]['filename_depth'],
                       r_dict[old_path_hash]['create_host'],
                       r_dict[old_path_hash]['filename_part2'],
                       r_dict[old_path_hash]['filename_part3'],
                       r_dict[old_path_hash]['filename_part4'],
                       r_dict[old_path_hash]['filename_part5'],
                       r_dict[old_path_hash]['filename_part6'],
                       r_dict[old_path_hash]['filename_part7'],
                       r_dict[old_path_hash]['filename_part8'],
                       r_dict[old_path_hash]['filename_part9'],
                       r_dict[old_path_hash]['BatchFile']>r_dict[old_path_hash]['UserFile']
                       ])


        if r_dict[old_path_hash]['BatchFile']>r_dict[old_path_hash]['UserFile']:
            if not ForceBalance:
                looptoken = 0
            if looptoken == 0:
                csv_Batchwriter.writerow(r_list)
                BatchFileNum += 1
                looptoken = 1
        else:
            if not ForceBalance:
                looptoken = 1
            if looptoken == 1:
                csv_Userwriter.writerow(r_list)
                UserFileNum += 1
                looptoken = 0
        del r_dict[old_path_hash]

        old_path_hash = path_hash
        #if host.startswith('vm'):
            #BatchFile = 1
            #UserFile = 0
        #else:
            #UserFile = 1
            #BatchFile = 0

        #if osize == 0.0:
            #NewFileFlag = 1
        #else:
            #NewFileFlag = 0


    pass

def main():
    csv_Batchfile = None
    csv_Bacthwriter = None
    csv_Userfile = None
    csv_Userwriter = None
    try:
        csv_Batchfile = open("csv/feature_batch.csv", 'w')
        csv_Bacthwriter = csv.writer(csv_Batchfile)
        csv_Userfile = open("csv/feature_user.csv", 'w')
        csv_Userwriter = csv.writer(csv_Userfile)
    except Exception as e:
        print e
    csv_Bacthwriter.writerow(["ruid","rgid","filename_suffix", "filename_depth", "create_host", "filename_part2","filename_part3","filename_part4",
                       "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Bacthwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])

    csv_Userwriter.writerow(
        ["ruid", "rgid", "filename_suffix", "filename_depth", "create_host","filename_part2", "filename_part3", "filename_part4",
         "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Userwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])

    rowStart = ScanHbase(rowStart=None, csv_Batchwriter=csv_Bacthwriter, csv_Userwriter=csv_Userwriter, limit=100000, ForceBalance=False)
    csv_Batchfile.close()
    csv_Userfile.close()

    try:
        csv_Batchfile = open("csv/test_batch.csv", 'w')
        csv_Bacthwriter = csv.writer(csv_Batchfile)
        csv_Userfile = open("csv/test_user.csv", 'w')
        csv_Userwriter = csv.writer(csv_Userfile)
    except Exception as e:
        print e
    csv_Bacthwriter.writerow(
        ["ruid", "rgid", "filename_suffix", "filename_depth", "create_host", "filename_part2", "filename_part3", "filename_part4",
         "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Bacthwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])
    csv_Userwriter.writerow(
        ["ruid", "rgid", "filename_suffix", "filename_depth", "create_host", "filename_part2", "filename_part3", "filename_part4",
         "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Userwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])

    rowStart = ScanHbase(rowStart=rowStart, csv_Batchwriter=csv_Bacthwriter, csv_Userwriter=csv_Userwriter,limit=10000,
                          ForceBalance=True)
    csv_Batchfile.close()
    csv_Userfile.close()

if __name__ == '__main__':
    #main()
    TimeEnd = 9999999999.99 - 1524585600.00
    TimeBegin = 9999999999.99 - 1522512000.00
    csv_Batchfile = None
    csv_Bacthwriter = None
    csv_Userfile = None
    csv_Userwriter = None
    try:
        csv_Batchfile = open("csv/feature_batch.csv", 'w')
        csv_Bacthwriter = csv.writer(csv_Batchfile)
        csv_Userfile = open("csv/feature_user.csv", 'w')
        csv_Userwriter = csv.writer(csv_Userfile)
    except Exception as e:
        print e
    csv_Bacthwriter.writerow(
        ["ruid", "rgid", "filename_suffix", "filename_depth", "create_host", "filename_part2", "filename_part3",
         "filename_part4",
         "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Bacthwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])

    csv_Userwriter.writerow(
        ["ruid", "rgid", "filename_suffix", "filename_depth", "create_host", "filename_part2", "filename_part3",
         "filename_part4",
         "filename_part5", "filename_part6", "filename_part7", "filename_part8", "filename_part9", "BatchFile"])
    csv_Userwriter.writerow(
        ["none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none"])



    statistics(TimeBegin=TimeBegin, TimeEnd=TimeEnd, csv_Batchwriter=csv_Bacthwriter, csv_Userwriter=csv_Userwriter)
