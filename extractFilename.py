import happybase

def main():
    connection = happybase.Connection('192.168.60.64', autoconnect=True)
    table = connection.table('eos_log')
    tableIndex = connection.table('eos_log_index')
    UidOtsPathTable = connection.table("eoslog_UidOtsPathMap")
    UidPathOtsTable = connection.table("eoslog_UidPathOtsMap")
    PathOtsUidTable = connection.table("eoslog_PathOtsUidMap")
    OtsUidPathTable = connection.table("eoslog_OtsUidPathMap")
    tmp = ''
    filename = ''
    FilenameFormat = {}
    Top10Filename = []
    num = 0


    """                                                           
    for key, data in table.scan():
        num += 1
        #if num>20000:
            #break
        if tmp != key[:32]:
            row = tableIndex.row(key[:32])
            if row == {}:
                continue
            filename = row['name:filename']
            filename = list(filename.split('/')[-1])

            index = 0
            for Achar in filename:
                if Achar.isdigit():
                    filename[index] = '0'
                index += 1
            filename =  ''.join(filename)
        #else:
            #print 'a'

        #rt = float(data['cf_r:rt'])
        seek = 0
        seek += float(data['seek:nfwds'])
        seek += float(data['seek:nbwds'])
        seek += float(data['seek:nxlfwds'])
        seek += float(data['seek:nxlbwds'])
        try:
            FilenameFormat[filename] += seek
        except Exception as e:
            FilenameFormat[filename] = seek
        tmp = key[:32]

        #if num>10000:
            #break

    Top10Filename = sorted(FilenameFormat.items(), lambda x, y:cmp(x[1], y[1]),reverse=True)
    Top10Filename = Top10Filename[:10]
    print Top10Filename
    """

    NewFileDict = {}
    for key, data in table.scan():
        num += 1
        #if num>10000:
            #break
        if tmp != key[:32]:
            row = tableIndex.row(key[:32])
            if row == {}:
                continue
            filename = row['name:filename']
            tmp = key[:32]
        OpenTime = float(key[32:])
        osize = float(data['size:osize'])
        csize = float(data['size:csize'])
        if osize==0 and csize>0:
            NewFileDict[filename]=[OpenTime, OpenTime, 1]
            continue

        try:
            if OpenTime<NewFileDict[filename][0]:
                NewFileDict[filename][0] = OpenTime
            if OpenTime>NewFileDict[filename][1]:
                NewFileDict[filename][1] = OpenTime
            NewFileDict[filename][2] += 1
        except Exception as e:
            continue

    #print NewFileDict
    TimeTop10Filename = sorted(NewFileDict.items(), lambda x, y: cmp(x[1][1]-x[1][0], y[1][1]-y[1][0]), reverse=True)
    TimeTop10Filename = TimeTop10Filename[:100]
    TimeTop10FilenameList = []
    for f in TimeTop10Filename:
        TimeTop10FilenameList.append(f[0])
    #print TimeTop10FilenameList

    NReadTop10Filename = sorted(NewFileDict.items(), lambda x, y: cmp(x[1][2], y[1][2]), reverse=True)
    NReadTop10Filename = NReadTop10Filename[:100]
    NReadTop10FilenameList = []
    for f in NReadTop10Filename:
        NReadTop10FilenameList.append(f[0])
    #print NReadTop10FilenameList

    for f in NReadTop10FilenameList:
        if f in TimeTop10Filename:
            print f
    #for f in TimeTop10Filename:
        #print f[1][1] - f[1][0]


    pass

if __name__ == '__main__':
    main()