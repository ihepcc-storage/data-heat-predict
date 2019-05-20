#!/bin/env python

import happybase

def main():
    pass

if __name__ == '__main__':
    connection = happybase.Connection('192.168.60.64', autoconnect=True)


    """
    connection.create_table(
        'eoslog_UidOtsPathMap',
        {
          'cf':dict(max_versions=1)
        }
    )
    connection.create_table(
        'eoslog_UidPathOtsMap',
        {
          'cf':dict(max_versions=1)
        }
    )
    connection.create_table(
        'eoslog_PathOtsUidMap',
        {
            'cf': dict(max_versions=1)
        }
    )
    connection.create_table(
        'eoslog_OtsUidPathMap',
        {
            'cf': dict(max_versions=1)
        }
    )




    connection.create_table(
        'eos_log_index',
        {
            'name':dict(max_versions=1)
        }
    
    )
    connection.create_table(
        'eos_log',
        {
            'Id':dict(max_versions=1),
            'size':dict(max_versions=1),
            'cf_r':dict(max_versions=1),
            'cf_w':dict(max_versions=1),
            'seek':dict(max_versions=1)
        }
    )
    """

    OtsUidPathTable = connection.table("eoslog_OtsUidPathMap")
    PathOtsUidTable = connection.table("eoslog_PathOtsUidMap")


    #table = connection.table('eos_log_index')

    '''
    num = 0
    for key, data in table.scan(row_start='003eae5d1a6948d344097d97f9637ff61509833600.00',row_stop='003eae5d1a6948d344097d97f9637ff61530179200.00'):
        num += 1
        print key
    print num
    '''
    main()
