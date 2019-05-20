#!/bin/env python
import sys,os
import json
import traceback


def JSONParse(InputStr=""):
    try:
        jsonObj = json.loads(InputStr)
        return jsonObj
    except Exception as e:
        print e
        traceback.print_exc()
        return -1
    pass
