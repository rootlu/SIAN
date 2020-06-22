# coding: utf-8
# author: yf lu
# create date: 2019/7/10 14:53


import os, shutil
import sys


class Logging():
    def __init__(self, filename):
        self.filename = filename

    def record(self, str_log):
        filename = self.filename
        print(str_log)
        sys.stdout.flush()
        # with open(filename, 'a') as f:
        #     f.write("%s\r\n" % str_log)
        #     f.flush()
