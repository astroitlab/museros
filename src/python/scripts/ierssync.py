#!/usr/bin/env python

# import os,sys
# path = os.path.abspath(os.path.dirname(__file__))
#
# if path.find('python') == -1:
#     print("Cannot locate proper directory")
#     exit(0)
# path1 = path[0:path.find('python')+7]
#
# sys.path.append(path1)

from pymuser import museriers


if __name__ == '__main__':

    iers = museriers.MuserIers()

    iers.update()


