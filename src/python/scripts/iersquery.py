#! /usr/bin/env python

import sys
import datetime
import struct
import os
import string
from argparse import *

path = os.path.abspath(os.path.dirname(__file__))

if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]

sys.path.append(path1)

from pymuser.museriers import *

def main(argv=None):
    parser = ArgumentParser()

    parser.add_argument('-s', "--date", help="Date format YYYY-MM-DD", dest='search_date', required=True)

    args = parser.parse_args()

    start_date = valid_date(args.search_date)

    iers = MuserIers()
    print iers.query(start_date.date().year, start_date.date().month, start_date.date().day)


def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise ArgumentTypeError(msg)


main()