#!/usr/bin/env python

import sys
import getopt

import musercli
import os

filepath = os.path.abspath(os.path.dirname(__file__))

homedir = os.getenv('MUSEROS_HOME')
if homedir==None:
    homedir = '~/museros'

print "Home directory: ", homedir

sys.stdout.flush()

try:
    opts, args = getopt.getopt(sys.argv[1:], \
                               "vdnhfg:c:", \
                               ["vervose", "debug", "nopasswd", "help",
                                "file=", "conf","server","gui","client"])
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    musercli.usage()
    sys.exit(2)

musercli.main(opts, args)
