#! /usr/bin/env python
# encoding: utf-8
"""
Creates a UVFITS file for CSRH-I.
"""
from __future__ import division
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
if path.find('python') == -1:
    print("Cannot locate proper directory")
    exit(0)
path1 = path[0:path.find('python')+7]
sys.path.append(path1)

from pymuser.muserclean import *
from argparse import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help="The UVFITS/FITSIDI file", dest='infile',
                        default="", required=True)
    parser.add_argument('-o', '--outfile', help="The output directory", dest='outfile',
                        default="", required=True)
    parser.add_argument('-t', '--fits', help="Generate fits file", dest='writefits', action='store_true',default=False)

    parser.add_argument('-d', '--debug', help="Debug", dest="debug", action='store_true', default=False)

    parser.add_argument('-p', '--plot', help="Generate png file", dest='plot', action='store_true', default=False)

    parser.add_argument('-c', '--correct', help="Correct Polar Angle", dest='polar_angle', action='store_true', default=False)


    args = parser.parse_args()
    parser.set_defaults(debug=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(writefits=False)
    infile = args.infile
    plotme = args.plot
    outdir = args.outfile
    writefits = args.writefits
    p_angle = args.polar_angle

    #Flag_Ant = [0, 4, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 37, 38, 39]
    #Flag_Ant = [8, 9,10, 11, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39] #20151101

    #Flag_Ant  = [4, 7, 10, 11,12,13,16,17,18, 19, 24, 25, 26, 36, 38, 39]   #20141111
               # [5 8 11 12 13 14 17 18 19 20 25 26 27 37 39 40];ant
    if not os.path.exists(infile):
        print "No file exist."
        exit()
    if infile != None:
        clean = MuserClean()
        clean.clean_with_fits(infile, outdir, plotme, writefits,p_angle, DEBUG=args.debug)


