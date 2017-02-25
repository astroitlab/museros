#! /usr/bin/env python
# encoding: utf-8
"""
json_numpy.py
=============

Helper utilities from reading and writing JSON data with numpy arrays. The native JSON
encoder cannot handle numpy arrays and will chuck a hissy-fit, hence these helper functions.
"""

try:
    import ujson as json
    USES_UJSON = True
except ImportError:
    import json
    USES_UJSON = False

import numpy as np

def sanitize_json(npDict):
        """ Converts a dictionary of numpy arrays into a dictionary of lists """
        for key in npDict.keys():

            ktype = type(npDict[key])
            carr  = type(np.chararray([0]))
            darr  = type(np.array([0]))

            if ktype in [carr, darr]:
                try:
                    if ktype == carr:
                        npDict["_"+key] = "chararray"
                    else:
                        npDict["_"+key] = "ndarray %s"%npDict[key].dtype
                    npDict[key] = npDict[key].tolist()
                except ValueError:
                    print "ERROR: %s"%key
                    raise
        return npDict

def dump_json(npDict_in, filename_out):
    """ Dump a dictionary of numpy arrays to an output file """
    npDict = npDict_in.copy()
    if type(filename_out) is file:
        outfile = filename_out
    else:
        outfile = open(filename_out, 'w')

    npDict = sanitize_json(npDict)
    json.dump(npDict, outfile)
    npDict.clear()

def load_json(filename):
    """ Load a JSON file """
    if type(filename) is str:
        fileo = open(filename, 'r')
    else:
        fileo = filename

    data = json.load(fileo)

    to_pop = []
    for key in data:
        if key.startswith("_"):
            dtype_str = data[key]
            to_pop.append(key)
            okey = key.strip("_")

            if dtype_str == 'chararray':
                #TODO: This currently doesn't work
                #print 'chararray'
                #data[okey] = np.chararray(data[okey])
                pass
            elif dtype_str.split(" ")[0] == 'ndarray':
                data[okey] = np.array(data[okey], dtype=dtype_str.split(" ")[1])

    [data.pop(key) for key in to_pop]

    return data

