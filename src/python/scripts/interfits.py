#! /usr/bin/env python
# encoding: utf-8
"""
interfits.py
============

Python class which reads a variety of different visibility formats.
Currently reads UV-FITS and FITS-IDI.
"""

import sys
import os
import re

import pyfits as pf
from lxml import etree


#from pyFitsidi import *
from src.python.pymuser.json_numpy import *

class LinePrint():
    """
    Print things to stdout on one line dynamically
    """

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def h1(headstr):
    """ Print a string as a header """
    print '\n', headstr
    underline = ''
    for i in range(len(headstr)):
        underline += '-'
    print underline

def h2(headstr):
    """ Print a string as a header """
    print '\n###  ', headstr


class VerificationError(Exception):
    """ Custom data verification exception """
    pass


class InterFits(object):
    """ InterFits: UV-data interchange class
    """
    def __init__(self, filename=None):
        self.filename = filename

        # Set up some basic details
        self.telescope = ""
        self.instrument = ""
        self.correlator = ""
        self.source = ""
        self.date_obs = ""
        self.obs_code = ""

        # Set up dictionaries to store data
        self.h_antenna = {}
        self.h_array_geometry = {}
        self.h_source = {}
        self.h_uv_data = {}
        self.h_frequency = {}
        self.h_common = {}
        self.h_params = {}

        self.d_antenna = {}
        self.d_array_geometry = {}
        self.d_source = {}
        self.d_uv_data = {}
        self.d_frequency = {}

        self.stokes_codes = {
            1: 'Stokes I',
            2: 'Stokes Q',
            3: 'Stokes U',
            4: 'Stokes V',
            -1: 'RR',
            -2: 'LL',
            -3: 'RL',
            -4: 'LR',
            -5: 'XX',
            -6: 'YY',
            -7: 'XY',
            -8: 'YX'
        }

        if filename:
            self.readFile(filename)

    def __repr__(self):
        to_print = ""
        to_print += "Telescope:  %s\n" % self.telescope
        to_print += "Instrument: %s\n" % self.instrument
        to_print += "Object:     %s\n" % self.source
        to_print += "Date obs:   %s\n" % self.date_obs
        return to_print

    def readFile(self, filename):
        # Check what kind of file to load
        if filename:
            regex = '([0-9A-Za-z-_]+).uvfits'
            match = re.search(regex, filename)
            if match: self._readFile('uvfits')

            regex = '([0-9A-Za-z-_]+).fitsidi'
            match = re.search(regex, filename)
            if match: self._readFile('fitsidi')

            regex = '([0-9A-Za-z-_]+).hdf'
            match = re.search(regex, filename)
            if match: self._readFile('hdf5')

    def _readFile(self, filetype):
        """ Lookup dictionary (case statement) for file types """
        return {
            'uvfits': self.readUvfits,
            'fitsidi': self.readFitsidi,
            'hdf5': self.readHdf5
        }.get(filetype)()

    def searchKeys(self, pattern, header):
        """ Search through a header, returning a list of matching keys 
        
        pattern: str
            regular expression pattern to match
        header: pyfits header
            header of a pyfits HDU to search within
        """

        matches = []
        keys = header.keys()

        for k in keys:
            m = re.search(pattern, k)
            if m:
                matches.append(m.string)
        matches.sort()
        return matches

    def readUvfits(self):
        """ Open and read the contents of a uv-fits file 
        
        Notes
        -----
        
        Regular axes for Data matrix
        (from FITS-IDI document, assuming uv-fits adheres)
        
        Name        Mandatory   Description
        -------     ---------   -----------
        COMPLEX     yes         Real, imaginary, weight
        STOKES      yes         Stokes parameter
        FREQ        yes         Frequency (spectral channel)
        BAND        no          Band number
        RA          yes         Right ascension of phase center
        DEC         yes         Declination of phase center
        -------     ---------   -----------

        """

        h1("Opening uvfits data")

        self.fits = pf.open(self.filename)
        self.uvdata = self.fits[0].data
        self.uvhead = self.fits[0].header

        self.antdata = self.fits[1].data
        self.anthead = self.fits[1].header

        # Load basic metadata
        self.telescope = self.uvhead['TELESCOP'].strip()
        self.instrument = self.uvhead['INSTRUME'].strip()
        self.source = self.uvhead['OBJECT'].strip()
        self.date_obs = self.uvhead['DATE-OBS'].strip()
        self.n_ant = self.antdata.shape[0]

        print self.fits.info()
        print "Telescope:  %s" % self.telescope
        print "Instrument: %s" % self.instrument
        print "Object:     %s" % self.source
        print "Date obs:   %s" % self.date_obs

        # Load array geometry data
        h2("Loading array geometry")
        ag_keywords = ['ARRAYX', 'ARRAYY', 'ARRAYZ', 'ARRNAM', 'FREQ']
        ag_data = ['ANNAME', 'STABXYZ', 'NOSTA', 'MNTSTA', 'STAXOF']
        for k in ag_keywords:
            self.h_array_geometry[k] = self.anthead[k]
        for k in ag_data:
            self.d_array_geometry[k] = self.antdata[k]

        self.h_common['RDATE'] = self.anthead['RDATE']

        # Load antenna table data
        h2("Loading antenna table")
        an_keywords = ['NOPCAL']
        an_data = ['POLTYA', 'POLAA', 'POLCALA', 'POLTYB', 'POLAB', 'POLCALB']
        for k in an_keywords:
            self.h_antenna[k] = self.anthead[k]
        for k in an_data:
            self.d_antenna[k] = self.antdata[k]

        # Load frequency table data
        # This is the first of the non-straightforward conversions
        # Need to find frequency index. It's probably CTYPE4 but safer to look it up
        h2("Loading frequency table")
        ctypes = self.searchKeys('CTYPE\d', self.uvhead)
        c = [(ct, self.uvhead[ct], self.uvhead['NAXIS%s' % ct.lstrip('CTYPE')]) for ct in ctypes]
        freq_cid = c[[x[1] == 'FREQ' for x in c].index(True)][0].lstrip('CTYPE')

        ch_width = float(self.uvhead['CDELT%s' % freq_cid])
        ref_freq = float(self.uvhead['CRVAL%s' % freq_cid])
        ref_pixl = float(self.uvhead['CRPIX%s' % freq_cid])
        bw = ch_width * int(self.uvhead['NAXIS%s' % freq_cid])
        self.h_common['REF_FREQ'] = ref_freq
        self.h_common['CHAN_BW'] = ch_width
        self.h_common['REF_PIXL'] = ref_pixl

        self.d_frequency['TOTAL_BANDWIDTH'] = bw
        self.d_frequency['CH_WIDTH'] = ch_width

        # Load source table data
        h2("Loading source table")
        self.d_source['SOURCE'] = self.source
        self.d_source['RAEPO'] = self.uvhead['OBSRA']
        self.d_source['DECEPO'] = self.uvhead['OBSDEC']

        # Load UV-DATA
        h2("Loading UV-data")
        self.h_uv_data['TELESCOP'] = self.telescope
        uv_datacols = ['UU', 'VV', 'WW', 'BASELINE', 'DATE']
        for k in uv_datacols: self.d_uv_data[k] = self.uvdata[k]

        s = self.uvdata['DATA'].shape

        # Find stokes axis type and values
        stokes_axid = 0
        for ct in ctypes:
            if self.uvhead[ct].strip() == 'STOKES':
                stokes_axid = int(ct.lstrip('CTYPE'))
                break

        stokes_axis_len = int(self.uvhead['NAXIS%i' % stokes_axid])
        stokes_code = int(self.uvhead['CRVAL%i' % stokes_axid])
        stokes_delt = int(self.uvhead['CDELT%i' % stokes_axid])
        stokes_vals = range(stokes_code, stokes_code + stokes_delt * stokes_axis_len, stokes_delt)
        self.stokes_vals = stokes_vals
        self.stokes_axis = [self.stokes_codes[i] for i in stokes_vals]

        # Should have 7 axes, likely
        if len(s) != 7:
            if len(s) == 6:
                #print "Reshaping uv-data..."
                new_shape = (s[0], s[1], s[2], 1, s[3], s[4], s[5])
                self.d_uv_data['DATA'] = self.uvdata['DATA'].reshape(new_shape)
            else:
                print "ERROR: Data axis not understood, incorrect length"
                print c
                raise
        else:
            self.d_uv_data['DATA'] = self.uvdata['DATA']

        # Best line in the history of indexing below
        # Note the 0:2 and *2 at the end is to not include weights
        print "Converting DATA column to FLUX convention..."
        self.d_uv_data['FLUX'] = self.d_uv_data['DATA'][..., 0:2].astype('float32').reshape(s[0], s[3] * s[4] * 2)

        self.h_params["NSTOKES"] = len(self.stokes_vals)
        self.h_params["NBAND"] = self.d_uv_data['DATA'].shape[-4]
        self.h_params["NCHAN"] = self.d_uv_data['DATA'].shape[-3]

        self.d_uv_data.pop('DATA')

        num_rows = self.d_uv_data['FLUX'].shape[0]

        print "NOTE: Setting INTTIM to 1.0 (not supplied by uvfits)."
        print "NOTE: Setting FREQID to 1 (for FITS-IDI tables)"
        print "NOTE: Setting SOURCE to 1 (for FITS-IDI tables)"
        self.d_uv_data['INTTIM'] = np.ones(num_rows)
        self.d_uv_data['FREQID'] = np.ones(num_rows)
        self.d_uv_data['SOURCE'] = np.ones(num_rows)

    def readFitsidi(self, from_file=True, load_uv_data=True):
        """ Open and read the contents of a FITS-IDI file

        Notes
        -----

        Regular axes for Data matrix (from FITS-IDI document)

        Name        Mandatory   Description
        -------     ---------   -----------
        COMPLEX     yes         Real, imaginary, weight
        STOKES      yes         Stokes parameter
        FREQ        yes         Frequency (spectral channel)
        BAND        no          Band number
        RA          yes         Right ascension of phase center
        DEC         yes         Declination of phase center
        -------     ---------   -----------

        """

        if from_file:
            h1("Opening FITS-IDI data")
            self.fits = pf.open(self.filename)
        else:
            pass  # Assumes that self.fits is already populated

        # Match tables
        for tbl in self.fits:
            try:
                if tbl.header['EXTNAME'] == 'ARRAY_GEOMETRY':
                    self.tbl_array_geometry = tbl
                elif tbl.header['EXTNAME'] == 'FREQUENCY':
                    self.tbl_frequency = tbl
                elif tbl.header['EXTNAME'] == 'ANTENNA':
                    self.tbl_antenna = tbl
                elif tbl.header['EXTNAME'] == 'SOURCE':
                    self.tbl_source = tbl
                elif tbl.header['EXTNAME'] == 'UV_DATA':
                    self.tbl_uv_data = tbl
                else:
                    print "Warning: %s not recognized" % tbl.header["EXTNAME"]
            except KeyError:
                pass

        # Load basic metadata
        if load_uv_data:
            self.telescope = self.tbl_uv_data.header['TELESCOP'].strip()
            try:
                self.date_obs = self.tbl_uv_data.header['DATE-OBS'].strip()
            except AttributeError:
                self.date_obs = 0.0
        else:
            self.telescope, self.date_obs = '', 0.0
        self.instrument = self.tbl_array_geometry.header['ARRNAM'].strip()
        self.source = str(self.tbl_source.data['SOURCE'][0]).lstrip("\"[\'").rstrip("\']\"")
        self.n_ant = self.tbl_antenna.data.shape[0]

        if from_file:
            print self.fits.info()
            print "Telescope:  %s" % self.telescope
            print "Instrument: %s" % self.instrument
            print "Object:     %s" % self.source
            print "Date obs:   %s" % self.date_obs

        # Load array geometry data
        h2("Loading array geometry")
        ag_keywords = ['ARRAYX', 'ARRAYY', 'ARRAYZ', 'ARRNAM', 'FREQ']
        ag_data = ['ANNAME', 'STABXYZ', 'NOSTA', 'MNTSTA', 'STAXOF']
        for k in ag_keywords:
            self.h_array_geometry[k] = self.tbl_array_geometry.header[k]
        for k in ag_data:
            self.d_array_geometry[k] = self.tbl_array_geometry.data[k]


        # Load antenna table data
        h2("Loading antenna table")
        an_keywords = ['NOPCAL']
        an_data = ['POLTYA', 'POLAA', 'POLCALA', 'POLTYB', 'POLAB', ' POLCALB']
        for k in an_keywords: self.h_antenna[k] = self.tbl_antenna.header[k]
        for k in an_data:
            try:
                self.d_antenna[k] = self.tbl_antenna.data[k]
            except KeyError:
                print "\tWARNING: %s key error raised." % k

        # Load frequency table data
        # This is the first of the non-straightforward conversions
        h2("Loading frequency table")
        frq_keywords = ['FREQID', 'BANDFREQ', 'CH_WIDTH', 'TOTAL_BANDWIDTH', 'SIDEBAND']
        for k in frq_keywords:
            try:
                self.d_frequency[k] = self.tbl_frequency.data[k]
            except KeyError:
                print "\tWARNING: %s key error raised." % k

        # Load source table data
        h2("Loading source table")
        src_keywords = ['SOURCE_ID', 'SOURCE', 'QUAL', 'CALCODE', 'FREQID', 'IFLUX',
                        'QFLUX', 'UFLUX', 'VFLUX', 'ALPHA', 'FREQOFF', 'RAEPO', 'DECEPO',
                        'EQUINOX', 'RAAPP', 'DECAPP', 'SYSVEL', 'VELTYP', 'VELDEF', 'RESTFREQ',
                        'PMRA', 'PMDEC', 'PARALLAX']
        for k in src_keywords:
            try:
                self.d_source[k] = self.tbl_source.data[k]
            except KeyError:
                print "\tWARNING: %s key error raised." % k

        # Load common (mandatory) keywords
        h2("Loading common keywords")
        com_keywords = ['STK_1', 'NO_BAND', 'NO_STKD', 'REF_PIXL', 'REF_FREQ', 'CHAN_BW', 'NO_CHAN', 'RDATE']
        for k in com_keywords:
            try:
                self.h_common[k] = self.tbl_frequency.header[k]
            except KeyError:
                print "\tWARNING: %s key error raised." % k

        # Also fill in the parameter header dictionary (needed for XML generation).
        self.h_params["NSTOKES"] = self.h_common["NO_STKD"]
        self.h_params["NBAND"] = self.h_common["NO_BAND"]
        self.h_params["NCHAN"] = self.h_common["NO_CHAN"]

        # Load UV-DATA
        if load_uv_data:
            h2("Loading UV-data")
            uv_keywords = ['TELESCOP']
            for k in uv_keywords:
                try:
                    self.h_uv_data[k] = self.tbl_uv_data.header[k]
                except KeyError:
                    print "\tWARNING: %s key error raised." % k

            uv_datacols  = ['UU', 'VV', 'WW', 'BASELINE', 'DATE', 'FLUX', 'INTTIM', 'FREQID', 'SOURCE']
            for k in uv_datacols:
                self.d_uv_data[k] = self.tbl_uv_data.data[k]

            self.d_uv_data["FLUX"] = self.d_uv_data["FLUX"].astype('float32')

            # Find stokes axis type and values
            stokes_axid = 0

            ctypes = self.searchKeys('CTYPE\d', self.tbl_uv_data.header)
            for ct in ctypes:
                if self.tbl_uv_data.header[ct].strip() == 'STOKES': # WF: Can't understand
                    stokes_axid = int(ct.lstrip('CTYPE'))
                    break
            #stokes_axid = len(ctypes)
            stokes_axis_len = int(self.tbl_uv_data.header['MAXIS%i' % stokes_axid])
            print stokes_axis_len,"\n"
            stokes_code = int(self.tbl_uv_data.header['CRVAL%i' % stokes_axid])
            stokes_delt = int(self.tbl_uv_data.header['CDELT%i' % stokes_axid])
            stokes_vals = range(stokes_code, stokes_code + stokes_delt * stokes_axis_len, stokes_delt)
            self.stokes_vals = stokes_vals
            self.stokes_axis = [self.stokes_codes[i] for i in stokes_vals]


    def readHdf5(self):
        """ TODO """
        raise Exception("HDF5 support not yet implemented.")


    def setXml(self, table, keyword, value):
        """ Find a header parameter and replace it """
        try:
            self.xmlroot = self.xmlData.getroot()
            if type(value) == str:
                # Make sure strings are in single quotes
                value = "'" + value.strip("'") + "'"
            self.xmlroot.find(table).find(keyword).text = str(value)
        except:
            print "Error: Something went wrong with XML parsing"
            print "%s, %s, %s" % (table, keyword, value)

    def s2arr(self, val):
        """ Put a single value into a numpy array """
        return np.array([val])

    def setDefaults(self, n_uv_rows):
        """ FIll headers and data with default data """

        zero_vec = np.zeros(n_uv_rows).astype('float32')
        ones_vec = np.ones(n_uv_rows).astype('float32')
        self.d_uv_data["DATE"] = zero_vec
        self.d_uv_data["UU"] = zero_vec
        self.d_uv_data["VV"] = zero_vec
        self.d_uv_data["WW"] = zero_vec
        self.d_uv_data["FREQID"] = ones_vec
        self.d_uv_data["INTTIM"] = ones_vec
        self.d_uv_data["SOURCE"] = ones_vec


        self.stokes_axis = ['XX', 'YY', 'XY', 'YX']
        self.stokes_vals = [-5, -6, -7, -8]

        self.d_array_geometry["ANNAME"] = \
            np.array(["Stand%d"%(i + 1) for i in range(len(self.d_array_geometry["ANNAME"]))])
        self.d_array_geometry["NOSTA"]  = \
            np.array([i + 1 for i in range(len(self.d_array_geometry["NOSTA"]))])

        self.d_frequency["FREQID"]    = self.s2arr(1)
        self.d_frequency["BANDFREQ"]  = self.s2arr(0)

        self.d_source["EQUINOX"]   = self.s2arr('J2000')
        self.d_source["SOURCE"]    = self.s2arr('ZENITH')
        self.d_source["SOURCE_ID"] = self.s2arr(1)

    def generateFitsidiXml(self, xmlbase=None, filename_out=None):
        """ Generate XML file that encodes fitsidi structure 
        
        xmlbase: str
            name of basic input xml file
        filename_out: str
            name of output file
        """

        if xmlbase is None:
            dirname, filename = os.path.split(os.path.abspath(__file__))
            xmlbase = os.path.join(dirname, 'config/config.xml')

        self.xmlData = etree.parse(xmlbase)

        # Look in the config file or fits-idi convention for
        # info about what these should be and refer to
        self.setXml("PARAMETERS", "NSTOKES", self.h_params["NSTOKES"])
        self.setXml("PARAMETERS", "NBAND", self.h_params["NBAND"])
        self.setXml("PARAMETERS", "NCHAN", self.h_params["NCHAN"])

        # Common headers - required for each table
        self.setXml("COMMON", "OBSCODE", self.obs_code)
        self.setXml("COMMON", "STK_1", self.stokes_vals[0])
        self.setXml("COMMON", "REF_FREQ", self.h_common['REF_FREQ'])
        self.setXml("COMMON", "CHAN_BW", self.h_common['CHAN_BW'])
        self.setXml("COMMON", "REF_PIXL", self.h_common["REF_PIXL"])
        self.setXml("COMMON", "RDATE", self.h_common["RDATE"])

        self.setXml("ANTENNA", "NOPCAL", self.h_antenna["NOPCAL"])

        # Support tables
        self.setXml("PRIMARY", "CORRELAT", self.correlator)
        self.setXml("ARRAY_GEOMETRY", "ARRNAM", self.h_array_geometry["ARRNAM"])

        self.setXml("ARRAY_GEOMETRY", "FREQ", self.h_array_geometry['FREQ'])
        self.setXml("ARRAY_GEOMETRY", "ARRAYX", self.h_array_geometry["ARRAYX"])
        self.setXml("ARRAY_GEOMETRY", "ARRAYY", self.h_array_geometry["ARRAYY"])
        self.setXml("ARRAY_GEOMETRY", "ARRAYZ", self.h_array_geometry["ARRAYZ"])

        # UV-DATA
        self.setXml("UV_DATA", "DATE-OBS", self.date_obs)
        self.setXml("UV_DATA", "TELESCOP", self.telescope)
        stokes_delt = self.stokes_vals[1] - self.stokes_vals[0]
        self.setXml("UV_DATA", "CDELT2", stokes_delt)
        self.setXml("UV_DATA", "CRVAL2", self.stokes_vals[0])
        if type(self.d_frequency['CH_WIDTH']) is np.ndarray:
            self.setXml("UV_DATA", "CDELT3", self.d_frequency['CH_WIDTH'][0])
        else:
            self.setXml("UV_DATA", "CDELT3", self.d_frequency['CH_WIDTH'])
        self.setXml("UV_DATA", "CRVAL3", self.h_common['REF_FREQ'])

        if filename_out:
            if os.path.isfile(filename_out):
                os.remove(filename_out)
            print "Writing to %s" % filename_out
            with open(filename_out, 'w') as f:
                f.write(etree.tostring(self.xmlData))

    def exportFitsidi(self, filename_out, config_xml=None):
        """ Export data as FITS IDI 
        
        filename_out: str
            output filename
        config_xml: str
        config_xml: str
            path to config file
        
        """

        h1('Generating FITS-IDI XML schema')
        if config_xml is None:
            dirname, this_file = os.path.split(os.path.abspath(__file__))
            config_xml = os.path.join(dirname, 'config/config.xml')
        xmlfile = filename_out.replace(".fitsidi", "").replace(".fits", "") + ".xml"
        self.generateFitsidiXml(config_xml, xmlfile)
        config_xml = xmlfile


        h1('Creating Primary HDU')
        hdu_primary = make_primary(config=config_xml)
        print hdu_primary.header.ascardlist()

        h1('\nCreating ARRAY_GEOMETRY')
        tbl_array_geometry = make_array_geometry(config=config_xml, num_rows=self.n_ant)
        print tbl_array_geometry.header.ascardlist()

        h1('\nCreating ANTENNA')
        tbl_antenna = make_antenna(config=config_xml, num_rows=self.n_ant)
        print tbl_antenna.header.ascardlist()

        h1('\nCreating FREQUENCY')
        tbl_frequency = make_frequency(config=config_xml, num_rows=1)
        print tbl_frequency.header.ascardlist()

        h1('\nCreating SOURCE')
        tbl_source = make_source(config=config_xml, num_rows=1)
        print tbl_source.header.ascardlist()

        h1('\nCreating UV_DATA')
        num_rows = self.d_uv_data['FLUX'].shape[0]
        uvd = self.d_uv_data
        if type(uvd['BASELINE']) is list:
            uvd['BASELINE'] = np.array(uvd['BASELINE'])

        # TODO: Fix time and date to julian date
        tbl_uv_data = make_uv_data(config=config_xml, num_rows=num_rows,
                                   uu_data=uvd['UU'], vv_data=uvd['VV'], ww_data=uvd['WW'],
                                   date_data=uvd['DATE'], time_data=None, baseline_data=uvd['BASELINE'].astype('int32'),
                                   source_data=uvd['SOURCE'], freqid_data=uvd['FREQID'], inttim_data=uvd['INTTIM'],
                                   weights_data=None, flux_data=uvd['FLUX'], weights_col=False)

        print tbl_uv_data.header.ascardlist()

        h1('Filling in data')
        h2("ARRAY_GEOMETRY")
        for i in range(self.n_ant):
            for k in ['ANNAME', 'STABXYZ', 'NOSTA', 'MNTSTA', 'STAXOF']:
                tbl_array_geometry.data[k][i] = self.d_array_geometry[k][i]

        h2("ANTENNA")
        for i in range(self.n_ant):
            # TODO: 'POLCALB' and POLCALA               
            for k in ['POLTYA', 'POLAA', 'POLTYB', 'POLAB']:
                try:
                    tbl_antenna.data['ANNAME'][i] = self.d_array_geometry['ANNAME'][i]
                    tbl_antenna.data['ANTENNA_NO'][i] = i + 1
                    tbl_antenna.data['ARRAY'][i] = 1
                    tbl_antenna.data['FREQID'][i] = 1
                    tbl_antenna.data['NO_LEVELS'][i] = 255
                    tbl_antenna.data[k][i] = self.d_antenna[k][i]
                except:
                    print "Warning: keyword error: %s" % k

        h2("FREQUENCY")
        tbl_frequency.data["FREQID"][0] = 1
        tbl_frequency.data['BANDFREQ'][0] = 0
        tbl_frequency.data['CH_WIDTH'][0] = self.d_frequency['CH_WIDTH']
        tbl_frequency.data['TOTAL_BANDWIDTH'][0] = self.d_frequency['TOTAL_BANDWIDTH']

        h2("SOURCE")
        if type(self.d_source['SOURCE']) == str:
            for k in ['SOURCE', 'RAEPO', 'DECEPO']:
                try:
                    tbl_source.data['SOURCE_ID'][0] = 1
                    tbl_source.data['EQUINOX'][0] = 'J2000'
                    tbl_source.data[k][0] = self.d_source[k]
                except:
                    print "Warning: keyword error: %s" % k
                    raise
        else:
            n_rows = len(self.d_source['SOURCE'])
            for i in range(n_rows):
                tbl_source.data['SOURCE_ID'][i] = i + 1
                tbl_source.data['EQUINOX'][i] = 'J2000'
                for k in ['SOURCE', 'RAEPO', 'DECEPO']:
                    try:
                        tbl_source.data[k][i] = self.d_source[k][i]
                    except:
                        print "Warning: keyword error: %s" % k

        h2("UV_DATA")
        print "Pre-filled"
        # NOTE: This is now superfluous, thanks to the make_uv_data call above
        #for i in range(self.d_uv_data['DATA'].shape[0]):
        #    LinePrint("Row %i of %i"%(i+1, self.d_uv_data['DATA'].shape[0]))
        #    for k in ['UU','VV','WW','BASELINE','DATE']:
        #        try:
        #            tbl_uv_data.data[k][i] = self.d_uv_data[k][i]
        #            #tbl_uv_data.data['FLUX'][i] = self.d_uv_data['FLUX'][i]
        #        except:
        #            raise

        h1('Creating HDU list')
        hdulist = pf.HDUList(
            [hdu_primary,
             tbl_array_geometry,
             tbl_frequency,
             tbl_antenna,
             tbl_source,
             tbl_uv_data
            ])
        print hdulist.info()

        print '\nVerifying integrity...'
        hdulist.verify()

        if os.path.isfile(filename_out):
            print 'Removing existing file %s...' % filename_out
            os.remove(filename_out)
        print 'Writing to file %s...' % filename_out
        hdulist.writeto(filename_out)

    def verify_baseline_order(self):
        """ Check baseline IDs are in order """

        print "Verification: Checking uv_data baseline order..."

        bls = [int(b) for b in self.d_uv_data['BASELINE']]

        # Generate lower an upper triangular matrices
        bl_lower, bl_upper = [], []
        for i in range(1, self.n_ant + 1):
            for j in range(1, self.n_ant + 1):
                if j >= i:
                    bl_lower.append(256 * i + j)
                elif j <= i:
                    bl_upper.append(256 * j + i)

        # Check every baseline is right, over all dumps
        n_bls = self.n_ant * (self.n_ant - 1) / 2 + self.n_ant
        n_dumps = len(bls) / n_bls

        lower_t, upper_t = False, False
        for i in range(n_dumps):
            if bls[i * n_bls:(i + 1) * n_bls] == bl_lower:
                #print "LOWER"
                upper_t = False
                lower_t = True
            if bls[i * n_bls:(i + 1) * n_bls] == bl_upper:
                #print "UPPER"
                lower_t = False
                upper_t = True

            if not lower_t and not upper_t:
                raise VerificationError("Baseline order neither upper or lower triangular.")

        if lower_t:
            print "Verification: OK. Baselines in lower triangular order."
        if upper_t:
            print "Verification: OK. Baselines in upper triangular order."

        return True

    def verify(self):
        """ Run a series of diagnostics to test data validity """
        h1("Data verification")
        self.verify_baseline_order()

    def remove_miriad_baselines(self):
        """ Remove baseline data for all antennas with IDs > 255

        Miriad-type UVFITS files use the convention
            ant1*256+ant2 if ants < 255
            ant1*2048+ant2+65536 if ants >255
        The miriad convention screws up import into many reduction packages.
        """

        bls = self.d_uv_data["BASELINE"]
        max_bl = 255 * 256 + 255
        ok_bls = bls < max_bl
        for k in self.d_uv_data.keys():
            self.d_uv_data[k] = self.d_uv_data[k][ok_bls]

    def formatStokes(self):
        """ Return data as complex stokes vector """

        data = self.d_uv_data["FLUX"]

        xx_data = data[:, 0::8] + 1j * data[:, 1::8]
        yy_data = data[:, 2::8] + 1j * data[:, 3::8]
        xy_data = data[:, 4::8] + 1j * data[:, 5::8]
        yx_data = data[:, 6::8] + 1j * data[:, 7::8]

        return np.array((xx_data, yy_data, xy_data, yx_data)).astype('complex128')

    def get_antenna_id(self, bl_id):
        """ Convert baseline ID into an antenna pair.

        Uses MIRIAD convention for antennas > 256
        Returns a tuple of antenna IDs.
        """
        if bl_id > 65536:
            ant1 = (bl_id - 65536) / 2048
            ant2 = (bl_id - 65536) % 2048
        else:
            ant1 = bl_id / 256
            ant2 = bl_id % 256
        return ant1, ant2

    def get_baseline_id(self, ant1, ant2):
        """ Convert antenna pair into baseline ID """
        if ant1 > 255 or ant2 > 255:
            bl_id = ant1 * 2048 + ant2 + 65536
        else:
            bl_id = ant1 * 256 + ant2
        return bl_id

    def generateBaselineIds(self, n_ant):
        """ Generate a list of baseline IDs from
        """
        bls, ant_arr = [], []
        for ii in range(1, n_ant + 1):
            for jj in range(1, n_ant + 1):
                if jj >= ii:
                    ant_arr.append((ii, jj))
                    if ii > 255 or jj > 255:
                        bl_id = ii * 2048 + jj + 65536
                    else:
                        bl_id = 256 * ii + jj
                    bls.append(bl_id)
        return bls, ant_arr