# encoding: utf-8
"""
pyuvfits.py for muser-I/II
===========================

UVFITS Table Overview
~~~~~~~~~~~~~~~~~~~~~~~~

Here is a quick rundown of the tables that this script can make.
* **Primary**           Visibility data
* **AIPS AN**           Antenna polarization information
* **AIPS FQ**           Frequency setups
* **AIPS SU**           Sources observed
Module listing
~~~~~~~~~~~~~~

"""

import os
from operator import itemgetter
import math
import pyfits as pf
import numpy as np
from muserant import muserantenna
from muserephem import *

light_speed = 299792458  # Speed of light
PI = math.pi


np.set_printoptions(threshold='nan')


class Array:
    """ An antenna array class.

        Based on pyEphem's Observer class.
        Probably very similar to the one in AIPY.
        Parameters
        ----------
        lat: dd:mm:ss
          latitude of array centre, e.g. 44:31:24.88
        long: dd:mm:ss
          longitude of array centre, e.g. 11:38:45.56
        elev: float
          elevation in metres of array centrem e.g. 28.0
        antennas: np.array([x,y,z])
          numpy array of antenna positions, in xyz coordinates in meters,
          relative to the array centre.
    """

    def __init__(self, lat, long, elev, antennas):
        # super(Array, self).__init__()
        self.lat = lat
        self.long = long
        self.elev = elev
        self.antennas = antennas


class PyUVFITS:
    def __init__(self, config, latitude, longitude, altitude, muser):
        self.config = config
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.muser = muser
        self.ephem = MuserEphem()
        self.source_dict = []
        self.antenna_dict = []
        self.frequency_dict = []
        self.baselie_dict = []
        self.visibility_data = np.zeros(shape=(self.muser.antennas * (self.muser.antennas - 1) / 2, 1, 1, 16, 2, 3), dtype='float32')
        self.datauu = []
        self.datavv = []
        self.dataww = []
        self.databaseline = []
        self.datadate1=[]
        self.datadate2=[]
        self.datasource=[]
        self.datafreqsel=[]
        self.last_channel = 99

    def reset_dict (self):
        self.source_dict = []
        self.antenna_dict = []
        self.frequency_dict = []
        self.data_dict = []
        self.visibility_data = np.zeros(shape=(self.muser.antennas * (self.muser.antennas - 1) / 2, 1, 1, 16, 2, 3), dtype='float32')
        self.datauu = []
        self.datavv = []
        self.dataww = []
        self.databaseline = []
        self.datadate1=[]
        self.datadate2=[]
        self.datasource=[]
        self.datafreqsel=[]
        self.last_channel = 99
        self.source_id = 0

    def make_primary(self):
        """  Creates the primary header data unit (HDU).

        This function generates header keywords from the file headers/primary.tpl

        Parameters
        ----------
        config: string
          filename of xml configuration file, defaults to 'config,xml'
        """

        # Make a new FITS HDU muser.DROutputAntennas*(muser.DROutputAntennas-1)/2=60*60/2
        imdata = np.arange(self.muser.antennas * (self.muser.antennas - 1) / 2 * 16 * 3).reshape(
            self.muser.antennas * (self.muser.antennas - 1) / 2, 1, 1, 16, 1, 3)
        pdata1 = [itemgetter(0)(i) for i in self.muser.uvws_sum]
        pdata2 = [itemgetter(1)(i) for i in self.muser.uvws_sum]
        pdata3 = [itemgetter(2)(i) for i in self.muser.uvws_sum]
        pdata4 = self.muser.baseline
        pdata5 = self.muser.obs_date_sum  # - self.muser.obs_date_sum0
        pdata6 = self.muser.obs_time_sum

        groupData = pf.GroupData(imdata, bitpix=-32, parnames=['UU', 'VV', 'WW', 'BASELINE', 'DATE', 'DATE'],
                                 pardata=[pdata1, pdata2, pdata3, pdata4, pdata5, pdata6])
        hdu = pf.GroupsHDU(groupData)

        i = 0
        VISDATA = np.ndarray(shape=(self.muser.antennas * (self.muser.antennas - 1) / 2, 1, 1, 16, 1, 3),
                             dtype=float)

        for antenna1 in range(0, self.muser.antennas - 1):
            for antenna2 in range(antenna1 + 1, self.muser.antennas):
                for channel in range(0, 16):
                    VISDATA[i][0][0][channel][0] = [self.muser.baseline_data[i][channel].real,
                                                    self.muser.baseline_data[i][channel].imag, 1.0]
                i += 1

        # print "self.muser.DROutputAntennas", self.muser.DROutputAntennas * (self.muser.DROutputAntennas - 1) // 2
        for i in range(0, self.muser.antennas * (self.muser.antennas - 1) // 2):
            hdu.data.data[i] = VISDATA[i]
            i += 1

        # Generate headers from config file
        # primary = self.parse_config('PRIMARY')
        # primaryTable = self.parse_config('PRIMARYTable')
        primary = {'BITPIX': 8, 'EXTEND': True, 'NAXIS': 0, 'PCOUNT': 0, 'GROUPS': True, 'GCOUNT': 0}
        primaryTable = {'PZERO4': 0.0, 'PZERO5': 0.0, 'PZERO6': 0.0, 'PZERO1': 0.0, 'PZERO2': 0.0, 'PZERO3': 0.0,
                        'CDELT2': 0, 'OBJECT': 'MUSER-1', 'CDELT4': 25000000.0, 'CDELT5': 0.0, 'CDELT6': 0.0,
                        'DATE-OBS': '2014-05-12T13:14:59.988771120', 'STRT-OBS': '2014-05-12T13:14:59.988771120',
                        'END-OBS': '2014-05-12T13:14:59.988771120', 'OBSDEC': 18.113205304565984,
                        'OBSRA': 3.266187801373725, 'PSCAL5': 1.0, 'PSCAL4': 1.0, 'PSCAL6': 1.0, 'PSCAL1': 1.0,
                        'PSCAL3': 1.0, 'PSCAL2': 1.0, 'INSTRUME': 'MUSER', 'CRVAL6': 0.0, 'CDELT3': -1, 'CRPIX6': 1.0,
                        'EPOCH': 2000.0, 'CRPIX4': 1.0, 'CRPIX5': 1.0, 'CRVAL4': 400000000, 'CRVAL5': 0.0,
                        'CRVAL2': 0.0, 'CRVAL3': -2, 'CRPIX2': 1.0, 'CRPIX3': 1.0, 'TELESCOP': 'MUSER', 'CTYPE6': 'DEC',
                        'CTYPE5': 'RA', 'CTYPE4': 'FREQ', 'CTYPE3': 'STOKES', 'CTYPE2': 'COMPLEX', 'BSCALE': 1.0}
        primaryTable['OBSRA'] = self.muser.ra_sum  #PZERO5:2456789.5
        primaryTable['OBSDEC'] = self.muser.dec_sum

        primaryTable['DATE-OBS'] = self.muser.current_frame_time.get_fits_date_time()
        primaryTable['STRT-OBS'] = self.muser.start_date_time_fits.get_fits_date_time()
        primaryTable['END-OBS'] = self.muser.end_date_time_fits.get_fits_date_time()
        # primaryTable['PZERO5'] = self.muser.obs_date_sum
        primaryTable['OBJECT'] = "MUSER-" + str(self.muser.sub_array)
        primaryTable['CRVAL3'] = self.muser.polarization - 2
        if self.muser.sub_array == 2:
            primaryTable['CRVAL4'] = 2000000000

        for key in primary: hdu.header.set(key, primary[key])
        for key in primaryTable: hdu.header.set(key, primaryTable[key])
        hdu.verify()  # Will raise a warning if there's an issue
        return hdu




    def make_antenna(self, num_rows):
        # Creates a vanilla ANTENNA table HDU

        # cards = self.parse_config('ANTENNA')
        # print cards
        cards = {'ARRAYZ': 4263977.594931615, 'ARRAYY': 4280075.238577609, 'ARRAYX': -2018660.1921682167,
                 'GSTIA0': 121.267028238182, 'DEGPDY': 360.985, 'NUMORB': 0, 'IATUTC': 33.0, 'POLARY': 0.30368,
                 'EXTNAME': 'AIPS AN', 'POLARX': 0.12633, 'RDATE': '2012-11-21 ', 'UT1UTC': 0.318725, 'DATUTC': 33.0,
                 'TIMSYS': 'UTC', 'ARRNAM': 'MUSER', 'FREQ': 400000000.0, 'NOPCAL': 3, 'POLTYPE': 'APPROX'}
        # common = parse_config('COMMON', config)
        # cards['RDATE'] = self.muser.obs_date

        c = []

        c.append(pf.Column(name='ANNAME', format='8A', \
                           array=np.zeros(num_rows, dtype='a8')))

        c.append(pf.Column(name='STABXYZ', format='3D', \
                           unit='METERS', array=np.zeros(num_rows, dtype='3float64')))

        c.append(pf.Column(name='NOSTA', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        c.append(pf.Column(name='MNTSTA', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        c.append(pf.Column(name='STAXOF', format='E', \
                           unit='METERS', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='POLTYA', format='1A', \
                           array=np.zeros(num_rows, dtype='a1')))

        c.append(pf.Column(name='POLTYB', format='1A', \
                           array=np.zeros(num_rows, dtype='a1')))

        c.append(pf.Column(name='POLAA', format='1E', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='POLAB', format='1E', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='POLCALA', format='3E', \
                           array=np.zeros(num_rows, dtype='3float32')))

        c.append(pf.Column(name='POLCALB', format='3E', \
                           array=np.zeros(num_rows, dtype='3float32')))

        coldefs = pf.ColDefs(c)
        tblhdu = pf.BinTableHDU.from_columns(coldefs)

        for key in cards: tblhdu.header.set(key, cards[key])

        return tblhdu

    def make_frequency(self, num_rows):
        # Creates a vanilla FREQUENCY table HDU
        # Generate headers from config file
        #cards = self.parse_config('FREQUENCY')


        cards={'NO_IF': 1, 'EXTVER': 1, 'EXTNAME': 'AIPS FQ'}

        c = []

        c.append(pf.Column(name='FRQSEL', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        c.append(pf.Column(name='IF FREQ', format='1D', \
                           unit='HZ', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='CH WIDTH', format='1E', \
                           unit='HZ', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='TOTAL BANDWIDTH', format='1E', \
                           unit='HZ', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='SIDEBAND', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        coldefs = pf.ColDefs(c)
        tblhdu = pf.BinTableHDU.from_columns(coldefs)
        for key in cards: tblhdu.header.set(key, cards[key])

        return tblhdu

    def make_source(self, num_rows):
        # Creates a vanilla SOURCE table HDU
        #cards = self.parse_config('SOURCE')
        cards = {'NO_IF': 1, 'EXTVER': 1, 'FREQID': 1, 'EXTNAME': 'AIPS SU'}
        so_format = '1E'
        so_dtype = 'float32'

        c = []

        c.append(pf.Column(name='ID. NO.', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        c.append(pf.Column(name='SOURCE', format='16A', \
                           array=np.zeros(num_rows, dtype='16a')))

        c.append(pf.Column(name='QUAL', format='1J', \
                           array=np.zeros(num_rows, dtype='int32')))

        c.append(pf.Column(name='CALCODE', format='4A', \
                           array=np.zeros(num_rows, dtype='4a')))

        c.append(pf.Column(name='IFLUX', format=so_format, \
                           unit='JY', array=np.zeros(num_rows, dtype=so_dtype)))

        c.append(pf.Column(name='QFLUX', format=so_format, \
                           unit='JY', array=np.zeros(num_rows, dtype=so_dtype)))

        c.append(pf.Column(name='UFLUX', format=so_format, \
                           unit='JY', array=np.zeros(num_rows, dtype=so_dtype)))

        c.append(pf.Column(name='VFLUX', format=so_format, \
                           unit='JY', array=np.zeros(num_rows, dtype=so_dtype)))

        c.append(pf.Column(name='FREQOFF', format='1D',
                           unit='HZ', array=np.zeros(num_rows, dtype=so_dtype)))

        c.append(pf.Column(name='BANDWIDTH', format='1D', \
                           unit='HZ', array=np.zeros(num_rows, dtype='float32')))

        c.append(pf.Column(name='RAEPO', format='1D', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='DECEPO', format='1D', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='EPOCH', format='1D', \
                           unit='YEARS', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='RAAPP', format='1D', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='DECAPP', format='1D', \
                           unit='DEGREES', array=np.zeros(num_rows, dtype='float64')))

        sv_format = '1D'
        sv_dtype = 'float64'
        c.append(pf.Column(name='LSRVEL', format=sv_format, \
                           unit='METERS/SEC', array=np.zeros(num_rows, dtype=sv_dtype)))

        rf_format = '1D'
        rf_dtype = 'float64'
        c.append(pf.Column(name='RESTFREQ', format=rf_format, \
                           unit='HZ', array=np.zeros(num_rows, dtype=rf_dtype)))

        c.append(pf.Column(name='PMRA', format='1D', \
                           unit='DEGREES/DAY', array=np.zeros(num_rows, dtype='float64')))

        c.append(pf.Column(name='PMDEC', format='1D', \
                           unit='DEGREES/DAY', array=np.zeros(num_rows, dtype='float64')))

        coldefs = pf.ColDefs(c)
        tblhdu = pf.BinTableHDU.from_columns(coldefs)
        for key in cards: tblhdu.header.set(key, cards[key])

        return tblhdu

    def locxyz2itrf(self, lat, longitude, locx=0.0, locy=0.0, locz=0.0):
        """
        Returns the nominal ITRF (X, Y, Z) coordinates (m) for a point at "local"
        (x, y, z) (m) measured at geodetic latitude lat and longitude longitude
        (degrees).  The ITRF frame used is not the official ITRF, just a right
        handed Cartesian system with X going through 0 latitude and 0 longitude,
        and Z going through the north pole.  The "local" (x, y, z) are measured
        relative to the closest point to (lat, longitude) on the WGS84 reference
        ellipsoid, with z normal to the ellipsoid and y pointing north.
        """
        # from Rob Reid;  need to generalize to use any datum...
        phi, lmbda = map(math.radians, (lat, longitude))
        sphi = math.sin(phi)
        a = 6378137.0  # WGS84 equatorial semimajor axis
        b = 6356752.3142  # WGS84 polar semimajor axis
        ae = math.acos(b / a)
        N = a / math.sqrt(1.0 - (math.sin(ae) * sphi) ** 2)

        # Now you see the connection between the Old Ones and Antarctica...
        # Nploczcphimlocysphi = (N + locz) * pl.cos(phi) - locy * sphi
        Nploczcphimlocysphi = (N + locz) * math.cos(phi) - locy * sphi

        clmb = math.cos(lmbda)
        slmb = math.sin(lmbda)

        x = Nploczcphimlocysphi * clmb - locx * slmb
        y = Nploczcphimlocysphi * slmb + locx * clmb
        z = (N * (b / a) ** 2 + locz) * sphi + locy * math.cos(phi)

        return x, y, z

    def makeSource(self, name, ra="", dec="", flux=0, epoch=2000):
        """ Create a pyEphem FixedBody

        Parameters
        ----------
        name: string
          Name of source, e.g. CasA
        ra: hh:mm:ss
          right ascension, e.g. 23:23:26
        dec: dd:mm:ss
          declination e.g. 58:48:22.21
        flux: float
          flux brightness in Jy (not actually used here)
        epoch: J2000
          Defaults to J2000, i.e. 2000"""

        # line = "%s,f,%s,%s,%s,%d" % (name, ra, dec, flux, epoch)
        if name.lower() == 'sun':
            body = Body(name=name)
        else:
            body = Body(name=name, ra=ra, dec=dec)
        return body

    def computeUVW(self, xyz, H, d):
        """ Converts X-Y-Z coordinates into U-V-W

        Uses the transform from Thompson Moran Swenson (4.1, pg86)

        Parameters
        ----------
        xyz: should be a numpy array [x,y,z]
        H: float (degrees)
          is the hour angle of the phase reference position
        d: float (degrees)
          is the declination
        """
        H, d = map(math.radians, (H, d))
        sin = np.sin
        cos = np.cos

        xyz = np.matrix(xyz)  # Cast into a matrix

        trans = np.matrix([
            [sin(H), cos(H), 0],
            [-sin(d) * cos(H), sin(d) * sin(H), cos(d)],
            [cos(d) * cos(H), -cos(d) * sin(H), sin(d)]
        ])

        uvw = trans * xyz.T

        uvw = np.array(uvw)

        return uvw[:, 0]

    def ant_array_name(self):
        """ The antenna array name for muser
        This doesn't really need to be a function.
        """
        if self.muser.sub_array == 1:  # muser-I
            ns = np.array(
                ['IA0', 'IA1', 'IA2', 'IA3', 'IA4', 'IA5', 'IA6', 'IA7', 'IA8', 'IA9', 'IA10', 'IA11', 'IA12', 'IA13',
                 'IB1', 'IB2', 'IB3', 'IB4', 'IB5', 'IB6', 'IB7', 'IB8', 'IB9', 'IB10', 'IB11', 'IB12', 'IB13',
                 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', 'IC7', 'IC8', 'IC9', 'IC10', 'IC11', 'IC12', 'IC13'],
                dtype='string')

        elif self.muser.sub_array == 2:  # muser-II
            ns = np.array(
                ['HA1', 'HA2', 'HA3', 'HA4', 'HA5', 'HA6', 'HA7', 'HA8', 'HA9', 'HA10', 'HA11', 'HA12', 'HA13', 'HA14',
                 'HA15', 'HA16', 'HA17', 'HA18', 'HA19', 'HA20', 'HB1', 'HB2', 'HB3', 'HB4', 'HB5', 'HB6', 'HB7', 'HB8',
                 'HB9', 'HB10',
                 'HB11', 'HB12', 'HB13', 'HB14', 'HB15', 'HB16', 'HB17', 'HB18', 'HB19', 'HB20', 'HC1', 'HC2', 'HC3',
                 'HC4', 'HC5', 'HC6',
                 'HC7', 'HC8', 'HC9', 'HC10', 'HC11', 'HC12', 'HC13', 'HC14', 'HC15', 'HC16', 'HC17', 'HC18', 'HC19',
                 'HC20'],
                dtype='string')

        return ns

    def ant_array(self, rdate = None):
        """ The antenna array for muser.
        This doesn't really need to be a function.
        """

        # We are at Neimeng, China
        # http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # ECEF from latitude,longitude, Height (ellipsoidal)

        # X-Y-Z in nanoseconds (re-ordered by hand)

        xyzold = []
        if rdate is None:
            rdate = '2999-12-31 00:00:00'
        xyzold = muserantenna.get_muser_antennas_position(self.muser.sub_array, rdate)
        xyzs = np.array(xyzold)
        xyz = xyzs.reshape(self.muser.antennas, 3)
        xyz_m = np.ndarray(shape=(self.muser.antennas, 3), dtype='float64')

        # Returns the nominal ITRF (X, Y, Z) coordinates (m) for a point at "local"(x, y, z) (m)
        xx, yy, zz = self.locxyz2itrf(self.latitude, self.longitude, 0., 0., self.altitude)

        for bl in range(0, self.muser.antennas):
            xyz_m[bl][:] = self.locxyz2itrf(self.latitude, self.longitude, xyz[bl][0], xyz[bl][1],
                                            self.altitude + xyz[bl][2])

        for bl in range(0, self.muser.antennas):
            xyz_m[bl][0] -= xx
            xyz_m[bl][1] -= yy
            xyz_m[bl][2] -= zz

        # print xyz_m
        # X-Y-Z in metres
        # xyz_m = xyz_ns * 10 ** -9 * 299792458

        return xyz_m

    # #######################
    # CONFIG FUNCTIONS   #
    # #######################



    def config_antenna(self, tbl):
        """ Configures the antenna table.

        Parameters
        ----------
        tbl: pyfits.hdu
          table to be configured
        """

        xyz_m = self.ant_array() #self.array_geometry
        antname = self.ant_array_name()
        antenna = tbl.data

        for i in range(0, tbl.data.size):
            antenna[i]['ANNAME'] = antname[i]
            antenna[i]['STABXYZ'] = xyz_m[i]
            antenna[i]['NOSTA'] = i+i
            antenna[i]['MNTSTA'] = 2
            antenna[i]['STAXOF'] = 0
            antenna[i]['POLTYA'] = 'R'
            antenna[i]['POLTYB'] = 'L'
            antenna[i]['POLAA'] = 0
            antenna[i]['POLAB'] = 0
            antenna[i]['POLCALA'] = [0, 0, 0]
            antenna[i]['POLCALB'] = [0, 0, 0]

        tbl.data = antenna

        return tbl

    def config_source(self, tbl, objsource):
        """  Configures the source table.
        Parameters
        ----------
        tbl: pyfits.hdu
          table to be configured
        source: ephem.fixedBody
          source to be phased to (use makeSource())
        """
        source_raapp = objsource.appra
        source_decapp = objsource.appdec
        source_ratopo = objsource.topora
        source_dectopo = objsource.topodec
        source_name = objsource.name

        # print('   Source is: %s' % objsource.name)

        source = tbl.data[0]

        source['ID. NO.'] = 1
        source['SOURCE'] = source_name
        source['QUAL'] = 0
        source['CALCODE'] = 'OBJ'
        source['IFLUX'] = 0
        source['QFLUX'] = 0
        source['UFLUX'] = 0
        source['VFLUX'] = 0
        source['FREQOFF'] = 0
        source['BANDWIDTH'] = 2.5 * 10 ** 7
        source['RAEPO'] = source_ratopo * 15 - 360
        source['DECEPO'] = source_dectopo
        source['EPOCH'] = 2000
        source['RAAPP'] = source_raapp * 15 - 360
        source['DECAPP'] = source_decapp
        source['LSRVEL'] = 0
        source['RESTFREQ'] = self.muser.frequency
        source['PMRA'] = 0
        source['PMDEC'] = 0

        tbl.data[0] = source

        return tbl

    def config_frequency(self, tbl):
        """
        Configures the frequency table.

        Parameters
        ----------
        tbl: pyfits.hdu
          table to be configured
        """
        frequency = tbl.data[0]

        frequency['FRQSEL'] = 1
        if self.muser.sub_array ==1:
            frequency['IF FREQ'] = self.muser.frequency - 400*10**6
        else:
            frequency['IF FREQ'] = self.muser.frequency - 2*10**9
        frequency['CH WIDTH'] = 2.5 * 10 ** 7
        frequency['TOTAL BANDWIDTH'] = 400 * 10 ** 6
        frequency['SIDEBAND'] = 1

        tbl.data[0] = frequency

        return tbl


    def make_primary_big(self): ### Time means: 1 minute, two minutes or more!!!!
        """  Creates the primary header data unit (HDU).

        This function generates header keywords from the file headers/primary.tpl
\        """

        # Make a new FITS HDU muser.DROutputAntennas*(muser.DROutputAntennas-1)/2=60*60/2

        #group = 60*1000/25*8*(self.muser.antennas * (self.muser.antennas - 1) // 2)
        group = self.visibility_data.shape[0]

        pdata1 = np.asarray([a for a in self.datauu], dtype = np.float32)
        pdata2 = np.asarray([a for a in self.datavv], dtype = np.float32)
        pdata3 = np.asarray([a for a in self.dataww], dtype = np.float32)
        pdata4 = np.asarray([a for a in self.databaseline], dtype = np.float32)
        pdata5 = np.asarray([a for a in self.datadate1], dtype = np.float32)
        pdata6 = np.asarray([a for a in self.datadate2], dtype = np.float32)
        pdata7 = np.asarray([a for a in self.datasource], dtype = np.int32)
        pdata8 = np.asarray([a for a in self.datafreqsel], dtype = np.int32)
        pdata9 = 1


        groupData = pf.GroupData(self.visibility_data, bitpix=-32, parnames=['UU', 'VV', 'WW', 'BASELINE', 'DATE', 'DATE','SOURCE', 'FREQSEL','INTTIM'],
                                 pardata=[pdata1, pdata2, pdata3, pdata4, pdata5, pdata6, pdata7,pdata8,pdata9])
        hdu = pf.GroupsHDU(groupData)

        # Generate headers from config file
        primary = {'BITPIX': 8, 'EXTEND': True, 'NAXIS': 0, 'PCOUNT': 0, 'GROUPS': True, 'GCOUNT': 0}
        primaryTable = {'PZERO4': 0.0, 'PZERO5': 0.0, 'PZERO6': 0.0, 'PZERO1': 0.0, 'PZERO2': 0.0, 'PZERO3': 0.0,
                        'CDELT2': 0, 'OBJECT': 'MUSER-1', 'CDELT4': 25000000.0, 'CDELT5': 0.0, 'CDELT6': 0.0,
                        'DATE-OBS': '2014-05-12T13:14:59.988771120', 'OBSDEC': 18.113205304565984,
                        'OBSRA': 3.266187801373725, 'PSCAL5': 1.0, 'PSCAL4': 1.0, 'PSCAL6': 1.0, 'PSCAL1': 1.0,
                        'PSCAL3': 1.0, 'PSCAL2': 1.0, 'INSTRUME': 'MUSER', 'CRVAL6': 0.0, 'CDELT3': -1, 'CRPIX6': 1.0,
                        'EPOCH': 2000.0, 'CRPIX4': 1.0, 'CRPIX5': 1.0, 'CRVAL4': 400000000, 'CRVAL5': 0.0,
                        'CRVAL2': 0.0, 'CRVAL3': -2, 'CRPIX2': 1.0, 'CRPIX3': 1.0, 'TELESCOP': 'MUSER', 'CTYPE6': 'DEC',
                        'CTYPE5': 'RA', 'CTYPE4': 'FREQ', 'CTYPE3': 'STOKES', 'CTYPE2': 'COMPLEX', 'BSCALE': 1.0}
        primaryTable['OBSRA'] = self.muser.ra_sum
        primaryTable['OBSDEC'] = self.muser.dec_sum
        primaryTable['DATE-OBS'] = self.muser.current_frame_time.get_fits_date_time()  # self.muser.obs_date + "T00:00:00.0"
        primaryTable['OBJECT'] = "MUSER-" + str(self.muser.sub_array)
        primaryTable['CRVAL3'] = self.muser.polarization - 2
        if self.muser.sub_array == 2:
            primaryTable['CRVAL4'] = 2000000000

        for key in primary: hdu.header.set(key, primary[key])
        for key in primaryTable: hdu.header.set(key, primaryTable[key])
        hdu.verify()  # Will raise a warning if there's an issue
        return hdu

    def set_visibility_data(self, group):
        self.visibility_data = np.zeros(shape=(group*self.muser.antennas * (self.muser.antennas - 1) / 2, 1, 1, 16, 2, 3), dtype='float32')
        return

    def config_primary_big(self, source_id, error = 0, if_append = True):
        # Append UVW and related data

        # if in loop mode and the first frame is not left polarization, just skip
        data_count = self.muser.antennas * (self.muser.antennas - 1) / 2
        data_start = (source_id - 1)*data_count

        if self.muser.is_loop_mode == True:

            if (self.muser.polarization ==0 and if_append == True) or (error != 0 and if_append == True):
                self.datauu.extend((itemgetter(0)(i) for i in self.muser.uvws_sum))
                self.datavv.extend((itemgetter(1)(i) for i in self.muser.uvws_sum))
                self.dataww.extend((itemgetter(2)(i) for i in self.muser.uvws_sum))

                self.databaseline.extend([i for i in self.muser.baseline])
                self.datadate1.extend([self.muser.obs_date_sum for i in range (data_count)])
                self.datadate2.extend([self.muser.obs_time_sum for i in range (data_count)])
                self.datasource.extend([source_id for i in range (data_count)])
                self.datafreqsel.extend([self.muser.freqid for i in range (data_count)])

            i=0
            if error == 0:  # No Error
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[data_start+i][0][0][channel][self.muser.polarization] = [self.muser.baseline_data[i][channel].real, self.muser.baseline_data[i][channel].imag, 1.0]
                        i += 1
            elif error == 1: # Error with condition 1: iLoop % 2 == 0 and self.polarization == 1
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[data_start+i][0][0][channel][self.muser.polarization - 1] = [0., 0., 0.0]
                            self.visibility_data[data_start+i][0][0][channel][self.muser.polarization] = [self.muser.baseline_data[i][channel].real, self.muser.baseline_data[i][channel].imag, 1.0]
                        i += 1
            elif error == 2:  # Error with condition 2: iLoop % 2 == 1 and self.polarization == 0
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[data_start+i-data_count][0][0][channel][self.muser.polarization + 1] = [0., 0., 0.0]
                            if if_append == True:
                                self.visibility_data[data_start+i][0][0][channel][self.muser.polarization] = [self.muser.baseline_data[i][channel].real, self.muser.baseline_data[i][channel].imag, 1.0]
                        i += 1

            elif error == 3:  # Error with condition 3: skipped more than two frames, self.polarization == 1
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[data_start+i-data_count][0][0][channel][self.muser.polarization] = [0., 0., 0.0]
                            if if_append == True:
                                self.visibility_data[data_start+i][0][0][channel][self.muser.polarization - 1] = [0., 0., 0.0]
                                self.visibility_data[data_start+i][0][0][channel][self.muser.polarization] = [self.muser.baseline_data[i][channel].real, self.muser.baseline_data[i][channel].imag, 1.0]
                        i += 1

        else:
            self.datauu.extend((itemgetter(0)(i) for i in self.muser.uvws_sum))
            self.datavv.extend((itemgetter(1)(i) for i in self.muser.uvws_sum))
            self.dataww.extend((itemgetter(2)(i) for i in self.muser.uvws_sum))

            self.databaseline.extend([i for i in self.muser.baseline])
            self.datadate1.extend([self.muser.obs_date_sum for i in range (data_count)])
            self.datadate2.extend([self.muser.obs_time_sum for i in range (data_count)])
            self.datasource.extend([source_id for i in range (data_count)])
            self.datafreqsel.extend([self.muser.freqid for i in range (data_count)])

            i=0
            for antenna1 in range(0, self.muser.antennas - 1):
                for antenna2 in range(antenna1 + 1, self.muser.antennas):
                    for channel in range(0, 16):
                        self.visibility_data[data_start+i][0][0][channel][self.muser.polarization] = [self.muser.baseline_data[i][channel].real, self.muser.baseline_data[i][channel].imag, 1.0]
                    i += 1

        return


    def config_merge_primary_big(self, task_number, repeatnumber):

        data_count = self.muser.antennas * (self.muser.antennas - 1) / 2
        group = self.visibility_data.shape[0]

        if self.muser.is_loop_mode == True:
            if self.muser.polarization ==0:
                self.datauu = (itemgetter(0)(i) for i in self.muser.uvwdata)
                self.datavv = (itemgetter(1)(i) for i in self.muser.uvwdata)
                self.dataww = (itemgetter(2)(i) for i in self.muser.uvwdata)

                self.datadate1 = self.muser.date1
                self.datadate2 = self.muser.date2

                self.databaseline = self.muser.baseline_merge
                self.datasource = (i for i in range (group))

                freqsel = [1, 1, 2, 2, 3, 3, 4, 4]
                for i in range(0, repeatnumber//8): # for i in range(0, repeatnumber//33)
                    self.datafreqsel.extend([freqsel])

            i=0
            for repeat  in range(0, task_number*repeatnumber):
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[:][0][0][channel][repeatnumber*task_number%2] = [self.muser.vis[i][channel].real, self.muser.vis[i][channel].imag, 1.0]
                        i += 1

        else:
            #TODO
            self.datauu = (itemgetter(0)(i) for i in self.muser.uvwdata)
            self.datavv = (itemgetter(1)(i) for i in self.muser.uvwdata)
            self.dataww = (itemgetter(2)(i) for i in self.muser.uvwdata)

            self.datadate1 = self.muser.date1
            self.datadate2 = self.muser.date2

            self.databaseline = self.muser.baseline_merge
            self.datasource = (i for i in range (group))
            self.datafreqsel = (1 for i in range (group))


            i=0
            for repeat in range(0, task_number*repeatnumber):
                for antenna1 in range(0, self.muser.antennas - 1):
                    for antenna2 in range(antenna1 + 1, self.muser.antennas):
                        for channel in range(0, 16):
                            self.visibility_data[:][0][0][channel][task_number*repeatnumber%2] = [self.muser.vis[i][channel].real, self.muser.vis[i][channel].imag, 1.0]
                        i += 1

        return


    def config_merger_source_big(self, repeatnumber):
        """  Configures the source table.
        Parameters
        ----------
        tbl: pyfits.hdu
          table to be configured
        source: ephem.fixedBody
          source to be phased to (use makeSource())
        """
        for repeat in range(0, repeatnumber):

            # source_raapp = objsource[repeatnumber].appra # An array stores all the RA in one minute!!!!!
            # source_decapp = objsource[repeatnumber].appdec
            # source_ratopo = objsource[repeatnumber].topora
            # source_dectopo = objsource[repeatnumber].topodec
            # source_name = objsource[repeatnumber].name

            # print('   Source is: %s' % objsource.name)

            source = {}

            source['IDNO'] = repeatnumber
            source['SOURCE'] = "SUN"
            source['QUAL'] = 0
            source['CALCODE'] = 'OBJ'
            source['IFLUX'] = 0
            source['QFLUX'] = 0
            source['UFLUX'] = 0
            source['VFLUX'] = 0
            source['FREQOFF'] = 0
            source['BANDWIDTH'] = 2.5 * 10 ** 7
            source['RAEPO'] = self.muser.topora[repeatnumber] * 15 - 360
            source['DECEPO'] = self.muser.topodec[repeatnumber]
            source['EPOCH'] = 2000
            source['RAAPP'] = self.muser.appra[repeatnumber] * 15 - 360
            source['DECAPP'] = self.muser.appdec[repeatnumber]
            source['LSRVEL'] = 0
            source['RESTFREQ'] = self.muser.frequency[repeatnumber]
            source['PMRA'] = 0
            source['PMDEC'] = 0

            self.source_dict.append(source)
        return


    def config_source_big(self, objsource, sourceid):
        """  Configures the source table.
        Parameters
        ----------
        tbl: pyfits.hdu
          table to be configured
        source: ephem.fixedBody
          source to be phased to (use makeSource())
        """
        source_raapp = objsource.appra # An array stores all the RA in one minute!!!!!
        source_decapp = objsource.appdec
        source_ratopo = objsource.topora
        source_dectopo = objsource.topodec
        source_name = objsource.name

        # print('   Source is: %s' % objsource.name)

        source = {}

        source['IDNO'] = sourceid
        source['SOURCE'] = source_name
        source['QUAL'] = 0
        source['CALCODE'] = 'OBJ'
        source['IFLUX'] = 0
        source['QFLUX'] = 0
        source['UFLUX'] = 0
        source['VFLUX'] = 0
        source['FREQOFF'] = 0
        source['BANDWIDTH'] = 2.5 * 10 ** 7
        source['RAEPO'] = source_ratopo * 15 - 360
        source['DECEPO'] = source_dectopo
        source['EPOCH'] = 2000
        source['RAAPP'] = source_raapp * 15 - 360
        source['DECAPP'] = source_decapp
        source['LSRVEL'] = 0
        source['RESTFREQ'] = self.muser.frequency
        source['PMRA'] = 0
        source['PMDEC'] = 0

        self.source_dict.append(source)
        return

    def make_source_big(self):
        # Creates a vanilla SOURCE table HDU
        #cards = self.parse_config('SOURCE')
        cards = {'NO_IF': 1, 'EXTVER': 1, 'FREQID': 1, 'EXTNAME': 'AIPS SU'}
        so_format = '1E'
        so_dtype = 'float32'

        c = []

        c.append(pf.Column(name='ID. NO.', format='1J', \
                           array= [a['IDNO'] for a in self.source_dict]))

        c.append(pf.Column(name='SOURCE', format='16A', \
                           array= [a['SOURCE'] for a in self.source_dict]))

        c.append(pf.Column(name='QUAL', format='1J', \
                           array= [a['QUAL'] for a in self.source_dict]))

        c.append(pf.Column(name='CALCODE', format='4A', \
                           array= [a['CALCODE'] for a in self.source_dict]))

        c.append(pf.Column(name='IFLUX', format=so_format, \
                           unit='JY', array= [a['IFLUX'] for a in self.source_dict]))

        c.append(pf.Column(name='QFLUX', format=so_format, \
                           unit='JY', array= [a['QFLUX'] for a in self.source_dict]))

        c.append(pf.Column(name='UFLUX', format=so_format, \
                           unit='JY', array= [a['UFLUX'] for a in self.source_dict]))

        c.append(pf.Column(name='VFLUX', format=so_format, \
                           unit='JY', array= [a['VFLUX'] for a in self.source_dict]))

        c.append(pf.Column(name='FREQOFF', format='1D',
                           unit='HZ', array= [a['FREQOFF'] for a in self.source_dict]))

        c.append(pf.Column(name='BANDWIDTH', format='1D', \
                           unit='HZ', array= [a['BANDWIDTH'] for a in self.source_dict]))

        c.append(pf.Column(name='RAEPO', format='1D', \
                           unit='DEGREES', array= [a['RAEPO'] for a in self.source_dict]))

        c.append(pf.Column(name='DECEPO', format='1D', \
                           unit='DEGREES', array= [a['DECEPO'] for a in self.source_dict]))

        c.append(pf.Column(name='EPOCH', format='1D', \
                           unit='YEARS', array= [a['EPOCH'] for a in self.source_dict]))

        c.append(pf.Column(name='RAAPP', format='1D', \
                           unit='DEGREES', array= [a['RAAPP'] for a in self.source_dict]))

        c.append(pf.Column(name='DECAPP', format='1D', \
                           unit='DEGREES', array= [a['DECAPP'] for a in self.source_dict]))

        c.append(pf.Column(name='LSRVEL', format='1D', \
                           unit='METERS/SEC', array= [a['LSRVEL'] for a in self.source_dict]))

        c.append(pf.Column(name='RESTFREQ', format='1D', \
                           unit='HZ', array= [a['RESTFREQ'] for a in self.source_dict]))

        c.append(pf.Column(name='PMRA', format='1D', \
                           unit='DEGREES/DAY', array= [a['PMRA'] for a in self.source_dict]))

        c.append(pf.Column(name='PMDEC', format='1D', \
                           unit='DEGREES/DAY', array= [a['PMDEC'] for a in self.source_dict]))

        coldefs = pf.ColDefs(c)
        tblhdu = pf.BinTableHDU.from_columns(coldefs)
        for key in cards: tblhdu.header.set(key, cards[key])

        return tblhdu

    def make_frequency_big(self):
        # Creates a vanilla FREQUENCY table HDU
        # Generate headers from config file
        #cards = self.parse_config('FREQUENCY')


        cards={'NO_IF': 1, 'EXTVER': 1, 'EXTNAME': 'AIPS FQ'}

        c = []

        c.append(pf.Column(name='FRQSEL', format='1J', \
                           array=[a['FRQSEL'] for a in self.frequency_dict]))

        c.append(pf.Column(name='IF FREQ', format='1D', \
                           unit='HZ', array=([a['IF FREQ'] for a in self.frequency_dict])))

        c.append(pf.Column(name='CH WIDTH', format='1E', \
                           unit='HZ', array=([a['CHWIDTH'] for a in self.frequency_dict])))

        c.append(pf.Column(name='TOTAL BANDWIDTH', format='1E', \
                           unit='HZ',  array=([a['TOTALBANDWIDTH'] for a in self.frequency_dict])))

        c.append(pf.Column(name='SIDEBAND', format='1J', \
                           array=([a['SIDEBAND'] for a in self.frequency_dict])))

        coldefs = pf.ColDefs(c)
        tblhdu = pf.BinTableHDU.from_columns(coldefs)
        for key in cards: tblhdu.header.set(key, cards[key])

        return tblhdu

    def config_frequency_big(self):
        if self.muser.sub_array == 1:
            for i in range(0, 4):
                frequency = {}
                frequency['FRQSEL'] = i+1
                frequency['IF FREQ'] = (i+1)*400E6 - 400E6
                frequency['CHWIDTH'] = 25E6
                frequency['TOTALBANDWIDTH'] = 400E6
                frequency['SIDEBAND'] = 1
                self.frequency_dict.append(frequency)
        else:
            for i in range(0, 33):
                frequency = {}
                frequency['FRQSEL'] = i+1
                frequency['IF FREQ'] = i*400E6
                frequency['CHWIDTH'] = 25E6
                frequency['TOTALBANDWIDTH'] = 400E6
                frequency['SIDEBAND'] = 1

                if i==32:
                    frequency['IF FREQ'] = i * 400E6 -200E6

                self.frequency_dict.append(frequency)
        return


    def config_baseline_ID(self, bl_len):
        # print('     Creating baseline IDs...')
        self.array_geometry = self.ant_array(self.muser.current_frame_time.get_short_string())
        antennas = self.array_geometry #self.ant_array()
        bl_order = np.ndarray(shape=(bl_len, 2), dtype=int)
        blen = 0

        for border1 in range(0, self.muser.antennas):
            for border2 in range(border1 + 1, self.muser.antennas):
                bl_order[blen][0] = border1 + 1
                bl_order[blen][1] = border2 + 1
                blen = blen + 1

        baselines = []

        for bl in range(0, bl_len):
            # Baseline is in stupid 256*baseline1 + baseline2 format
            ant1, ant2 = bl_order[bl][0], bl_order[bl][1]
            bl_id = 256 * ant1 + ant2

            # Generate the XYZ vectors too
            # From CASA measurement set definition
            # uvw coordinates for the baseline from ANTENNE2 to ANTENNA1,
            # i.e. the baseline is equal to the difference POSITION2 - POSITION1.
            bl_vector = antennas[ant2 - 1] - antennas[ant1 - 1]
            baselines.append((bl_id, bl_vector))

        return bl_order, baselines

    def write_single_uvfits(self, uvfitsFile, hourangle=999, declination=999):

        # The source is our phase centre for UVW coordinates

        Sun = self.makeSource(name=self.muser.obs_target)
        self.source = Sun

        obs = Observatory(lon=self.longitude, lat=self.latitude, altitude=self.altitude)


        self.source.midnightJD, midnightMJD = self.ephem.convert_date(self.muser.obs_date, '00:00:00')
        # We should compute the target's position firstly
        self.array_geometry = self.ant_array(self.muser.current_frame_time.get_short_string())
        antenna_array = Array(lat=self.latitude, long=self.longitude, elev=self.altitude, antennas=self.array_geometry)
        self.source.compute(cobs=obs, cdate=self.muser.obs_date, ctime=self.muser.obs_time)

        print "Current Frame Time：", self.muser.obs_date, self.muser.obs_time

        uvws = []
        self.muser.baseline = []
        bl_len = int(self.muser.antennas * (self.muser.antennas - 1) / 2)
        (bl_order, baselines) = self.config_baseline_ID(bl_len)
        for baseline in baselines:
            vector = baseline[1]
            self.muser.baseline.append(baseline[0])
            if hourangle==999 and declination==999:
               H, d = (self.source.gast - self.source.appra, self.source.appdec)
            else:
               H, d = hourangle, declination
            uvws.append(self.computeUVW(vector, H * 15., d))

        uvws = np.array(uvws)
        self.muser.uvws_sum = uvws.reshape(bl_len, 3) / light_speed  # units: SECONDS
        self.muser.obs_date_sum = self.source.midnightJD
        self.muser.obs_time_sum = self.source.JD - self.source.midnightJD
        self.muser.ra_sum = self.source.appra
        self.muser.dec_sum = self.source.appdec
        # print "RA, DEC", self.muser.ra_sum, self.muser.dec_sum

        # print self.muser.uvws_sum  #>> "/Users/mying/Downloads/uvw_high.txt"


        # Make a new blank FITS HDU
        hdu = self.make_primary()
        tbl_frequency = self.make_frequency(num_rows=1)
        tbl_frequency = self.config_frequency(tbl_frequency)
        tbl_antenna = self.make_antenna(num_rows=self.muser.antennas)
        tbl_antenna = self.config_antenna(tbl_antenna)
        tbl_source = self.make_source(num_rows=1)
        tbl_source = self.config_source(tbl_source, self.source)

        hdulist = pf.HDUList(
            [hdu,
             tbl_frequency,
             tbl_antenna,
             tbl_source,
             ])

        # hdulist.info()
        hdulist.verify()  # Verify all values in the instance. Output verification option.
        if (os.path.isfile(uvfitsFile)):
            os.remove(uvfitsFile)

        hdulist.writeto(uvfitsFile)
