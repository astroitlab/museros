# gpu-login
#module load courses/cs205/2012

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import scikits.cuda.fft as fft
# Initialize the CUDA device
# Elementwise stuff
from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath
from src.python.pymuser.muserframe import *
from src.python.pymuser.astroCoords import *
from src.python.pymuser import muserephem

PI = math.pi
light_speed = 299792458


######################
# CUDA kernels
######################

def delayProcess(csrh, planet):
    #print "delay Processing..."
    parameter = 0.
    if planet == 'sun':
        parameter = 12.5
    elif planet == 'satellite':
        parameter = 2.5

    delayNs = np.array([-56, 0, 48, 921, 13, 59, -3, 460, 49, 69, -675, -157, 363, -65,
                        30, 42, 51, 121, -2, 73, 35, 26, 74, 35, -3, 47, -71, 75, 343,
                        56, 32, 313, 678, 12, -30, 48, -18, 20, 10, -1666, 0, 0, 0, 0],
                       dtype="int")

    delayNsAdd = np.array([0, 0, -30, -20, 10, 9, -15, -1, -48, -33, 0, 186, 0, 0, 0,  #20140120
                           -26, 0, 0, 0, 0, -24, -1, -57, 0, 0, 3, 66, -12, 0, -20, -1,
                           0, -22, 0, 7, 10, 2, 1, 0, 0, 0, 0, 0, 0],
                          dtype="int")
    DELAY_NS_ADD1 = np.array([0, 0, 0, 0, 56, 56, 56, 56, 0, 0, 0, 0, 56, 56,
                              56, 56, 0, 0, 0, 0, 56, 56, 56, 56, 0, 0, 0,
                              0, 56, 56, 56, 56, 0, 0, 0, 0, 56, 56, 56, 56,
                              0, 0, 0, 0], dtype="int")

    delay = np.ndarray(shape=(csrh.DROutputAntennas), dtype=float)
    for i in range(0, csrh.DROutputAntennas):  # DROutputAntennas = 44
        delayNs[i] += delayNsAdd[i]
        delayNs[i] += DELAY_NS_ADD1[i]
        delay[i] = csrh.par_Delay[i] * 10 ** 9 - delayNs[i]  # Multiply par_Delay[i] by 10**9

    for antenna1 in range(0, 40 - 1):  #SubChannelsLow = 16
        for antenna2 in range(antenna1 + 1, 40):
            for channel in range(0, 16):
                Frf = (1600 + channel * 25 + parameter) / 1000.0
                Fif = (channel * 25 + parameter + 50.0) / 1000.0
                tg = delay[antenna2] - delay[antenna1]
                tg0 = int(delay[antenna2]) - int(delay[antenna1])
                phai = 2 * PI * (Frf * tg - Fif * tg0)
                #print channel, antenna1, antenna2, tg, tg0,phai,math.pi

                #print channel, antenna1, antenna2, phai

                csrh.csrhData[antenna1][antenna2][channel] = complex(
                    csrh.csrhData[antenna1][antenna2][channel].real * math.cos(phai) +
                    csrh.csrhData[antenna1][antenna2][channel].imag * math.sin(phai),
                    csrh.csrhData[antenna1][antenna2][channel].imag * math.cos(phai) -
                    csrh.csrhData[antenna1][antenna2][channel].real * math.sin(phai))

    return csrh


def makeSource(name, ra="", dec="", flux=0, epoch=2000):
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
        body = muserephem.Body(name=name)
    else:
        body = muserephem.Body(name=name, ra=ra, dec=dec)
    return body


def locxyz2itrf(lat, longitude, locx=0.0, locy=0.0, locz=0.0):
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
    #Nploczcphimlocysphi = (N + locz) * pl.cos(phi) - locy * sphi
    Nploczcphimlocysphi = (N + locz) * math.cos(phi) - locy * sphi

    clmb = math.cos(lmbda)
    slmb = math.sin(lmbda)

    x = Nploczcphimlocysphi * clmb - locx * slmb
    y = Nploczcphimlocysphi * slmb + locx * clmb
    z = (N * (b / a) ** 2 + locz) * sphi + locy * math.cos(phi)

    return x, y, z


def ant_array():
    """ The antenna array for CSRH.
    This doesn't really need to be a function.
    """

    # We are at Neimeng, China
    # http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # ECEF from Latitude,Longitude, Height (ellipsoidal)

    (latitude, longitude, elevation) = (42.211833333333333, 0, 1365)
    #(latitude, longitude, elevation) = (42.211833333333333, 115.2505, 1365)

    # X-Y-Z in nanoseconds (re-ordered by hand)

    xyz = np.array([[0, 0, 0],
                    [-0.022, 7.988, 0],
                    [-6.426, 19.456, 0],
                    [-21.025, 28.661, 0],
                    [-44.371, 31.799, 0],
                    [-79.806, -0.535, 0.1],
                    [-92.243, -67.981, 0],
                    [-49.823, -156.924, 0],
                    [75.111, -226.202, 0],
                    [283.605, -203.226, 0],
                    [517.111, 3.454, 0],
                    [624.882, 459.903, 0],
                    [356.229, 1120.404, 5.5],
                    [-286.46, 1817.53, 10],
                    [-6.884, -4.112, 0],
                    [-13.559, -15.394, 0],
                    [-14.214, -32.499, 0],
                    [-5.365, -54.364, 0],
                    [40.334, -68.776, 0],
                    [105.005, -45.994, 0],
                    [160.745, 35.21, 0],
                    [158.353, 178.005, 0],
                    [34.311, 347.22, 0],
                    [-261.386, 446.416, 5.5],
                    [-710.941, 311.107, 10],
                    [-1148.753, -251.691, 10],
                    [-1194.157, -1342.201, 10],
                    [6.966, -3.99, 0],
                    [20.1, -4.2, 0],
                    [35.379, 3.976, 0],
                    [50.088, 26.883, 0],
                    [39.461, 69.25, 0],
                    [-12.702, 113.93, 0],
                    [-110.987, 121.621, 2],
                    [-233.431, 47.856, 2],
                    [-317.836, -143.928, 2],
                    [-255.74, -449.563, 2],
                    [86.093, -771.081, 0],
                    [792.346, -868.716, 10],
                    [1759.184, -362.106, -0.1]],
                   dtype='float64')  #CSRH Ant_pos.txt

    #Returns the nominal ITRF (X, Y, Z) coordinates (m) for a point at "local"(x, y, z) (m)
    xyz_m = np.ndarray(shape=(40, 3), dtype='float64')
    xx, yy, zz = locxyz2itrf(latitude, longitude, 0., 0., 1365.)
    for bl in range(0, 40):
        xyz_m[bl][:] = locxyz2itrf(latitude, longitude, xyz[bl][0], xyz[bl][1], 1365. + xyz[bl][2])

    for bl in range(0, 40):
        xyz_m[bl][0] -= xx
        xyz_m[bl][1] -= yy
        xyz_m[bl][2] -= zz

    '''for bl in range(0, 40):
        print "ANT", bl, xyz_m[bl]'''

    #print xyz_m
    # X-Y-Z in metres
    #xyz_m = xyz_ns * 10 ** -9 * 299792458

    return xyz_m


def config_baseline_ID(bl_len):
    #print('     Creating baseline IDs...')
    antennas = ant_array()
    bl_order = np.ndarray(shape=(bl_len, 2), dtype=int)
    blen = 0

    for border1 in range(0, 39):
        for border2 in range(border1 + 1, 40):
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


def computeUVW(xyz, H, d):
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


def visibility(csrh_sun, csrh_satellite, chan):
    # Create CSRHRAWData object and set inFile

    (t_len, chan_len, bl_len, pol_len, ri_len) = (1, 16, 780, 1, 2)
    # time, channels, baselines, polarisation, then data=(real, imaginary)

    # Read one frame from the raw data file
    if chan == 4:

        file = open("Satellite.txt", 'w')
        for antenna1 in range(0, 40 - 1):
            for antenna2 in range(antenna1 + 1, 40):
                b = ('%3d%5d%5d  %20.5f %20.5f') % (
                chan, antenna1, antenna2, csrh_satellite.csrhData[antenna1][antenna2][chan].real,
                csrh_satellite.csrhData[antenna1][antenna2][chan].imag)
                #print b
                file.writelines(b + "\n")
        file.close()

    #print "TIME************\n", csrh_sun.obsdate, csrh_sun.obstime
    print "Time************\n", csrh_satellite.obsdate, csrh_satellite.obstime
    #print "VIS-------------\n", csrh_sun.csrhData


    if csrh_sun.currentFrameHeader.stripSwitch == 0xCCCCCCCC:
        delayProcess(csrh_sun, 'sun')

    if csrh_satellite.currentFrameHeader.stripSwitch == 0xCCCCCCCC:
        delayProcess(csrh_satellite, 'satellite')

    if chan == 4:

        file1 = open("Satellite_Dealy.txt", 'w')

        for antenna1 in range(0, 40 - 1):
            for antenna2 in range(antenna1 + 1, 40):
                b = ('%3d%5d%5d  %20.5f %20.5f') % (
                chan, antenna1, antenna2, csrh_satellite.csrhData[antenna1][antenna2][chan].real,
                csrh_satellite.csrhData[antenna1][antenna2][chan].imag)
                #print b
                file1.writelines(b + "\n")
        file1.close()

    if chan == 4:
        file1 = open("Sun.txt", 'w')

        for antenna1 in range(0, 40 - 1):
            for antenna2 in range(antenna1 + 1, 40):
                b = ('%3d%5d%5d  %20.5f %20.5f') % (
                chan, antenna1, antenna2, csrh_sun.csrhData[antenna1][antenna2][chan].real,
                csrh_sun.csrhData[antenna1][antenna2][chan].imag)
                #print b
                file1.writelines(b + "\n")
        file1.close()

    VisaData = np.ndarray(shape=(44, 44, 16), dtype=complex)
    phai = np.ndarray(shape=(780), dtype=complex)
    phai_sun = np.ndarray(shape=(780), dtype=complex)

    #print "Time:", csrh_sun.obsdate, csrh_sun.obstime
    #print "VIS-------------\n", csrh_sun.csrhData


    for antenna1 in range(0, 38):
        for antenna2 in range(antenna1 + 1, 39):
            A = math.sqrt(
                csrh_sun.csrhData[antenna1][antenna2][chan].imag * csrh_sun.csrhData[antenna1][antenna2][chan].imag +
                csrh_sun.csrhData[antenna1][antenna2][chan].real * csrh_sun.csrhData[antenna1][antenna2][chan].real)
            phai_sun = math.atan2(csrh_sun.csrhData[antenna1][antenna2][chan].imag,
                                  csrh_sun.csrhData[antenna1][antenna2][chan].real)
            phai = phai_sun - math.atan2(csrh_satellite.csrhData[antenna1][antenna2][chan].imag,
                                         csrh_satellite.csrhData[antenna1][antenna2][chan].real)

            csrh_sun.csrhData[antenna1][antenna2][chan] = complex(A * math.cos(phai), A * math.sin(phai))
            #csrh_sun.csrhData[antenna1][antenna2][chan] = complex(csrh_sun.csrhData[antenna1][antenna2][chan].real*math.cos(phai) + csrh_sun.csrhData[antenna1][antenna2][chan].imag*math.sin(phai),
            #csrh_sun.csrhData[antenna1][antenna2][chan].imag*math.cos(phai) - csrh_sun.csrhData[antenna1][antenna2][chan].real*math.sin(phai))

    if chan == 4:

        file2 = open("aaaa.txt", 'w')

        #for channel in range(0, 16):
        for antenna1 in range(0, 40 - 1):
            for antenna2 in range(antenna1 + 1, 40):
                b = ('%3d%5d%5d  %20.5f %20.5f') % (
                chan, antenna1, antenna2, csrh_sun.csrhData[antenna1][antenna2][chan].real,
                csrh_sun.csrhData[antenna1][antenna2][chan].imag)
                #print b
                file2.writelines(b + "\n")
        file2.close()
    obs = muserephem.Observatory()
    Sun = makeSource(name="sun")
    source = Sun
    source.midnightJD, midnightMJD = muserephem.convertDate(csrh_sun.obsdate, '00:00:00')
    #We should compute the target's position firstly
    source.compute(cobs=obs, cdate=csrh_sun.obsdate, ctime=csrh_sun.obstime)

    #print " Ephem   : Midnight_JD=%.10f;JD=%.10f;RA=%.10f;DEC=%.10f;GAST=%.10f" %(source.midnightJD, source.JD, source.appra, source.appdec, source.gast)
    #print " Ephem   : APPRA=%.10f;APPDEC=%.10f;TOPORA=%.10f;TOPODEC=%.10f;" %(source.appra, source.appdec, source.topora, source.topodec)


    #print ' LOOP MODE:', csrh.Polarization, csrh.channelGroup
    print "RA=", source.appra, "DEC=", source.appdec, "H=", source.last - source.appra, "GAST=", source.gast
    print "LAST=", source.last

    '''fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d' % (
    csrh.currentFrameTime.Year, csrh.currentFrameTime.Month, csrh.currentFrameTime.Day,
    csrh.currentFrameTime.Hour, csrh.currentFrameTime.Minute,
    csrh.currentFrameTime.Second, csrh.currentFrameTime.miliSecond,
    csrh.currentFrameTime.microSecond, csrh.currentFrameTime.nanoSecond))+'.uvfits'''''

    print " Computing UVW......"
    uvws = []
    csrh_sun.Baseline = []
    bl_len = 780
    (bl_order, baselines) = config_baseline_ID(bl_len)
    for baseline in baselines:
        vector = baseline[1]
        csrh_sun.Baseline.append(baseline[0])
        H, d = (source.last - source.appra, source.appdec)
        uvws.append(computeUVW(vector, H * 15., d))
        #print "LAST:",source.last
        #print "HOUR ANGLE=", H

    uvws = np.array(uvws)
    csrh_sun.uvws_sum = uvws.reshape(bl_len, 3) / light_speed  # units: SECONDS
    file3 = open("frame_UVW.txt", 'w+')

    for bl in range(0, bl_len):
        c = ('%5d %e %e %e') % (
        bl, csrh_sun.uvws_sum[bl][0], csrh_sun.uvws_sum[bl][1], csrh_sun.uvws_sum[bl][2])
        #print c
        file3.writelines(c + "\n")
    file3.close()

    return csrh_sun.csrhData, csrh_sun.uvws_sum


def cuda_compile(source_string, function_name):
    print "Compiling a CUDA kernel..."
    # Compile the CUDA Kernel at runtime
    source_module = nvcc.SourceModule(source_string)
    # Return a handle to the compiled CUDA kernel
    return source_module.get_function(function_name)


GRID = lambda x, y, W: ((x) + ((y) * W))

IGRIDX = lambda tid, W: tid % W
IGRIDY = lambda tid, W: int(tid) / int(W)

# -------------------
# Gridding kernels
# -------------------

code = \
    """
    #define WIDTH 6
    #define NCGF 12
    #define HWIDTH 3
    #define STEP 4

    __device__ __constant__ float cgf[32];

    // *********************
    // MAP KERNELS
    // *********************

    __global__ void gridVis_wBM_kernel(float2 *Grd, float2 *bm, int *cnt, float *d_u, float *d_v, float *d_re,
        float *d_im, int nu, float du, int gcount, int umax, int vmax){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if(iu >= u0 && iu <= u0+umax && iv <= u0+vmax){
        for (int ivis = 0; ivis < gcount; ivis++){
          float mu = d_u[ivis];
          float mv = d_v[ivis];
          int hflag = 1;
          if (mu < 0){
            hflag = -1;
            mu = -1*mu;
            mv = -1*mv;
          }
          float uu = mu/du+u0;
          float vv = mv/du+u0;
          int cnu=abs(iu-uu),cnv=abs(iv-vv);
          int ind = iv*nu+iu;
          if (cnu < HWIDTH && cnv < HWIDTH){
            float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
            Grd[ind].x +=       wgt*d_re[ivis];
            Grd[ind].y += hflag*wgt*d_im[ivis];
            cnt[ind]   += 1;
            bm [ind].x += wgt;
           }
          // deal with points&pixels close to u=0 boundary
          if (iu-u0 < HWIDTH && mu/du < HWIDTH) {
            mu = -1*mu;
            mv = -1*mv;
            uu = mu/du+u0;
            vv = mv/du+u0;
            cnu=abs(iu-uu),cnv=abs(iv-vv);
            if (cnu < HWIDTH && cnv < HWIDTH){
              float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
              Grd[ind].x +=          wgt*d_re[ivis];
              Grd[ind].y += -1*hflag*wgt*d_im[ivis];
              cnt[ind]   += 1;
              bm [ind].x += wgt;
            }
          }
        }
      }
    }

    __global__ void dblGrid_kernel(float2 *Grd, int nu, int hfac){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if (iu > 0 && iu < u0 && iv < nu && iv > 0){
        int niu = nu-iu;
        int niv = nu-iv;
        Grd[iv*nu+iu].x =      Grd[niv*nu+niu].x;
        Grd[iv*nu+iu].y = hfac*Grd[niv*nu+niu].y;
      }
    }

    __global__ void wgtGrid_kernel(float2 *Grd, int *cnt, float briggs, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if (iu >= u0 && iu < nu && iv < nu){
        if (cnt[iv*nu+iu]!= 0){
          int ind = iv*nu+iu;
          float foo = cnt[ind];
          float wgt = 1./sqrt(1 + foo*foo/(briggs*briggs));
          Grd[ind].x = Grd[ind].x*wgt;
          Grd[ind].y = Grd[ind].y*wgt;
        }
      }
    }

    __global__ void nrmGrid_kernel(float *Grd, float nrm, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if ( iu < nu &&  iv < nu){
          Grd[iv*nu + iu] = Grd[iv*nu+iu]*nrm;
      }
    }

    __global__ void corrGrid_kernel(float2 *Grd, float *corr, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if (iu < nu && iv < nu ){
          Grd[iv*nu + iu].x = Grd[iv*nu+iu].x*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
          Grd[iv*nu + iu].y = Grd[iv*nu+iu].y*corr[nu/2]*corr[nu/2]/(corr[iu]*corr[iv]);
      }
    }

    // *********************
    // BEAM KERNELS
    // *********************
    __global__ void nrmBeam_kernel(float *bmR, float nrm, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if(iu < nu && iv < nu){
        bmR[iv*nu+iu] = nrm*bmR[iv*nu+iu];
      }
    }

    // *********************
    // MORE semi-USEFUL KERNELS
    // *********************

    __global__ void shiftGrid_kernel(float2 *Grd, float2 *nGrd, int nu){
      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      if(iu < nu && iv < nu){
        int niu,niv,nud2 = 0.5*nu;
        if(iu < nud2) niu = nud2+iu;
          else niu = iu-nud2;
        if(iv < nud2) niv = nud2+iv;
          else niv = iv-nud2;
        nGrd[niv*nu + niu].x = Grd[iv*nu+iu].x;
        nGrd[niv*nu + niu].y = Grd[iv*nu+iu].y;
      }
    }

    __global__ void trimIm_kernel(float2 *im, float *nim, int noff, int nx, int nnx){
      int ix = blockDim.x*blockIdx.x + threadIdx.x;
      int iy = blockDim.y*blockIdx.y + threadIdx.y;
      if(iy < nnx && ix < nnx){
        nim[iy*nnx + ix] = im[(iy+noff)*nx+ix+noff].x;
      }
    }
    """
module = nvcc.SourceModule(code)
gridVis_wBM_kernel = module.get_function("gridVis_wBM_kernel")
shiftGrid_kernel = module.get_function("shiftGrid_kernel")
nrmGrid_kernel = module.get_function("nrmGrid_kernel")
wgtGrid_kernel = module.get_function("wgtGrid_kernel")
dblGrid_kernel = module.get_function("dblGrid_kernel")
corrGrid_kernel = module.get_function("corrGrid_kernel")
nrmBeam_kernel = module.get_function("nrmBeam_kernel")
trimIm_kernel = module.get_function("trimIm_kernel")

# -------------------
# CLEAN kernels
# -------------------

find_max_kernel_source = \
    """
    // Function to compute 1D array positioncuda_gri
    #define GRID(x,y,W) ((x)+((y)*W))

    __global__ void find_max_kernel(float* dimg, int* maxid, float maxval, int W, int H, float* model)
    {
      // Identify place on grid
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      int id  = GRID(idy,idx,H);

      // Ignore boundary pixels
      if (idx>-1 && idx<W && idy>-1 && idy<H) {
        // Is this greater than the current max?
        if (dimg[id]==maxval) {
          // Do an atomic replace
          // This might be #improvable#, but I think atomic operations are actually most efficient
          // in a situation like this where few (typically 1) threads will pass this conditional.
          // Note: this is a race condition!  If there are multiple instances of the max value,
          // this will end up returning one randomly
          // See atomic operation info here: http://rpm.pbone.net/index.php3/stat/45/idpl/12463013/numer/3/nazwa/atomicExch
          // See also https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
          int dummy = atomicExch(maxid,id);
        }
      }
      // Update the model
      void __syncthreads();
      if (id==maxid[0]) {
        model[id]+=dimg[id];
      }
    }
    """
find_max_kernel = cuda_compile(find_max_kernel_source, "find_max_kernel")

sub_beam_kernel_source = \
    """
    // Function to compute 1D array position
    #define GRID(x,y,W) ((x)+((y)*W))
    // Inverse
    #define IGRIDX(x,W) ((x)%(W))
    #define IGRIDY(x,W) ((x)/(W))

    __global__ void sub_beam_kernel(float* dimg, float* dpsf, int* mid, float* cimg, float* cpsf, float scaler, int W, int H)
    {
      // Identify place on grid
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      int id  = GRID(idy,idx,H);
      // Identify position of maximum
      int midy = IGRIDX(mid[0],W);
      int midx = IGRIDY(mid[0],W);
      // Calculate position on the dirty beam
      int bidy = (idx-midx)+W/2;
      int bidx = (idy-midy)+H/2;
      int bid = GRID(bidx,bidy,W);

      // Stay within the bounds
      if (idx>-1 && idx<W && idy>-1 && idy<H && bidx>-1 && bidx<W && bidy>-1 && bidy<H) {
        // Subtract dirty beam from dirty map
        dimg[id]=dimg[id]-dpsf[bid]*scaler;
        // Add clean beam to clean map
        cimg[id]=cimg[id]+cpsf[bid]*scaler;
      };
    }
    """
sub_beam_kernel = cuda_compile(sub_beam_kernel_source, "sub_beam_kernel")

add_noise_kernel = ElementwiseKernel(
    "float *a, float* b, int N",
    "b[i] = a[i]+b[i]",
    "gpunoise")


######################
# Gridding functions
######################

def spheroid(eta, m, alpha):
    """
    Calculates spheriodal wave functions. See Schwab 1984 for details.
    This implementation follows MIRIAD's grid.for subroutine.
    """

    twoalp = 2 * alpha
    if np.abs(eta) > 1:
        print 'bad eta value!'
    if (twoalp < 1 or twoalp > 4):
        print 'bad alpha value!'
    if (m < 4 or m > 8):
        print 'bad width value!'

    etalim = np.float32([1., 1., 0.75, 0.775, 0.775])
    nnum = np.int8([5, 7, 5, 5, 6])
    ndenom = np.int8([3, 2, 3, 3, 3])
    p = np.float32(
        [
            [[5.613913E-2, -3.019847E-1, 6.256387E-1,
              -6.324887E-1, 3.303194E-1, 0.0, 0.0],
             [6.843713E-2, -3.342119E-1, 6.302307E-1,
              -5.829747E-1, 2.765700E-1, 0.0, 0.0],
             [8.203343E-2, -3.644705E-1, 6.278660E-1,
              -5.335581E-1, 2.312756E-1, 0.0, 0.0],
             [9.675562E-2, -3.922489E-1, 6.197133E-1,
              -4.857470E-1, 1.934013E-1, 0.0, 0.0],
             [1.124069E-1, -4.172349E-1, 6.069622E-1,
              -4.405326E-1, 1.618978E-1, 0.0, 0.0]
            ],
            [[8.531865E-4, -1.616105E-2, 6.888533E-2,
              -1.109391E-1, 7.747182E-2, 0.0, 0.0],
             [2.060760E-3, -2.558954E-2, 8.595213E-2,
              -1.170228E-1, 7.094106E-2, 0.0, 0.0],
             [4.028559E-3, -3.697768E-2, 1.021332E-1,
              -1.201436E-1, 6.412774E-2, 0.0, 0.0],
             [6.887946E-3, -4.994202E-2, 1.168451E-1,
              -1.207733E-1, 5.744210E-2, 0.0, 0.0],
             [1.071895E-2, -6.404749E-2, 1.297386E-1,
              -1.194208E-1, 5.112822E-2, 0.0, 0.0]
            ]
        ])
    q = np.float32(
        [
            [[1., 9.077644E-1, 2.535284E-1],
             [1., 8.626056E-1, 2.291400E-1],
             [1., 8.212018E-1, 2.078043E-1],
             [1., 7.831755E-1, 1.890848E-1],
             [1., 7.481828E-1, 1.726085E-1]
            ],
            [[1., 1.101270, 3.858544E-1],
             [1., 1.025431, 3.337648E-1],
             [1., 9.599102E-1, 2.918724E-1],
             [1., 9.025276E-1, 2.575337E-1],
             [1., 8.517470E-1, 2.289667E-1]
            ]
        ])

    i = m - 4
    if (np.abs(eta) > etalim[i]):
        ip = 1
        x = eta * eta - 1
    else:
        ip = 0
        x = eta * eta - etalim[i] * etalim[i]
        # numerator via Horner's rule
    mnp = nnum[i] - 1
    num = p[ip, twoalp, mnp]
    for j in np.arange(mnp):
        num = num * x + p[ip, twoalp, mnp - 1 - j]
        # denominator via Horner's rule
    nq = ndenom[i] - 1
    denom = q[ip, twoalp, nq]
    for j in np.arange(nq):
        denom = denom * x + q[ip, twoalp, nq - 1 - j]

    return np.float32(num / denom)


def gcf(n, width):
    """
    Create table with spheroidal gridding function, C
    This implementation follows MIRIAD's grid.for subroutine.
    """
    alpha = 1.
    j = 2 * alpha
    p = 0.5 * j
    phi = np.zeros(n, dtype=np.float32)
    for i in np.arange(n):
        x = np.float32(2 * i - (n - 1)) / (n - 1)
        phi[i] = (np.sqrt(1 - x * x) ** j) * spheroid(x, width, p)
    return phi


def corrfun(n, width):
    """
    Create gridding correction function, c
    This implementation follows MIRIAD's grid.for subroutine.
    """
    alpha = 1.
    dx = 2. / n
    i0 = n / 2 + 1
    phi = np.zeros(n, dtype=np.float32)
    for i in np.arange(n):
        x = (i - i0 + 1) * dx
        phi[i] = spheroid(x, width, alpha)
    return phi


def cuda_gridvis(csrh_sun, csrh_satellite, settings, plan, chan):
    """
    Grid the visibilities parallelized by pixel.
    References:
      - Chapter 10 in "Interferometry and Synthesis in Radio Astronomy"
          by Thompson, Moran, & Swenson
      - Daniel Brigg's PhD Thesis: http://www.aoc.nrao.edu/dissertations/dbriggs/
    """
    print "Gridding the visibilities"
    t_start = time.time()

    #f = pyfits.open(settings['vfile'])

    # unpack parameters
    vfile = settings['vfile']
    briggs = settings['briggs']
    imsize = settings['imsize']
    cell = settings['cell']
    nx = np.int32(2 * imsize)
    noff = np.int32((nx - imsize) / 2)

    ## constants
    arc2rad = np.float32(np.pi / 180. / 3600.)
    du = np.float32(1. / (arc2rad * cell * nx))
    ## grab data
    #f = pyfits.open(settings['vfile'])

    Data = np.ndarray(
        shape=(44, 44, 16),
        dtype=complex)
    UVW = np.ndarray(shape=(780, 1), dtype='float64')
    Data, UVW = visibility(csrh_sun, csrh_satellite, chan)
    print "UVW*****\n", UVW

    # determin the file type (uvfits or fitsidi)
    h_uu = np.ndarray(shape=(780), dtype='float64')
    h_vv = np.ndarray(shape=(780), dtype='float64')
    h_rere = np.ndarray(shape=(780), dtype='float32')
    h_imim = np.ndarray(shape=(780), dtype='float32')

    freq = 1702500000.
    light_speed = 299792458.  # Speed of light

    ## quickly figure out what data is not flagged
    #np.float32(f[7].header['CRVAL3']) 299792458vvvv
    #good  = np.where(f[0].data.data[:,0,0,0,0,0,0] != 0)

    #h_u   = np.float32(freq*f[0].data.par('uu')[good])
    #h_v   = np.float32(freq*f[0].data.par('vv')[good])

    blen = 0

    for antenna1 in range(0, 39):
        for antenna2 in range(antenna1 + 1, 40):
            h_rere[blen] = Data[antenna1][antenna2][chan].real
            h_imim[blen] = Data[antenna1][antenna2][chan].imag
            h_uu[blen] =  freq*UVW[blen][0]
            h_vv[blen] =  freq*UVW[blen][1]
            blen += 1

    print "h_u", h_uu
    #h_u = np.float32(h_u.ravel())
    #h_v = np.float32(h_v.ravel())
    gcount = np.int32(np.size(h_uu))
    #gcount = len(gcount.ravel())
    #h_re = np.float32(h_re.ravel())
    #h_im = np.float32(h_im.ravel())
    #freq = 3.45E11  #np.float32(f[0].header['CRVAL4'])

    blen = 0
    bl_order = np.ndarray(shape=(780, 2), dtype=int)
    good = []

    for border1 in range(0, 39):
        for border2 in range(border1 + 1, 40):
            bl_order[blen][0] = border1
            bl_order[blen][1] = border2
            blen = blen + 1

    blen = 0

    h_u = []
    h_v = []
    h_re = []
    h_im = []
    Flag_Ant = [0, 4, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 37, 38, 39]
    for blen in range(0, 780):
        if (bl_order[blen][0] not in Flag_Ant) and (bl_order[blen][1] not in Flag_Ant):
            good.append(blen)
            h_u.append(h_uu[blen])
            h_v.append(h_vv[blen])
            h_re.append(h_rere[blen])
            h_im.append(h_imim[blen])

    #print "Good:",good

    gcount = np.int32(np.size(h_u))
    ## assume data is unpolarized
    #print chan
    print 'GCOUNT', gcount
    #print "H_U", h_u
    #print "H_V", h_v
    #print h_re
    #print h_im

    # h_ : host,  d_ : device
    h_grd = np.zeros((nx, nx), dtype=np.complex64)
    h_cnt = np.zeros((nx, nx), dtype=np.int32)
    d_u = gpu.to_gpu(np.array(h_uu,dtype='float32'))
    d_v = gpu.to_gpu(np.array(h_vv,dtype='float32'))
    d_re = gpu.to_gpu(np.array(h_rere,dtype='float32'))
    d_im = gpu.to_gpu(np.array(h_imim,dtype='float32'))
    d_cnt = gpu.zeros((np.int(nx), np.int(nx)), np.int32)
    d_grd = gpu.zeros((np.int(nx), np.int(nx)), np.complex64)
    d_ngrd = gpu.zeros_like(d_grd)
    d_bm = gpu.zeros_like(d_grd)
    d_nbm = gpu.zeros_like(d_grd)
    d_fim = gpu.zeros((np.int(imsize), np.int(imsize)), np.float32)
    ## define kernel parameters
    if imsize == 1024:
        blocksize2D = (8, 16, 1)
        gridsize2D = (np.int(np.ceil(1. * nx / blocksize2D[0])), np.int(np.ceil(1. * nx / blocksize2D[1])))
        blocksizeF2D = (16, 16, 1)
        gridsizeF2D = (np.int(np.ceil(1. * imsize / blocksizeF2D[0])), np.int(np.ceil(1. * imsize / blocksizeF2D[1])))
        blocksize1D = (256, 1, 1)
    else:
        blocksize2D = (16, 32, 1)
        gridsize2D = (np.int(np.ceil(1. * nx / blocksize2D[0])), np.int(np.ceil(1. * nx / blocksize2D[1])))
        blocksizeF2D = (32, 32, 1)
        gridsizeF2D = (np.int(np.ceil(1. * imsize / blocksizeF2D[0])), np.int(np.ceil(1. * imsize / blocksizeF2D[1])))
        blocksize1D = (512, 1, 1)

    gridsize1D = (np.int(np.ceil(1. * gcount / blocksize1D[0])), 1)

    # ------------------------
    # make gridding kernels
    # ------------------------
    ## make spheroidal convolution kernel (don't mess with these!)
    width = 6.
    ngcf = 24.
    h_cgf = gcf(ngcf, width)
    ## make grid correction
    h_corr = corrfun(nx, width)
    d_cgf = module.get_global('cgf')[0]
    d_corr = gpu.to_gpu(h_corr)
    cu.memcpy_htod(d_cgf, h_cgf)

    # ------------------------
    # grid it up
    # ------------------------
    d_umax = gpu.max(cumath.fabs(d_u))
    d_vmax = gpu.max(cumath.fabs(d_v))
    umax = np.int32(np.ceil(d_umax.get() / du))
    vmax = np.int32(np.ceil(d_vmax.get() / du))

    ## grid ($$)
    #  This should be improvable via:
    #    - shared memory solution? I tried...
    #    - better coalesced memory access? I tried...
    #    - reorganzing and indexing UV data beforehand?
    #       (i.e. http://www.nvidia.com/docs/IO/47905/ECE757_Project_Report_Gregerson.pdf)
    #    - storing V(u,v) in texture memory?
    gridVis_wBM_kernel(d_grd, d_bm, d_cnt, d_u, d_v, d_re, d_im, nx, du, gcount, umax, vmax, \
                       block=blocksize2D, grid=gridsize2D)

    ## apply weights
    wgtGrid_kernel(d_bm, d_cnt, briggs, nx, block=blocksize2D, grid=gridsize2D)
    hfac = np.int32(1)
    dblGrid_kernel(d_bm, nx, hfac, block=blocksize2D, grid=gridsize2D)
    shiftGrid_kernel(d_bm, d_nbm, nx, block=blocksize2D, grid=gridsize2D)
    ## normalize

    wgtGrid_kernel(d_grd, d_cnt, briggs, nx, block=blocksize2D, grid=gridsize2D)
    ## Reflect grid about v axis
    hfac = np.int32(-1)
    dblGrid_kernel(d_grd, nx, hfac, block=blocksize2D, grid=gridsize2D)
    ## Shift both
    shiftGrid_kernel(d_grd, d_ngrd, nx, block=blocksize2D, grid=gridsize2D)

    # ------------------------
    # Make the beam
    # ------------------------
    ## Transform to image plane
    fft.fft(d_nbm, d_bm, plan)
    ## Shift
    shiftGrid_kernel(d_bm, d_nbm, nx, block=blocksize2D, grid=gridsize2D)
    ## Correct for C
    corrGrid_kernel(d_nbm, d_corr, nx, block=blocksize2D, grid=gridsize2D)
    # Trim
    trimIm_kernel(d_nbm, d_fim, noff, nx, imsize, block=blocksizeF2D, grid=gridsizeF2D)
    ## Normalize
    d_bmax = gpu.max(d_fim)
    bmax = d_bmax.get()
    bmax = np.float32(1. / bmax)
    nrmBeam_kernel(d_fim, bmax, imsize, block=blocksizeF2D, grid=gridsizeF2D)
    ## Pull onto CPU
    dpsf = d_fim.get()

    # ------------------------
    # Make the map
    # ------------------------
    ## Transform to image plane
    fft.fft(d_ngrd, d_grd, plan)
    ## Shift
    shiftGrid_kernel(d_grd, d_ngrd, nx, block=blocksize2D, grid=gridsize2D)
    ## Correct for C
    corrGrid_kernel(d_ngrd, d_corr, nx, block=blocksize2D, grid=gridsize2D)
    ## Trim
    trimIm_kernel(d_ngrd, d_fim, noff, nx, imsize, block=blocksizeF2D, grid=gridsizeF2D)
    ## Normalize (Jy/beam)
    nrmGrid_kernel(d_fim, bmax, imsize, block=blocksizeF2D, grid=gridsizeF2D)

    ## Finish timers
    t_end = time.time()
    t_full = t_end - t_start
    print "Gridding execution time %0.5f" % t_full + ' s'
    print "\t%0.5f" % (t_full / gcount) + ' s per visibility'

    ## Return dirty psf (CPU) and dirty image (GPU)
    return dpsf, d_fim


######################
# CLEAN functions
######################

def serial_clean_beam(dpsf, window=20):
    """
    Clean a dirty beam on the CPU
    A very simple approach - just extract the central beam #improvable#
    Another solution would be fitting a 2D Gaussian,
    e.g. http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
    """
    print "Cleaning the dirty beam"
    h, w = np.shape(dpsf)
    cpsf = np.zeros([h, w])
    cpsf[w / 2 - window:w / 2 + window, h / 2 - window:h / 2 + window] = dpsf[w / 2 - window:w / 2 + window,
                                                                         h / 2 - window:h / 2 + window]
    ##Normalize
    cpsf = cpsf / np.max(cpsf)
    return np.float32(cpsf)


def gpu_getmax(map):
    """
    Use pycuda to get the maximum absolute deviation of the residual map,
    with the correct sign
    """
    imax = gpu.max(cumath.fabs(map)).get()
    if gpu.max(map).get() != imax: imax *= -1
    return np.float32(imax)


def cuda_hogbom(gpu_dirty, gpu_dpsf, gpu_cpsf, thresh=0.2, damp=1, gain=0.1, prefix='test'):
    """
    Use CUDA to implement the Hogbom CLEAN algorithm

    A nice description of the algorithm is given by the NRAO, here:
    http://www.cv.nrao.edu/~abridle/deconvol/node8.html

    Parameters:
    * dirty: The dirty image (2D numpy array)
    * dpsf: The dirty beam psf  (2D numpy array)
    * thresh: User-defined threshold to stop iteration, as a fraction of the max pixel intensity (float)
    * damp: The damping factor to scale the dirty beam by
    * prefix: prefix for output image file names
    """
    height, width = np.shape(gpu_dirty)
    ## Grid parameters - #improvable#
    tsize = 8
    blocksize = (int(tsize), int(tsize), 1)  # The number of threads per block (x,y,z)
    gridsize = (int(width / tsize), int(height / tsize))  # The number of thread blocks     (x,y)
    ## Setup cleam image and point source model
    gpu_pmodel = gpu.zeros([height, width], dtype=np.float32)
    gpu_clean = gpu.zeros([height, width], dtype=np.float32)
    ## Setup GPU constants
    gpu_max_id = gpu.to_gpu(np.int32(0))
    imax = gpu_getmax(gpu_dirty)
    thresh_val = np.float32(thresh * imax)
    ## Steps 1-3 - Iterate until threshold has been reached
    t_start = time.time()
    i = 0
    print "Subtracting dirty beam..."
    while (abs(imax) > (thresh_val)) and (i < 1000):
        '''if (np.mod(i, 100) == 0):
            print "Hogbom iteration", i'''
        ## Step 1 - Find max
        find_max_kernel(gpu_dirty, gpu_max_id, imax, np.int32(width), np.int32(height), gpu_pmodel, \
                        block=blocksize, grid=gridsize)
        ## Step 2 - Subtract the beam (assume that it is normalized to have max 1)
        ##          This kernel simultaneously reconstructs the CLEANed image.
        '''if PLOTME: print "Subtracting dirty beam " + str(i) + ", maxval=%0.8f" % imax + ' at x=' + str(
            gpu_max_id.get() % width) + \
                         ', y=' + str(gpu_max_id.get() / width)'''
        sub_beam_kernel(gpu_dirty, gpu_dpsf, gpu_max_id, gpu_clean, gpu_cpsf, np.float32(gain * imax), np.int32(width), \
                        np.int32(height), block=blocksize, grid=gridsize)
        i += 1
        ## Step 3 - Find maximum value using gpuarray
        imax = gpu_getmax(gpu_dirty)
    t_end = time.time()
    t_full = t_end - t_start
    print "Hogbom execution time %0.5f" % t_full + ' s'
    #print "\t%0.5f" % (t_full / i) + ' s per iteration'
    ## Step 4 - Add the residuals back in
    add_noise_kernel(gpu_dirty, gpu_clean, np.float32(width + height))
    return gpu_dirty, gpu_pmodel, gpu_clean


if __name__ == '__main__':

    ## Load command line options

    # Which example?
    if len(sys.argv) > 1:
        example = sys.argv[1]
    if len(sys.argv) > 2:
        ISIZE = float(sys.argv[2])
    else:
        ISIZE = 1024
        # Make plots?
    if len(sys.argv) > 3:
        PLOTME = float(sys.argv[3])
    else:
        PLOTME = 1

    # Load settings for each example
    settings = dict([])
    # setup filename
    settings['vfile'] = example
    settings['imsize'] = np.int32(ISIZE)  # number of image pixels
    # 1 degree viewfield, 1*3.1415926535/180*3600 =
    settings['cell'] = np.float32( 3600. / ISIZE)  # pixel size in arcseconds (rad ? degree?)
    settings['briggs'] = np.float32(1e7)  # weight parameter

    ## make cuFFT plan #improvable#

    ## Create the PSF & dirty image
    #   dpsf - PSF, gpu_im ( dirty image)
    #   dpsf is computed by CPU, gpu_im is in the GPU

    imsize = settings['imsize']

    # nx - 2 imsize, it means 2048 when imsize=1024
    nx = np.int32(2 * imsize)
    # create fft plan nx*nx
    plan = fft.Plan((np.int(nx), np.int(nx)), np.complex64, np.complex64)

    '''f = pyfits.open(settings['vfile'])
    channel = f[0].data.data.shape[3]'''

    inFile_sun = open('right.rawdata', 'rb')
    inFile_sun.seek(0, 0)
    csrh_sun = CSRHRawData(inFile_sun)

    inFile_satellite = open('CSRH_20140512-141019_181161703_Satellite', 'rb')
    inFile_satellite.seek(0, 0)
    csrh_satellite = CSRHRawData(inFile_satellite)

    print " SUN: Search frame header  ....."
    if csrh_sun.readOneFrame() == False:
        print("Finished. Not a complete data!")
        inFile_sun.close()
        exit(0)

    #print "VIS------\n", csrh_sun.csrhData
    print " Satellite: Search frame header  ....."
    if csrh_satellite.readOneFrame() == False:
        print("Finished. Not a complete data!")
        inFile_satellite.close()
        exit(0)

    for chan in range(4, 5):

        #dpsf, gpu_im = cuda_gridvis(f, settings, plan, chan)
        dpsf, gpu_im = cuda_gridvis(csrh_sun, csrh_satellite, settings, plan, chan)

        gpu_dpsf = gpu.to_gpu(dpsf)
        print "**********************************************"

        if PLOTME:
            dirty = np.roll(np.fliplr(gpu_im.get()), 1, axis=1)

        ## Clean the PSF
        if imsize >= 1024:
            cpsf = serial_clean_beam(dpsf, imsize / 50.)
        elif imsize >= 512:
            cpsf = serial_clean_beam(dpsf, imsize / 25.)
        elif imsize >= 256:
            cpsf = serial_clean_beam(dpsf, imsize / 12.)

        gpu_cpsf = gpu.to_gpu(cpsf)

        if PLOTME:
            print "Plotting dirty and cleaned beam"
            fig, axs = plt.subplots();  #1, 2, sharex=True, sharey=True);
            plt.subplots_adjust(wspace=0)
            axs.imshow(dpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            #axs[1].imshow(cpsf, vmin=np.percentile(dpsf, 0), vmax=np.percentile(dpsf, 99), cmap=cm.gray)
            pathPrefix = os.getenv("MUSEROS_WORK")
            if pathPrefix == None:
                plt.savefig('test_cleanbeam_%d.png' % chan)
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                plt.savefig(pathPrefix + '/' + 'test_cleanbeam_%d.png' % chan)
            plt.close()

        ## Run CLEAN
        gpu_dirty, gpu_pmodel, gpu_clean = cuda_hogbom(gpu_im, gpu_dpsf, gpu_cpsf, thresh=0.2, gain=0.1)

        if PLOTME:
            prefix = example
            prefix, ext = os.path.splitext(os.path.basename(prefix))
            try:
                vra
            except NameError:
                vra = [np.percentile(dirty, 1), np.percentile(dirty, 99)]

            print "Plotting dirty image and dirty image after iterative source removal"
            fig, axs = plt.subplots()  #1, 2, sharex=True, sharey=True, figsize=(12.2, 6));
            plt.subplots_adjust(wspace=0)
            axs.imshow(dirty, vmin=vra[0], vmax=vra[1], cmap=cm.jet, origin='lower')
            axs.set_title('Original dirty image')
            #axs[1].imshow(np.roll(np.fliplr(gpu_dirty.get()), 1, axis=1), vmin=vra[0], vmax=vra[1], cmap=cm.gray,
            #          origin='lower')
            #axs[1].set_title('Dirty image cleaned of sources')
            pathPrefix = os.getenv("MUSEROS_WORK")
            if pathPrefix == None:
                plt.savefig(prefix + '_dirty_final_%d.png' % chan)
                #dirty.tofile(prefix+'_dirty_final_axs0_%d.dat'%chan)
                #(np.roll(np.fliplr(gpu_dirty.get()),1,axis=1)).tofile(prefix+'_dirty_final_axs1.dat')
            else:
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                plt.savefig(pathPrefix + '/' + prefix + '_dirty_final_%d.png' % chan)
                #dirty.tofile(pathPrefix+'/'+prefix+'_dirty_final_axs0_%d.dat'%chan)
                #(np.roll(np.fliplr(gpu_dirty.get()),1,axis=1)).tofile(pathPrefix+'/'+prefix+'_dirty_final_axs1.dat')
            plt.close()

            print "Plotting dirty image and final clean image"
            vra = [np.percentile(dirty, 1), np.percentile(dirty, 99)]
            fig, axs = plt.subplots(figsize=(6.1, 6))  #1, 2, sharex=True, sharey=True, figsize=(12.2, 6));
            plt.subplots_adjust(wspace=0)
            clean = np.roll(np.fliplr(gpu_clean.get()), 1, axis=1)
            #axs.imshow(dirty, vmin=vra[0], vmax=vra[1], cmap=cm.gray, origin='lower')
            #axs.set_title('Original dirty image')
            axs.imshow(clean, vmin=vra[0], vmax=vra[1], cmap=cm.hot, origin='lower')
            axs.set_title('Final cleaned image')
            if os.getenv("MUSEROS_WORK") == None:
                plt.savefig(prefix + '_clean_final_%d.png' % chan)
                #dirty.tofile(prefix+'_clean_final_axs0_%d.dat'%chan)
                #clean.tofile(prefix+'_clean_final_axs1_%d.dat'%chan)
            else:
                pathPrefix = os.getenv("MUSEROS_WORK")
                if pathPrefix[-1:] == '/':
                    pathPrefix = pathPrefix[:-1]
                plt.savefig(pathPrefix + '/' + prefix + '_clean_final_%d.png' % chan)
                #dirty.tofile(pathPrefix+'/'+prefix+'_clean_final_axs0_%d.dat'%chan)
                #clean.tofile(pathPrefix+'/'+prefix+'_clean_final_axs1_%d.dat'%chan)
            plt.close()





