import sys, os, datetime, time, math
import codecs, string, sgp4io, struct, string
from muserenv import *
from musertime import *

#from muserlogger import  logger
logger = logging.getLogger('muser')

# Class MuserFrameHeader

class MuserFrameHeader(object):
    def __init__(self):
        self.system_time = None
        self.frequency_switch = 0l
        self.bandwidth = 0l
        self.quantization_leve = 0l
        self.delay_switch = 0l
        self.strip_switch = 0l
        self.sub_band_switch = 0l

# Muser Data base class
class MuserBase(object):
    # Dictionary of Bandwidth of Muser
    IF_BANDWIDTH = {0x33333333: 25000000, 0x77777777: 12500000, 0x88888888: 6250000, 0xbbbbbbbb: 3125000,
                    0xcccccccc: 1562500}

    # Dictionary of Bandwidth of Muser-I
    LOOP_MODE_LOW = {0x0: ( 0, 0, 400000000, 1 ), 0x1: ( 1, 0, 400000000, 1 ), 0x2: ( 0, 16, 800000000, 2 ),
                     0x3: ( 1, 16, 800000000, 2 ), 0x4: ( 0, 32, 1200000000, 3 ), 0x5: ( 1, 32, 1200000000, 3 ),
                     0x6: ( 0, 48, 1600000000, 4 ), 0x7: ( 1, 48, 1600000000, 4 )}

    # Dictionary of Non-loop mode of Muser-I
    NON_LOOP_MODE_LOW = {0x3333: (0 , 400000000, 1), 0x7777: ( 16, 800000000, 2),
                         0xbbbb: ( 32, 1200000000, 3), 0xcccc: ( 48, 1600000000, 4)}

    # Dictionary of Loop mode of Muser-II
    LOOP_MODE_HIGH = {0X0: ( 0, 0, 2000000000 ), 0X1: ( 1, 0, 2000000000 ), 0X2: ( 0, 16, 2400000000 ),
                      0X3: ( 1, 16, 2400000000 ), 0X4: ( 0, 32, 2800000000 ), 0X5: ( 1, 32, 2800000000 ),
                      0X6: ( 0, 48, 3200000000 ), 0X7: ( 1, 48, 3200000000 ), 0X8: ( 0, 64, 3600000000 ),
                      0X9: ( 1, 64, 3600000000 ), 0XA: ( 0, 80, 4000000000 ), 0XB: ( 1, 80, 4000000000 ),
                      0XC: ( 0, 96, 4400000000 ), 0XD: ( 1, 96, 4400000000 ), 0XE: ( 0, 112, 4800000000 ),
                      0XF: ( 1, 112, 4800000000 ), 0X10: ( 0, 128, 5200000000 ), 0X11: ( 1, 128, 5200000000 ),
                      0X12: ( 0, 144, 5600000000 ), 0X13: ( 1, 144, 5600000000 ), 0X14: ( 0, 160, 6000000000 ),
                      0X15: ( 1, 160, 6000000000 ), 0X16: ( 0, 176, 6400000000 ), 0X17: ( 1, 176, 6400000000 ),
                      0X18: ( 0, 192, 6800000000 ), 0X19: ( 1, 192, 6800000000 ), 0X1A: ( 0, 208, 7200000000 ),
                      0X1B: ( 1, 208, 7200000000 ), 0X1C: ( 0, 224, 7600000000 ), 0X1D: ( 1, 224, 7600000000 ),
                      0X1E: ( 0, 240, 8000000000 ), 0X1F: ( 1, 240, 8000000000 ), 0X20: ( 0, 256, 8400000000 ),
                      0X21: ( 1, 256, 8400000000 ), 0X22: ( 0, 272, 8800000000 ), 0X23: ( 1, 272, 8800000000 ),
                      0X24: ( 0, 288, 9200000000 ), 0X25: ( 1, 288, 9200000000 ), 0X26: ( 0, 304, 9600000000 ),
                      0X27: ( 1, 304, 9600000000 ), 0X28: ( 0, 320, 10000000000 ), 0X29: ( 1, 320, 10000000000 ),
                      0X2A: ( 0, 336, 10400000000 ), 0X2B: ( 1, 336, 10400000000 ), 0X2C: ( 0, 352, 10800000000 ),
                      0X2D: ( 1, 352, 10800000000 ), 0X2E: ( 0, 368, 11200000000 ), 0X2F: ( 1, 368, 11200000000 ),
                      0X30: ( 0, 384, 11600000000 ), 0X31: ( 1, 384, 11600000000 ), 0X32: ( 0, 400, 12000000000 ),
                      0X33: ( 1, 400, 12000000000 ), 0X34: ( 0, 416, 12400000000 ), 0X35: ( 1, 416, 12400000000 ),
                      0X36: ( 0, 432, 12800000000 ), 0X37: ( 1, 432, 12800000000 ), 0X38: ( 0, 448, 13200000000 ),
                      0X39: ( 1, 448, 13200000000 ), 0X3A: ( 0, 464, 13600000000 ), 0X3B: ( 1, 464, 13600000000 ),
                      0X3C: ( 0, 480, 14000000000 ), 0X3D: ( 1, 480, 14000000000 ), 0X3E: ( 0, 496, 14400000000 ),
                      0X3F: ( 1, 496, 14400000000 ), 0X40: ( 0, 512, 14600000000 ), 0X41: ( 1, 512, 14600000000 )}
    NON_LOOP_MODE_HIGH = {0x0: ( 0, 2000000000, 1), 0x101: ( 16, 2400000000, 2), 0x202: ( 32, 2800000000, 3),
                          0x303: ( 48, 3200000000, 4), 0x404: ( 64, 3600000000, 5), 0x505: ( 80, 4000000000, 6),
                          0x606: ( 96, 4400000000, 7), 0x707: ( 112, 4800000000, 8), 0x808: ( 128, 5200000000, 9),
                          0x909: ( 144, 5600000000, 10), 0xa0a: ( 160, 6000000000, 11), 0xb0b: ( 176, 6400000000, 12),
                          0xc0c: ( 192, 6800000000, 13), 0xd0d: ( 208, 7200000000, 14), 0xe0e: ( 224, 7600000000, 15),
                          0xf0f: ( 240, 8000000000, 16), 0x1010: ( 256, 8400000000, 17), 0x1111: ( 272, 8800000000, 18),
                          0x1212: ( 288, 9200000000, 19), 0x1313: ( 304, 9600000000, 20), 0x1414: ( 320, 10000000000, 21),
                          0x1515: ( 336, 10400000000, 22), 0x1616: ( 352, 10800000000, 23),
                          0x1717: ( 368, 11200000000, 24),
                          0x1818: ( 384, 11600000000, 25), 0x1919: ( 400, 12000000000, 26),
                          0x1a1a: ( 416, 12400000000, 27),
                          0x1b1b: ( 432, 12800000000, 28), 0x1c1c: ( 448, 13200000000, 29),
                          0x1d1d: ( 464, 13600000000, 30),
                          0x1e1e: ( 480, 14000000000, 31), 0x1f1f: ( 496, 14400000000, 32),
                          0x2020: ( 512, 14600000000, 33)}


    def __init__(self, sub_array=1):
        self.debug = 0
        self.sub_array = sub_array
        self.total_polarization = 2
        if self.sub_array==1:
            self.total_band = 4
        else:
            self.total_band = 32

        self.env = muserenv
        self.input_file_name = ''


    def convert_time(self, stime):
        tmp = MuserTime()
        # print '%x' % stime
        tmp.nanosecond = (stime & 0x3f)
        if tmp.nanosecond >= 50:
            tmp.nanosecond = 0
        tmp.nanosecond *= 20
        stime >>= 6
        # read microsecond, 6-15
        tmp.microsecond = (stime & 0x3ff)
        stime >>= 10
        # read millisecond 16-25
        tmp.millisecond = (stime & 0x3ff)
        stime >>= 10
        # read second, 26-31
        tmp.second = (stime & 0x3f)
        stime >>= 6
        # read minute, 32-37
        tmp.minute = (stime & 0x3f)
        stime >>= 6
        # read hour, 38-42
        tmp.hour = (stime & 0x1f)
        stime >>= 5
        # read day
        tmp.day = (stime & 0x1f)
        stime >>= 5
        # read month, 48-51
        tmp.month = (stime & 0x0f)
        stime >>= 4
        # read year
        tmp.year = (stime & 0xfff) + 2000
        # print tmp
        return tmp

    def signed(self, a):
        if a >= 128l * 256l * 256l:
            a = a - 256l * 256l * 256l
        return a

    def convert_cross_correlation(self, buff):
        # read imaginary part of second channel
        c1 = c2 = complex()

        imag2 = struct.unpack('B', buff[8])[0]
        imag2 <<= 8
        imag2 |= struct.unpack('B', buff[4])[0]
        imag2 <<= 8
        imag2 |= struct.unpack('B', buff[0])[0]
        imag2 = self.signed(imag2)

        # read real part of second channel
        real2 = struct.unpack('B', buff[9])[0]
        real2 <<= 8
        real2 |= struct.unpack('B', buff[5])[0]
        real2 <<= 8
        real2 |= struct.unpack('B', buff[1])[0]
        real2 = self.signed(real2)

        c2 = complex(real2, imag2)

        # read imaginary part of second channel
        imag1 = struct.unpack('B', buff[10])[0]
        imag1 <<= 8
        imag1 |= struct.unpack('B', buff[6])[0]
        imag1 <<= 8
        imag1 |= struct.unpack('B', buff[2])[0]
        imag1 = self.signed(imag1)

        # read real part of second channel
        real1 = struct.unpack('B', buff[11])[0]
        real1 <<= 8
        real1 |= struct.unpack('B', buff[7])[0]
        real1 <<= 8
        real1 |= struct.unpack('B', buff[3])[0]
        real1 = self.signed(real1)

        c1 = complex(real1, imag1)
        return c1, c2

    def convert_auto_correlation(self, buff):
        r1 = struct.unpack('B', buff[3])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[2])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[1])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[0])[0]
        # r1 = self.signed(r1)

        r2 = struct.unpack('B', buff[7])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[6])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[5])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[4])[0]
        # r2 = self.signed(r2)

        return r1, r2

    def convert_time_offset(self, buff):
        r = []
        r1 = struct.unpack('B', buff[1])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[0])[0]

        r2 = struct.unpack('B', buff[9])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[8])[0]

        r.append((r2 + r1 * 0.0001) * 1e-09)

        r1 = struct.unpack('B', buff[3])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[2])[0]

        r2 = struct.unpack('B', buff[11])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[10])[0]

        r.append((r2 + r1 * 0.0001) * 1e-09)

        r1 = struct.unpack('B', buff[5])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[4])[0]

        r2 = struct.unpack('B', buff[13])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[12])[0]

        r.append((r2 + r1 * 0.0001) * 1e-09)

        r1 = struct.unpack('B', buff[7])[0]
        r1 <<= 8
        r1 |= struct.unpack('B', buff[6])[0]

        r2 = struct.unpack('B', buff[15])[0]
        r2 <<= 8
        r2 |= struct.unpack('B', buff[14])[0]

        r.append((r2 + r1 * 0.0001) * 1e-09)

        return r

    def convert_time_offset_high(self, buff):
        r = []
        r0 = struct.unpack('B', buff[0])[0]
        r1 = struct.unpack('B', buff[1])[0]
        r2 = struct.unpack('B', buff[2])[0]
        r3 = struct.unpack('B', buff[3])[0]
        r4 = struct.unpack('B', buff[4])[0]
        r5 = struct.unpack('B', buff[5])[0]
        r6 = struct.unpack('B', buff[6])[0]
        r7 = struct.unpack('B', buff[7])[0]
        r8 = struct.unpack('B', buff[8])[0]
        r9 = struct.unpack('B', buff[9])[0]
        r10 = struct.unpack('B', buff[10])[0]
        r11 = struct.unpack('B', buff[11])[0]
        r12 = struct.unpack('B', buff[12])[0]
        r13 = struct.unpack('B', buff[13])[0]
        r14 = struct.unpack('B', buff[14])[0]
        r15 = struct.unpack('B', buff[15])[0]  # read 16*8bits: 4*15(integer part) + 4*17(decimal part)

        a1 = (r15 << 7) | (r14 >> 1)
        a2 = ((r14 & 1) << 14) | (r13 << 6) | ((r12 & 0xfc) >> 2)
        a3 = ((r12 & 0x03) << 13) | (r11 << 5) | ((r10 & 0xf8) >> 3)
        a4 = ((r10 & 0x07) << 12) | (r9 << 4) | ((r8 & 0xf0) >> 4)

        b1 = ((r8 & 0x0f) << 13) | (r7 << 5) | ((r6 & 0xf1) >> 3)
        b2 = ((r6 & 0x07) << 14) | (r5 << 6) | ((r4 & 0xfc) >> 2)
        b3 = ((r4 & 0x03) << 15) | (r3 << 7) | ((r2 & 0xfe) >> 1)
        b4 = ((r2 & 0x01) << 16) | (r1 << 8) | r0

        # print b1, b2, b3, b4

        r.append((a1 + b1 * 0.00001))  #ns
        r.append((a2 + b2 * 0.00001))  # * 1e-09)
        r.append((a3 + b3 * 0.00001))  # * 1e-09)
        r.append((a4 + b4 * 0.00001))  # * 1e-09)
        return r