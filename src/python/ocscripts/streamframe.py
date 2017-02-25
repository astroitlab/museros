import errors
import struct
import time
import os
import logging
import cStringIO
from opencluster.configuration import Conf
logger = logging.getLogger("streamframe")

class StreamFrame(object):
    headerSize = 4
    realTimeFileRootDir = "/opt/work/realtime/"
    #realTimeFileRootDir = "F:/temp/realtime/"
    frameLen = {"low":100000,"high":204800}

    fullFrameSize = {"low":16,"high":132} #16 frames  8,33 frames  66
    validSize = {"low":8,"high":66}
    validFlag = {"low":0x3333,"high":0x0000}
    validFrameLen = {"low":8*100000,"high":66*204800}
    freqOffset = 190


    def __init__(self,frameLen) :
        self.dataSize = frameLen
        self.data = b""
        self.frames = []

    def __repr__(self) :
        return "<%s.%s at %x, dataSize=%d>" % (self.__module__, self.__class__.__name__, id(self), self.dataSize)

    def loadData(self,dataBytes):
        self.data = dataBytes
        if self.dataSize != len(self.data) :
            raise errors.CRSHError("header data and data's length mismatch")

    @classmethod
    def from_header(cls, headerData):
        if not headerData or len(headerData)!=cls.headerSize:
            raise errors.ProtocolError("header data size mismatch")
        frameLen = struct.unpack("i", headerData)
        frame = StreamFrame(frameLen[0])
        return frame

    @classmethod
    def recvSegment(cls, connection):
        frame = cls.from_header(connection.recv(cls.headerSize))
        frame.loadData(connection.recv(frame.dataSize))
        return frame

    @classmethod
    def recv(cls, connection, lowOrHigh):
        #frame = cls.from_header(connection.recv(cls.headerSize))
        frame = StreamFrame(cls.frameLen[lowOrHigh])
        i = 0;
        while len(frame.frames) < cls.validSize[lowOrHigh] and i < cls.fullFrameSize[lowOrHigh]:
            frame.loadData(connection.recv(frame.dataSize))
            binFrame = cStringIO.StringIO(frame.data)
            binFrame.seek(190)
            subband = struct.unpack('H', binFrame.read(2))[0]
            subband = subband & 0xffff

            logger.info(str(i) + " frame[190-191]" + "%x"%subband)
            i = i + 1
            if len(frame.frames) < 2:
                if subband == cls.validFlag[lowOrHigh]:
                    frame.frames.append(frame.data)
                    continue
            if len(frame.frames) >= 2 :
                frame.frames.append(frame.data)

        if len(frame.frames) != cls.validSize[lowOrHigh]:
            del frame.frames[:]
            return None
        today = time.localtime(time.time())
        binFilePath = os.path.join(cls.realTimeFileRootDir,lowOrHigh,time.strftime("%Y",today),time.strftime("%m",today),time.strftime("%d",today))
        if not os.path.exists(binFilePath) :
            os.makedirs(binFilePath)
        binFileName = os.path.join(binFilePath,time.strftime("%Y%m%d%H%M%S",today)+".frames")
        binFile = open(binFileName,'wb')
        for f in frame.frames :
            binFile.write(f)
        binFile.close()

        return binFileName