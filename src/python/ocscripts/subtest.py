import time

import zmq
import numpy as np
try:
    import cPickle
    pickle = cPickle
except:
    cPickle = None
    import pickle
def main():

    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.connect("tcp://172.31.254.24:5005")
    s.setsockopt(zmq.SUBSCRIBE,'')
    poller = zmq.Poller()


    s2 = ctx.socket(zmq.SUB)
    s2.connect("tcp://172.31.254.20:5006")
    s2.setsockopt(zmq.SUBSCRIBE,'')

    poller.register(s, zmq.POLLIN)
    poller.register(s2, zmq.POLLIN)


    print "Receiving arrays..."

    def receiveData(socket) :
        head = socket.recv()
        if len(head) < 10 :
            if head and head.find("str_") > -1 :
                data = socket.recv_string()
                print "received str : " + data
            else:
                data = socket.recv_pyobj()
                print "received binary : " + data

            del data
            # if "vis" == str(head) :
            #     for antenna1 in range(0, 40-1):  #SubChannelsLow = 16
            #         for antenna2 in range(antenna1 + 1, 40):
            #             for channel in range(0, 16):
            #                 print "csrhData["+str(antenna1)+"]["+str(antenna2)+"]["+str(channel)+"]=" + str(data[antenna1][antenna2][channel].real)+","+str(data[antenna1][antenna2][channel].imag)
            if "png" == str(head).split("_")[0] :
                for x in data :
                    print x
        del head
    while True :
        socks = dict(poller.poll(1000))
        if socks.get(s) == zmq.POLLIN:
            print "s1-------------"
            receiveData(s)

        if socks.get(s2) == zmq.POLLIN:
            print "s2-------------"
            receiveData(s2)

    print "   Done."


if __name__ == "__main__":
    main()