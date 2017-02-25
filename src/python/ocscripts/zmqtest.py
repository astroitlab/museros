#! /usr/bin/env python
# encoding: utf-8
"""
zmqtest.py
integrate an external service to an opencluster node
===================================================
"""
from __future__ import division
import sys, os, datetime, time, math
from random import randint
from opencluster.configuration import Conf
import zmq


def queryEphem(date,ephemServer) :
    st = time.time()
    REQUEST_TIMEOUT = 1000
    REQUEST_RETRIES = 2
    SERVER_ENDPOINT = "tcp://" + ephemServer

    context = zmq.Context(1)

    client = context.socket(zmq.REQ)
    client.connect(SERVER_ENDPOINT)

    poll = zmq.Poller()
    poll.register(client, zmq.POLLIN)

    retValue = None
    retries_left = REQUEST_RETRIES
    while retries_left and retValue is None:
        client.send(date)

        expect_reply = True
        while expect_reply:
            socks = dict(poll.poll(REQUEST_TIMEOUT))
            if socks.get(client) == zmq.POLLIN:
                reply = client.recv()
                if not reply:
                    break
                retValue = str(reply)
                break
            else:
                print "W: No response from server, retrying…"
                # Socket is confused. Close and remove it.
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                poll.unregister(client)
                retries_left -= 1
                if retries_left == 0:
                    raise Exception("E: ephem's Server seems to be offline, abandoning")
                print "I: Reconnecting and resending (%s)" % date
                # Create new connection
                client = context.socket(zmq.REQ)
                client.connect(SERVER_ENDPOINT)
                poll.register(client, zmq.POLLIN)
                client.send(date)

    #context.term()
    print "%.3f s for ephem query "%(time.time()-st)
    return retValue
#####################
##       MAIN      ##
#####################
def sendHeartbeat() :
    context = zmq.Context(1)

    worker = context.socket(zmq.DEALER) # DEALER
    identity = "127.0.0.1:8484:testService"
    worker.setsockopt(zmq.IDENTITY, identity)

    poller = zmq.Poller()
    poller.register(worker, zmq.POLLIN)
    worker.connect("tcp://localhost:30000")
    worker.send(Conf.PPP_READY)

    heartbeat_at = time.time() + Conf.HEARTBEAT_INTERVAL
    cycles = 0
    while True:
        socks = dict(poller.poll(Conf.HEARTBEAT_INTERVAL * 1000))

        # Handle worker activity on backend
        if socks.get(worker) == zmq.POLLIN:
            #  Get message
            #  - 3-part envelope + content -> request
            #  - 1-part HEARTBEAT -> heartbeat
            frames = worker.recv_multipart()
            if not frames:
                break # Interrupted

            if len(frames) == 3:
                # Simulate various problems, after a few cycles
                cycles += 1
                if cycles > 3 and randint(0, 5) == 0:
                    print "I: Simulating a crash"
                    break
                if cycles > 3 and randint(0, 5) == 0:
                    print "I: Simulating CPU overload"
                    time.sleep(3)
                print "I: Normal reply"
                worker.send_multipart(frames)
                liveness = Conf.HEARTBEAT_LIVENESS
                time.sleep(1)  # Do some heavy work
            elif len(frames) == 1 and frames[0] == Conf.PPP_HEARTBEAT:
                print "I: Server heartbeat"
                liveness = Conf.HEARTBEAT_LIVENESS
            else:
                print "E: Invalid message: %s" % frames
            interval = Conf.INTERVAL_INIT

        if time.time() > heartbeat_at:
            heartbeat_at = time.time() + Conf.HEARTBEAT_INTERVAL
            print "I: Service heartbeat"
            worker.send(Conf.PPP_HEARTBEAT)

if __name__ == "__main__" :
    # for i in range(1000000):
    #     time.sleep(0.5)
    #     print queryEphem(sys.argv[1],"172.31.254.30:5555")
    sendHeartbeat()