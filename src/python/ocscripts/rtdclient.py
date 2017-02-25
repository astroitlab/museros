import time
import zmq
import logging

REQUEST_TIMEOUT = 1500  # msecs
SETTLE_DELAY = 1000  # before failing over
REQUEST_NUM = 2

logger = logging.getLogger(__name__)
from conf import MUSERConf

class RealTimeSender(object) :
    server = MUSERConf.getRtServer().split(",")

    def __init__(self):
        self.server_nbr = 0
        self.ctx = zmq.Context()
        self.need_connect = True

    def __connect__(self):
        self.client = self.ctx.socket(zmq.REQ)
        self.client.connect(RealTimeSender.server[self.server_nbr])
        self.poller = zmq.Poller()
        self.poller.register(self.client, zmq.POLLIN)
        self.need_connect = False

    def sendData(self, prefix, data):
        requestNum = 0
        expect_reply = True

        if self.need_connect :
            self.__connect__()

        while expect_reply:
            self.client.send_string(prefix,zmq.SNDMORE)
            self.client.send_pyobj(data)
            while requestNum <= REQUEST_NUM :
                socks = dict(self.poller.poll(REQUEST_TIMEOUT))

                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv_string()
                    if reply == "ok":
                        logger.info("I: server replied OK")
                        expect_reply = False
                        break
                    else:
                        logger.error("E: malformed reply from server: %s" % str(reply))
                        break
                    del reply

                elif requestNum < REQUEST_NUM:
                    logger.error("W: no response from server, failing over")
                    time.sleep(SETTLE_DELAY / 1000)

                    self.server_nbr = (self.server_nbr + 1) % len(RealTimeSender.server)
                    logger.error("I: connecting to server at %s.." % RealTimeSender.server[self.server_nbr])

                    # reconnect and resend request
                    self.poller.unregister(self.client)
                    self.client = None

                    self.__connect__()
                    self.client.send_string(prefix,zmq.SNDMORE)
                    self.client.send_pyobj(data)
                    requestNum += 1
                else:
                    logger.error("W: servers failing over,data sending canceled.")
                    expect_reply = False
                    self.need_connect = True
                    break



    def sendString(self,prefix, str):
        requestNum = 0
        expect_reply = True

        if self.need_connect :
            self.__connect__()

        while expect_reply:
            self.client.send_string("str_" + prefix,zmq.SNDMORE)
            self.client.send_string(str)
            while requestNum <= REQUEST_NUM :
                socks = dict(self.poller.poll(REQUEST_TIMEOUT))

                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv_string()
                    if reply == "ok":
                        logger.info("I: server replied OK")
                        expect_reply = False
                        break
                    else:
                        logger.error("E: malformed reply from server: %s" % str(reply))
                        break
                    del reply

                elif requestNum < REQUEST_NUM:
                    logger.error("W: no response from server, failing over")
                    time.sleep(SETTLE_DELAY / 1000)

                    self.server_nbr = (self.server_nbr + 1) % len(RealTimeSender.server)
                    logger.error("I: connecting to server at %s.." % RealTimeSender.server[self.server_nbr])

                    # reconnect and resend request
                    self.poller.unregister(self.client)
                    self.client = None

                    self.__connect__()
                    self.client.send_string("str_" + prefix,zmq.SNDMORE)
                    self.client.send_string(str)
                    requestNum += 1
                else:
                    logger.error("W: servers failing over,data sending canceled.")
                    expect_reply = False
                    self.need_connect = True
                    break


    def __del__(self):
        try:
            self.client.close()
            del self.ctx
        except Exception,e :
            pass

if __name__ == "__main__" :
    d = RealTimeSender()
