import time
import zmq
import logging

REQUEST_TIMEOUT = 1500  # msecs
SETTLE_DELAY = 1000  # before failing over
REQUEST_NUM = 2

logger = logging.getLogger(__name__)


class FusionClient(object) :
    server = ['tcp://172.31.254.23:5001', 'tcp://172.31.254.23:5002']

    def __init__(self):
        self.server_nbr = 0
        self.ctx = zmq.Context()
        self.client = self.ctx.socket(zmq.REQ)
        self.client.connect(FusionClient.server[self.server_nbr])
        self.poller = zmq.Poller()
        self.poller.register(self.client, zmq.POLLIN)

    def sendAutoCorrelationValue(self,data):
        try :
            self.client.send_string("acv",zmq.SNDMORE)
            self.client.send_pyobj(data)
            requestNum = 1
            expect_reply = True

            while requestNum <= REQUEST_NUM and expect_reply:
                socks = dict(self.poller.poll(REQUEST_TIMEOUT))
                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv_string()
                    if reply == "ok":
                        logger.info("I: server replied OK")
                        expect_reply = False
                    else:
                        logger.error("E: malformed reply from server: %s" % str(reply))
                elif requestNum < REQUEST_NUM and expect_reply:
                    logger.error("W: no response from server, failing over")
                    time.sleep(SETTLE_DELAY / 1000)
                    self.poller.unregister(self.client)
                    self.client.close()
                    self.server_nbr = (self.server_nbr + 1) % 2
                    logger.error("I: connecting to server at %s.." % FusionClient.server[self.server_nbr])
                    self.client = self.ctx.socket(zmq.REQ)
                    self.poller.register(self.client, zmq.POLLIN)
                    # reconnect and resend request
                    self.client.connect(FusionClient.server[self.server_nbr])
                    self.client.send_string("acv",zmq.SNDMORE)
                    self.client.send_pyobj(data)
                requestNum += 1

        except Exception,e :
            logger.error(e)

    def sendVisibility(self,data):
        try :
            self.client.send_string("vis",zmq.SNDMORE)
            self.client.send_pyobj(data)
            requestNum = 1
            expect_reply = True

            while requestNum <= REQUEST_NUM and expect_reply:
                socks = dict(self.poller.poll(REQUEST_TIMEOUT))
                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv_string()
                    if reply == "ok":
                        logger.info("I: server replied OK")
                        expect_reply = False
                    else:
                        logger.error("E: malformed reply from server: %s" % str(reply))
                elif requestNum < REQUEST_NUM and expect_reply:
                    logger.error("W: no response from server, failing over")
                    time.sleep(SETTLE_DELAY / 1000)
                    self.poller.unregister(self.client)
                    self.client.close()
                    self.server_nbr = (self.server_nbr + 1) % 2
                    logger.error("I: connecting to server at %s.." % FusionClient.server[self.server_nbr])
                    self.client = self.ctx.socket(zmq.REQ)
                    self.poller.register(self.client, zmq.POLLIN)
                    # reconnect and resend request
                    self.client.connect(FusionClient.server[self.server_nbr])
                    self.client.send_string("vis",zmq.SNDMORE)
                    self.client.send_pyobj(data)
                requestNum += 1

        except Exception,e :
            logger.error(e)
    def sendFileName(self,fileName):
        try :
            self.client.send_string("file",zmq.SNDMORE)
            self.client.send_string(fileName)
            expect_reply = True
            requestNum = 1
            while requestNum <= REQUEST_NUM and expect_reply:
                socks = dict(self.poller.poll(REQUEST_TIMEOUT))
                if socks.get(self.client) == zmq.POLLIN:
                    reply = self.client.recv_string()
                    if reply == "ok":
                        logger.info("I: server replied OK")
                        expect_reply = False
                    else:
                        logger.error("E: malformed reply from server: %s" % str(reply))
                elif requestNum < REQUEST_NUM and expect_reply:
                    print "W: no response from server, failing over"
                    time.sleep(SETTLE_DELAY / 1000)
                    self.poller.unregister(self.client)
                    self.client.close()
                    self.server_nbr = (self.server_nbr + 1) % 2
                    logger.error("I: connecting to server at %s.." % FusionClient.server[self.server_nbr])
                    self.client = self.ctx.socket(zmq.REQ)
                    self.poller.register(self.client, zmq.POLLIN)
                    # reconnect and resend request
                    self.client.connect(FusionClient.server[self.server_nbr])
                    self.client.send_string("file",zmq.SNDMORE)
                    self.client.send_string(fileName)
                requestNum += 1
        except Exception,e :
            logger.error(e)
    def __del__(self):
        self.client.close()

if __name__ == "__main__" :
    d = FusionClient()
    d.sendFileName("dddddddddddd")