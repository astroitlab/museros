from socket import *

def SocketServer():
    try:
        Colon = ServerUrl.find(':')
        IP = ServerUrl[0:Colon]
        Port = int(ServerUrl[Colon+1:])

        print 'Server start:%s'%ServerUrl
        sockobj = socket(AF_INET, SOCK_STREAM)
        sockobj.setsockopt(SOL_SOCKET,SO_REUSEADDR, 1)

        sockobj.bind((IP, Port))
        sockobj.listen(5)

        while True:
            connection, address = sockobj.accept( )
            print 'Server connected by client:', address
            while True:
                data = connection.recv(1024)
                if not data:
                    break
                print 'Receive MSG:%s'%data.strip()
            connection.close( )

    except Exception,ex:
        print ex

if __name__ == "__main__":
    ServerUrl = "0.0.0.0:2650"
    SocketServer()
