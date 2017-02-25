#!/usr/bin/python

import sys
import os
import getopt

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget


#from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.qt.manager import QtKernelManager
from PyQt4 import QtGui,QtCore,QtTest
from src.python.musercli.musergui import mainwindow
import musercli

#sys.path.append('/root/csrhostest/csrhtest/python/musercli')
#from musercli import csrhclient
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt


app = QtGui.QApplication(sys.argv)
def kernelmanager():
    kernel_manager = QtKernelManager()
    kernel_manager.start_kernel()
    #kernel = kernel_manager.kernel
    #kernel.gui = 'qt4'
    return kernel_manager

def kernelclient(kernel_manager):
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    return kernel_client

def stop():
    kernel_client.stop_channels()
    kernel_manager.shutdown_kernel()
    # here you should exit your application with a suitable call
    sys.exit()

def shell(kernel_manager,kernel_client):
    widget = RichIPythonWidget()
    widget.kernel_manager = kernel_manager
    widget.kernel_client = kernel_client
    widget.exit_requested.connect(stop)
    widget.setWindowTitle("IPython shell")
    return widget

def splash():
    splash=QtGui.QSplashScreen()
    #splash.setPixmap(QtGui.QPixmap("/root/csrhostest/csrhtest/python/qt/2.jpg"))
    splash.show()
    splash.showMessage("setting up the main window........",QtCore.Qt.red)
    QtTest.QTest.qSleep(1000)
    splash.showMessage("loading modules........",QtCore.Qt.red)
    QtTest.QTest.qSleep(2000)
    #splash.finish()
    del splash



homedir = os.getenv('MUSEROS_HOME')
if homedir=='':
    print "No environment variable MUSEROS_HOME"
    os.exit(1)

#kernel_manager=kernelmanager()
#kernel_client=kernelclient(kernel_manager)
#splash()
kernel_manager=kernelmanager()
kernel_client=kernelclient(kernel_manager)
widget=shell(kernel_manager,kernel_client)
#widget.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
print "Home directory: ", homedir
print "Run Display Window"
sys.stdout.flush()

widget.execute("%matplotlib inline")
scriptfile = 'run '+os.path.split(os.path.realpath(__file__))[0]+'/museros.py --gui --server'

widget.execute(scriptfile)

print "Run terminal"
#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#X,Y,Z=axes3d.get_test_data(0.05)
#cset=ax.contour(X,Y,Z)
#ax.clabel(cset,fontsize=9,inline=1)

#plt.show()

ipython_widget=mainwindow()

ipython_widget.setCentralWidget(widget)
ipython_widget.show()


#sapp.exec_()
#sys.path.append('/root/csrhostest/csrhtest/python/musercli')
try:
    opts, args = getopt.getopt(sys.argv[1:], \
                               "vdnhfg:c:", \
                               ["vervose", "debug", "nopasswd", "help",
                                "file=", "conf","server","gui","client"])
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    musercli.usage()
    sys.exit(2)

print opts, args

musercli.main(opts, args)

sys.exit()
