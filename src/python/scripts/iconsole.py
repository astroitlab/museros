import sys

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport
from PyQt4.QtGui import QApplication

app = QApplication(sys.argv)

kernel_manager = QtInProcessKernelManager()
kernel_manager.start_kernel()
kernel = kernel_manager.kernel
kernel.gui = 'qt4'

kernel_client = kernel_manager.client()
kernel_client.start_channels()

def stop():
    kernel_client.stop_channels()
    kernel_manager.shutdown_kernel()
    # here you should exit your application with a suitable call
    sys.exit()

widget = RichIPythonWidget()
widget.kernel_manager = kernel_manager
widget.kernel_client = kernel_client
widget.exit_requested.connect(stop)
widget.setWindowTitle("IPython shell")

ipython_widget = widget
ipython_widget.show()

app.exec_()
sys.exit()