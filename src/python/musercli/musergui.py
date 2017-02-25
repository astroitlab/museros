import sys

from PyQt4 import QtCore, QtGui

from ui_csrh import Ui_CSRH


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menu_File = QtGui.QMenu(self.menubar)
        self.menu_File.setObjectName(_fromUtf8("menu_File"))
        self.menu_Option = QtGui.QMenu(self.menubar)
        self.menu_Option.setObjectName(_fromUtf8("menu_Option"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_New = QtGui.QAction(MainWindow)
        self.action_New.setObjectName(_fromUtf8("action_New"))
        self.action_Open = QtGui.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/resource/open.jpg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Open.setIcon(icon)
        self.action_Open.setObjectName(_fromUtf8("action_Open"))
        self.action_Sun = QtGui.QAction(MainWindow)
        self.action_Sun.setObjectName(_fromUtf8("action_Sun"))
        self.menu_File.addAction(self.action_New)
        self.menu_File.addAction(self.action_Open)
        self.menu_Option.addAction(self.action_Sun)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menu_Option.menuAction())
        self.toolBar.addAction(self.action_Open)

        self.retranslateUi(MainWindow)
        QtCore.QObject.connect(self.action_Open, QtCore.SIGNAL(_fromUtf8("triggered()")), MainWindow.open)
        QtCore.QObject.connect(self.action_Sun, QtCore.SIGNAL(_fromUtf8("triggered()")), MainWindow.sun)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.menu_File.setTitle(_translate("MainWindow", "&File", None))
        self.menu_Option.setTitle(_translate("MainWindow", "&Option", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.action_New.setText(_translate("MainWindow", "&New", None))
        self.action_Open.setText(_translate("MainWindow", "&Open", None))
        self.action_Sun.setText(_translate("MainWindow", "&Sun", None))


class mainwindow(QtGui.QMainWindow):
    """"""

    #----------------------------------------------------------------------
    def __init__(self,parent=None):
        """Constructor"""
        QtGui.QMainWindow.__init__(self,parent)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
    #----------------------------------------------------------------------
    def open(self):
        """"""
    #----------------------------------------------------------------------
    def sun(self):
        """"""
        s=CSRHGUI(self)
        s.show()
    #----------------------------------------------------------------------
    def loadFile(self,fileName):
        """"""
        File=QtCore.QFile(fileName)
        if File.open(QtCore.QFile.ReadWrite)!=None:
            QtGui.QMessageBox.warning(self,"OpenFile","Cannot write file %1:\n%2.")
            return
        ins=QtCore.QTextStream(id(File))


def main():
    a=QtGui.QApplication(sys.argv)
    mya=mainwindow()
    mya.show()
    sys.exit(a.exec_())


########################################################################
class CSRHGUI(QtGui.QMainWindow):
    """"""

    #----------------------------------------------------------------------
    def __init__(self,parent=None):
        """Constructor"""
        QtGui.QMainWindow.__init__(self,parent)
        self.ui=Ui_CSRH()
        self.ui.setupUi(self)        
    #----------------------------------------------------------------------
    def Valid(self):
        """"""
        if len(self.ui.hEdit.text())==0 or len(self.ui.mEdit.text())==0 or len(self.ui.sEdit.text())==0:
            QtGui.QMessageBox.warning(self,"Warning","can't be empty!")
            i=0
        elif int(self.ui.hEdit.text())>=0 and int(self.ui.hEdit.text())<=23 and int(self.ui.mEdit.text())>=0 and int(self.ui.mEdit.text())<=59 and int(self.ui.sEdit.text())>=0 and int(self.ui.sEdit.text())<60:
            i=1  
        else:
            QtGui.QMessageBox.warning(self, "Warning","Time input is not valid!")
            i=0
            print 1      
    #----------------------------------------------------------------------
    def caculate(self):
        """"""
        self.Valid()
        
    #----------------------------------------------------------------------
    def main(self):
        """"""
        self.show()
                
            
#----------------------------------------------------------------------
def main():
    """"""
    app=QtGui.QApplication(sys.argv)
    myapp=CSRHGUI()
    myapp.show()
    sys.exit(app.exec_())
    
    
"""if __name__=="__main__":
    app=QtGui.QApplication(sys.argv)
    myapp=Sun()
    myapp.show()
    sys.exit(app.exec_())"""
        
        
    
    
