import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.signal import savgol_filter
import psutil
import warnings
import time as tm
warnings.filterwarnings("error")
# from PyQt5 import QtWidgets
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
# class MplCanvas(Canvas):
#     def __init__(self):
#         self.fig = Figure()
#         self.ax = self.fig.add_axes([0.12, 0.12, 0.83, 0.83], frameon=False)
#         self.ax.axes.get_xaxis().set_visible(False)
#         self.ax.axes.get_yaxis().set_visible(False)
#         Canvas.__init__(self, self.fig)
#         Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
#         Canvas.updateGeometry(self)
#
# # Matplotlib widget
# class MplWidget(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
#         self.canvas = MplCanvas()                  # Create canvas object
#         self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
#         self.vbl.addWidget(self.canvas)
#         self.setLayout(self.vbl)


def readDefaultNames():
    f = open('calibrationData\defaultFiles.txt', 'r')
    expName = f.readline().split(":")[1]
    expName = expName.split("\n")[0]
    theoName = f.readline().split(":")[1]
    f.close()
    return expName, theoName


def LoadFile(filename=None, field=None):
    if filename is None:
        Tk().withdraw()
        filename = askopenfilename()
    file = np.loadtxt(filename)
    if field:
        field.setText(filename)
    return file


def CalculateFWHM(wavelength, intensity):

    c = 3e2
    freq = 2*np.pi*c/wavelength
    dw = (freq[0]-freq[1])
    newFreq = np.arange(-10*freq[0], 10*freq[0], dw)
    wSpec = np.interp(newFreq, np.flip(freq), np.flip(intensity*(wavelength**2 /(2*np.pi*c))), left=1e-30, right=1e-30)
    pulse = np.fft.fftshift(np.fft.ifft(np.sqrt(np.abs(wSpec))))
    pulse = np.abs(pulse) ** 2 / np.max(np.abs(pulse) ** 2)

    try:
        time = np.fft.fftshift(np.fft.fftfreq(len(pulse), dw/(2*np.pi)))

        izqPulse = np.abs(pulse[0:int(len(pulse)/2)]-0.5)
        izqTime = time[0:int(len(pulse)/2)]
        derPulse = np.abs(pulse[int(len(pulse)/2):len(pulse)]-0.5)
        derTime = time[int(len(pulse)/2):]


        minIzqPulse = np.where(izqPulse == np.min(izqPulse))[0]
        minDerPulse = np.where(derPulse == np.min(derPulse))[0]

        return derTime[minDerPulse] - izqTime[minIzqPulse]

    except RuntimeWarning:
        return 'Error'


class CalibratorApp(QMainWindow):

    def __init__(self, UI):
        QMainWindow.__init__(self)
        loadUi(UI, self)
        self.show()
        self.specButton.clicked.connect(self.loadSpectrum)
        self.smoothSpecButton.clicked.connect(self.SmoothSpec)
        self.exportButton.clicked.connect(self.ExportSpec)

        if psutil.cpu_count() < 4:
            self.smoothButton.clicked.connect(self.SmoothSignalCalibration_LowResources)
        else:
            self.smoothButton.clicked.connect(self.SmoothSignalCalibration)

        self.filterCheckButton.toggled.connect(self.ToggleFilter)
        self.centerText.editingFinished.connect(self.UpdatePlot)
        self.widthText.editingFinished.connect(self.UpdatePlot)
        self.exponentText.editingFinished.connect(self.UpdatePlot)

        expName, theoName = readDefaultNames()

        try:
            self.expText.setText('calibrationData/' + expName)
            self.specCal = LoadFile(filename=self.expText.text())
            self.theoCal = LoadFile(filename=self.theoText.text())
        except:
            pass

        self.finalSpec = []
        self.expSpec = []
        self.expSpec_mod = []
        self.interpTheoCal = []
        self.R = []
        self.multiplyR = []
        self.calibratedSpec = []
        self.filter = []
        self.FWHM = []
        try:
            self.CalculateResponse()
        except:
            pass

    def loadSpectrum(self):
        try:
            spectrum = LoadFile(field=self.specText)
            self.expSpec = spectrum
            self.expSpec_mod = spectrum
            self.finalSpec = spectrum
            self.specFrame.setFrameStyle(QFrame.NoFrame)
            self.UpdatePlot()
            CalculateFWHM(spectrum[:, 0], spectrum[:, 1])
            self.smoothSpecButton.setEnabled(True)
            self.filterCheckButton.setEnabled(True)
            self.exportButton.setEnabled(True)

        except:
            self.specText.setText("Wrong file format")
            pass

    def CalculateResponse(self):
        self.interpTheoCal = np.interp(self.specCal[:, 0], self.theoCal[:, 0], self.theoCal[:, 1])
        self.R = np.divide(np.abs(self.specCal[:, 1]),
                           np.abs(self.interpTheoCal),
                           where=self.interpTheoCal != 0)
        self.R = self.R / np.max(self.R)
        self.multiplyR = np.divide(1, self.R, where=self.R != 0)
        self.UpdateCalPlot()

    def SuperGaussianFilter(self):
        try:
            self.filterCounter = 1
            center = float(self.centerText.text())
            width = float(self.widthText.text())
            exponent = int(self.exponentText.text())
            self.filter = np.exp(-((self.expSpec[:, 0] - center) / width) ** exponent)
        except:
            pass

    def SmoothSpec(self):
        try:
            self.finalSpec[:, 1] = savgol_filter(self.finalSpec[:, 1], 21, 3)
            self.UpdatePlot()
        except:
            pass

    def ToggleFilter(self):
        if self.filterCheckButton.isChecked():
            self.exponentText.setEnabled(True)
            self.centerText.setEnabled(True)
            self.widthText.setEnabled(True)
        else:
            self.exponentText.setEnabled(False)
            self.centerText.setEnabled(False)
            self.widthText.setEnabled(False)

        self.UpdatePlot()
    def SmoothSignalCalibration(self):
        try:
            self.R = savgol_filter(self.R, 21, 3)
            self.multiplyR = np.divide(1, self.R, where=self.R != 0)
        except:
            pass
        self.UpdateCalPlot()

    def SmoothSignalCalibration_LowResources(self):
        boxPoints = 10
        box = np.ones(boxPoints) / boxPoints
        self.R = np.convolve(self.R, box, mode='same')
        self.multiplyR = np.divide(1, self.R, where=self.R != 0)
        self.UpdateCalPlot()

    def ExportSpec(self):
        try:
            specNew, wavelength, norm = self.UpdatePlot()
            tm.sleep(0.5)
            spec = np.stack((wavelength, specNew*norm)).T
            np.savetxt(self.specText.text().split('.')[0]+'_calibrated.txt', spec,'%.2f')
            self.exportLabel.setText('Saved!')
        except:
            self.exportLabel.setText('Try again')

    def UpdatePlot(self):
        self.exportLabel.setText(' ')
        try:
            plotSpec = self.finalSpec[:, 1] * self.multiplyR
        except:
            plotSpec = np.abs(self.finalSpec[:, 1]) / np.max(self.finalSpec[:, 1])
        self.specPlot.canvas.ax.clear()
        self.specPlot.canvas.ax.plot(self.expSpec[:, 0], np.abs(self.expSpec[:, 1]) / np.max(self.expSpec[:, 1]))
        if self.filterCheckButton.isChecked():
            self.SuperGaussianFilter()
            plotSpec = self.filter*(plotSpec/np.max(plotSpec))
            self.specPlot.canvas.ax.plot(self.finalSpec[:, 0], (plotSpec/np.max(plotSpec)))
            self.specPlot.canvas.ax.plot(self.finalSpec[:, 0], self.filter)
        else:
            self.specPlot.canvas.ax.plot(self.finalSpec[:, 0], (plotSpec/ np.max(plotSpec)))
        self.specPlot.canvas.ax.axes.get_xaxis().set_visible(True)
        self.specPlot.canvas.ax.axes.get_yaxis().set_visible(True)
        self.specPlot.canvas.ax.set_frame_on(True)
        self.specPlot.canvas.ax.axes.set_xlabel('Wavelength (nm)')
        self.specPlot.canvas.ax.axes.set_ylabel('Intensity (a.u.)')
        self.specPlot.canvas.draw()
        self.FWHM = CalculateFWHM(self.finalSpec[:, 0], plotSpec)
        self.fwhmText.setText(str(np.round(self.FWHM[0], 2)))

        return plotSpec, np.reshape(self.expSpec[:, 0], (len(plotSpec),)), np.max(self.expSpec[:, 1])

    def UpdateCalPlot(self):
        self.calPlot.canvas.ax.clear()
        self.calPlot.canvas.ax.plot(self.specCal[:, 0], self.R / np.max(self.R))
        self.calPlot.canvas.ax.axes.get_xaxis().set_visible(True)
        self.calPlot.canvas.ax.axes.get_yaxis().set_visible(True)
        self.calPlot.canvas.ax.set_frame_on(True)
        self.calPlot.canvas.ax.axes.set_xlabel('Wavelength (nm)')
        self.calPlot.canvas.ax.axes.set_ylabel('Intensity (a.u.)')
        self.calFrame.setFrameStyle(QFrame.NoFrame)
        self.calPlot.canvas.draw()


app = QApplication([])

if app.desktop().screen().height() < 720:
    window = CalibratorApp(UI='FL_ui_mini.ui')
    window.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
    window.setWindowIcon(QtGui.QIcon("images/program_icon_square128.ico"))
    window.setWindowTitle("Spectrum calibrator")
    window.showMaximized()
else:
    window = CalibratorApp(UI='FL_ui.ui')
    window.setFixedSize(window.size())
    window.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
    window.setWindowIcon(QtGui.QIcon("images/program_icon_square128.ico"))
    window.setWindowTitle("Spectrum calibrator")
    window.show()

app.exec_()
