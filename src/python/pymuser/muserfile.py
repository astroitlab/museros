from muserdata import *
from pyuvfits import *
from muserephem import *
from muserobs import muserobservatory

logger = logging.getLogger('muser')


class MuserFile(MuserData):
    def __init__(self, sub_array):
        """
        Main function call. Process raw data: delay process and sum
        """
        super(MuserFile, self).__init__(sub_array)

        # self.config = os.path.join(os.path.abspath(os.path.dirname(__file__)), "museruvfits.xml")
        self.config = os.path.join(self.env.get_home_dir() + '/data', "museruvfits.xml")
        self.source = ""
        self.onefile = False

        # muserRawData Class
        # UVFITS Output Class
        (self.longitude, self.latitude, self.altitude) = muserobservatory.get_muser_observatory()  # (115.2505, 42.211833333, 1365)

        self.pyuvfits = PyUVFITS(self.config, self.latitude, self.longitude, self.altitude, self)

    def set_parameters(self, s_time, int_time=0, repeat_number=0, hourangle= 999, declination =999, priority=0, nocalibration=1, inputfile='', debug=0,
                       genraw=1):
        self.integral_time = int_time
        self.repeat_number = repeat_number
        self.calibration_priority = priority
        self.no_calibration = nocalibration
        self.input_file_name = inputfile
        self.hourangle = hourangle
        self.declination = declination
        start_time = MuserTime()
        start_time.set_with_date_time(s_time)
        self.set_data_date_time(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute,
                                start_time.second, start_time.millisecond, start_time.microsecond,
                                start_time.nanosecond)
        self.debug = debug
        self.set_priority(priority)
        if genraw == 0:
            self.write_fits = True
        else:
            self.write_fits = False

    def set_priority(self, priority):
        self.calibration_priority = priority

    def write_integral_uvfits(self):
        generatedFileList = []

        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)

        # If cannot locate a proper frame, return with False
        if self.search_first_frame() == False:
            logger.error("Cannot find observational data.")
            return str(False), []
        if self.debug:
            logger.info("File opened: %s" % self.input_file_name)

        # Read one frame from the raw data file
        if self.is_loop_mode == False:
            full_frame_number = 1
        else:
            if self.sub_array == 1:
                full_frame_number = 8
            else:
                full_frame_number = 66

        integral_polar = np.zeros((full_frame_number), dtype=int)
        integral_band = np.zeros((full_frame_number), dtype=int)

        # Load Calibration file from disk according to the observational date and time
        if self.no_calibration == 1:
            # Load Calibration file from disk according to the observational date and time
            self.load_calibration_data()

        read_num = 0
        for iLoop in range(self.repeat_number):
            muser_baseline_data = np.zeros(
                (full_frame_number, self.antennas * (self.antennas - 1) / 2, self.sub_channels),
                dtype=complex)
            start_date_time = []
            end_date_time = []
            frequency = []

            for jLoop in range(self.integral_time):
                for num in range(full_frame_number):
                    if read_num > 0:
                        if self.check_next_file() == True:
                            # print "###### File Changled. ######"
                            # Change a file
                            if self.open_next_file(1) == False:
                                return False

                        self.read_one_frame()

                    self.read_data()
                    if self.debug:
                        logger.info("Reading frame: TIME: %s FREQ:%5d(MHz) POL: %-3s" % (
                        self.current_frame_date_time, self.frequency/10**6, "LL" if self.polarization == 0 else "RR"))

                    integral_band[num] = self.sub_band
                    integral_polar[num] = self.polarization

                    # Delay process and Strip stop
                    if self.current_frame_header.strip_switch == 0xCCCCCCCC:
                        if self.debug:
                            logger.info("Delay processing and Stripe rotation...")
                        self.delay_process('sun')
                    else:
                        if self.debug:
                            logger.info("Delay processing and Stripe rotation(Done by system)")

                    if self.no_calibration == 1:
                        # Calibration
                        self.calibration()

                    if jLoop == 0:
                        start_date_time.append(self.current_frame_time.get_julian_date())
                        frequency.append(self.frequency)

                    if jLoop == self.integral_time - 1:
                        end_date_time.append(self.current_frame_time.get_julian_date())
                    muser_baseline_data[num][:][:] += self.baseline_data[:][:]
                    read_num  = read_num + 1

                    if jLoop == self.integral_time - 1:
                        tm = (start_date_time[num] + end_date_time[num]) / 2.
                        self.current_frame_time.from_julian_date(start_date_time[num])
                        self.start_date_time_fits = self.current_frame_time
                        self.current_frame_time.from_julian_date(end_date_time[num])
                        self.end_date_time_fits = self.current_frame_time
                        self.current_frame_time.from_julian_date(tm)
                        self.current_frame_utc_time.copy(self.current_frame_time)
                        self.current_frame_utc_time.set_with_date_time(
                            self.current_frame_utc_time.get_date_time() + datetime.timedelta(hours=-8))

                        self.obs_date = ('%4d-%02d-%02d') % (
                            self.current_frame_utc_time.year, self.current_frame_utc_time.month, self.current_frame_utc_time.day)
                        self.obs_time = ('%02d:%02d:%02d.%03d%03d%03d') % (
                            self.current_frame_utc_time.hour, self.current_frame_utc_time.minute, self.current_frame_utc_time.second,
                            self.current_frame_utc_time.millisecond, self.current_frame_utc_time.microsecond,
                            self.current_frame_utc_time.nanosecond)

                        fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d' % (
                            self.current_frame_time.year, self.current_frame_time.month,
                            self.current_frame_time.day,
                            self.current_frame_time.hour, self.current_frame_time.minute,
                            self.current_frame_time.second, self.current_frame_time.millisecond,
                            self.current_frame_time.microsecond, self.current_frame_time.nanosecond)) + 'I.uvfits'

                        self.baseline_data[:][:] = muser_baseline_data[num][:][:]
                        self.baseline_data[:][:] = self.baseline_data[:][:] / self.integral_time

                        fits_file = self.env.uvfits_file(self.sub_array, fitsfile)
                        if self.write_fits == True:
                            self.pyuvfits.write_single_uvfits(fits_file, self.hourangle, self.declination)
                            if self.debug:
                                logger.info("UVFITS file saved: %d %d file=%s" % (integral_band[num], integral_polar[num], os.path.basename(fits_file)))
                        else:
                            self.write_single_real_time()
                            if self.debug:
                                logger.info("Numpy file saved: file=%s" % (fits_file))
                        generatedFileList.append(fits_file)

        self.close_file()
        return str(True), generatedFileList

    def write_single_uvfits(self):

        generatedFileList = []
        (freq, polar, ra, dec)=(0.,0.,0.,0.)
        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)

        if len(self.input_file_name.strip()) > 0:
            no_search = True
        else:
            no_search = False
        # If cannot locate a proper frame, return with False
        if self.search_first_frame(specified_file=no_search) == False:
            logger.error("cannot find observational data.")
            return str(False), []
        if self.debug:
            logger.info("File opened: %s\n" % self.input_file_name)

        if self.no_calibration == 1:
            # Load Calibration file from disk according to the observational date and time
            self.load_calibration_data()

        # Read one frame from the raw data file
        for iLoop in range(self.repeat_number):
            if iLoop > 0:
                if self.check_next_file() == True:
                    # print "###### File Changled. ######"
                    # Change a file
                    if self.open_next_file(1) == False:
                        return str(False), generatedFileList

                self.read_one_frame()
            # Read Observational Data
            self.read_data()

            if self.debug:
                logger.info("Reading frame: TIME: %s PREQ:%5d(MHz) POL: %-3s" % (
                self.current_frame_date_time, self.frequency/10**6, "LL" if self.polarization == 0 else "RR"))

            # Delay process and Strip stop

            if self.current_frame_header.strip_switch == 0xCCCCCCCC:
                if self.debug:
                    logger.info("Delay processing and Stripe rotation...")
                self.delay_process('sun')
            else:
                if self.debug:
                    logger.info("Delay processing and Stripe rotation(Done by system)")
            # Calibration
            if self.no_calibration == 1:
                self.calibration()
                if self.debug:
                    logger.info("Calibration...")

            self.start_date_time_fits = self.current_frame_time
            self.end_date_time_fits = self.current_frame_time

            if self.write_fits == True:

                fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d.uvfits') % (
                    self.current_frame_time.get_detail_time())
                fitsfile = self.env.uvfits_file(self.sub_array, fitsfile)
                self.pyuvfits.write_single_uvfits(fitsfile, self.hourangle, self.declination)
                if self.debug:
                    logger.info("File Generation...")
                    logger.info("UVFITS file saved: %s\n" % os.path.basename(fitsfile))
            else:
                fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d_%02d%02d' % (
                    self.current_frame_time.year, self.current_frame_time.month, self.current_frame_time.day,
                    self.current_frame_time.hour, self.current_frame_time.minute,
                    self.current_frame_time.second, self.current_frame_time.millisecond,
                    self.current_frame_time.microsecond, self.current_frame_time.nanosecond,
                    self.sub_band, self.polarization))
                dis_path = self.env.rt_display_file(self.sub_array)
                fitsfile = os.path.join(dis_path, fitsfile)
                (freq, polar, ra, dec) = self.write_single_real_time()
            generatedFileList.append((fitsfile, freq, polar, ra, dec))

        self.close_file()

        return str(True), generatedFileList

    #################################################################
    #    2015-12-21 Write the data in one minute to a FITS file. #
    #################################################################

    def write_one_big_uvfits(self):

        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)
        # Read one frame from the raw data file'''

        # If cannot locate a proper frame, return with False
        if self.search_first_frame() == False:
            logger.error("cannot find observational data.")
            return str(False), []

        if self.debug:
            logger.info("File opened: %s\n" % self.input_file_name)

        source_id = 0
        iLoop = 0
        fitsfile = ""
        if self.is_loop_mode == True:
            if self.repeat_number % 2 <> 0:
                self.repeat_number = self.repeat_number - 1
            self.pyuvfits.set_visibility_data(self.repeat_number // 2 )
        else:
            self.pyuvfits.set_visibility_data(self.repeat_number)

        if self.debug ==1:
            logger.info("Repeat_number: %d", self.repeat_number)

        previous_time = self.current_frame_time.get_date_time()

        while iLoop < self.repeat_number:

            self.read_data()  # Read Observational Data

            if self.debug:
                logger.info("\nReading frame: Time: %s Freq:%5d(MHz) POL: %-3s" % (
                self.current_frame_date_time, self.frequency/10**6, "LL" if self.polarization == 0 else "RR"))

            # Skip the first frame which is right polarization in loop mode
            # Delay process and Strip stop
            if self.current_frame_header.strip_switch == 0xCCCCCCCC:
                if self.debug:
                    logger.info("Delay processing and Stripe rotation...")
                self.delay_process('sun')
            else:
                if self.debug:
                    logger.info("Delay processing and Stripe rotation(Done by the system)")

            if self.no_calibration == 1:
                # Load Calibration file from disk according to the observational date and time
                self.load_calibration_data()
                # Calibration
                self.calibration()

            error = 0 # No ERROR
            if_append = True
            t_offset = self.current_frame_time.get_date_time() - previous_time
            time_offset = t_offset.seconds * 1e6 + t_offset.microseconds

            if self.is_loop_mode == True and iLoop != 0:
                # Exception
                # If frame skipped, we have to skip some data
                if (iLoop % 2 == 0 and self.polarization == 1):
                    #  Condition 1: We need LL, no RR
                    error = 1
                    iLoop = iLoop + 1

                    if self.debug:
                        logger.info("Error with condition 1: iLoop % 2 == 0 and self.polarization == 1")

                elif (iLoop % 2 == 1 and self.polarization == 0):
                    #  Condition 2: We have LL, no RR
                    error = 2
                    iLoop = iLoop + 1

                    if self.debug:
                        logger.info("Error with condition 2: iLoop % 2 == 1 and self.polarization == 0")

                elif time_offset > 8000 and self.polarization == 1:
                    # Condition 3: We have LL and RR, but not in the same band (Skipped more than two frames)
                    error = 3
                    iLoop = iLoop + 2

                    if self.debug:
                        logger.info("Error with condition 3: skipped at least two frames, self.polarization == 1")

            if iLoop >= self.repeat_number:
                if_append = False

            if self.is_loop_mode == True:
                source_id = iLoop // 2 + 1
            else:
                source_id = iLoop + 1

            if iLoop == 0:
                fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03dB.uvfits') % (
                    self.current_frame_time.get_detail_time())
                fitsfile = self.env.uvfits_file(self.sub_array, fitsfile)

            self.start_date_time_fits = self.current_frame_time
            self.end_date_time_fits = self.current_frame_time

            if self.is_loop_mode == False or (self.is_loop_mode==True and iLoop%2==0) or error!=0:
                self.uvws_sum, self.source = self.compute_UVW(self.obs_date, self.obs_time)  # units: SECONDS
                self.obs_date_sum = self.source.midnightJD
                self.obs_time_sum = self.source.JD - self.source.midnightJD
                self.ra_sum = self.source.appra
                self.dec_sum = self.source.appdec

            self.pyuvfits.config_primary_big(source_id, error, if_append)

            if self.is_loop_mode == True and iLoop <= self.repeat_number:
                if (error == 0 and iLoop%2 == 0) or error!= 0:
                    self.pyuvfits.config_source_big(self.source, source_id)
            else:
                self.pyuvfits.config_source_big(self.source, source_id)

            previous_time = self.current_frame_date_time
            self.read_one_frame()
            iLoop = iLoop + 1

        self.pyuvfits.config_frequency_big()
        hdu = self.pyuvfits.make_primary_big()
        tbl_frequency = self.pyuvfits.make_frequency_big()
        tbl_antenna = self.pyuvfits.make_antenna(num_rows=self.antennas)
        tbl_antenna = self.pyuvfits.config_antenna( tbl_antenna)
        tbl_source = self.pyuvfits.make_source_big()

        hdulist = pf.HDUList(
            [hdu,
             tbl_frequency,
             tbl_antenna,
             tbl_source,
             ])
        hdulist.verify()  # Verify all values in the instance. Output verification option.

        if (os.path.isfile(fitsfile)):
            os.remove(fitsfile)
        if self.debug:
            logger.info('Write big UVFITS file: %s' % os.path.basename(fitsfile))
        hdulist.writeto(fitsfile)

    def write_single_real_time(self):

        Sun = self.pyuvfits.makeSource(name=self.obs_target)
        self.source = Sun

        obs = Observatory(lon=self.longitude, lat=self.latitude, altitude=self.altitude)

        array_geometry = self.pyuvfits.ant_array()
        antenna_array = Array(lat=self.latitude, long=self.longitude, elev=self.altitude, antennas=array_geometry)

        self.source.midnightJD, midnightMJD = self.pyuvfits.ephem.convert_date(self.obs_date, '00:00:00')
        # We should compute the target's position firstly
        self.source.compute(cobs=obs, cdate=self.obs_date, ctime=self.obs_time)

        uvws = []
        self.baseline = []
        bl_len = int(self.antennas * (self.antennas - 1) / 2)
        (bl_order, baselines) = self.pyuvfits.config_baseline_ID(self.bl_len)
        for baseline in baselines:
            vector = baseline[1]
            self.baseline.append(baseline[0])
            if self.hourangle==999 and self.declination ==999:
                H, d = (self.source.gast - self.source.appra, self.source.appdec)
            else:
                H, d = self.hourangle, self.declination

            uvws.append(self.pyuvfits.computeUVW(vector, H * 15., d))

        uvws = np.array(uvws)
        # print uvws
        self.uvws_sum = uvws.reshape(bl_len, 3) / light_speed  # units: SECONDS
        self.obs_date_sum = self.source.midnightJD
        self.obs_time_sum = self.source.JD - self.source.midnightJD
        # self.muser.Data_sum = self.muser.baseline_data
        # print "JD, MJD: ", source.JD, source.midnightJD
        self.ra_sum = self.source.appra
        self.dec_sum = self.source.appdec

        fitsfile = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d_%02d%02d' % (
            self.current_frame_time.year, self.current_frame_time.month, self.current_frame_time.day,
            self.current_frame_time.hour, self.current_frame_time.minute,
            self.current_frame_time.second, self.current_frame_time.millisecond,
            self.current_frame_time.microsecond, self.current_frame_time.nanosecond,
            self.sub_band, self.polarization))

        dis_path = self.env.rt_display_file(self.sub_array)

        fileCrossCorrelation = os.path.join(dis_path, fitsfile + "_cross.dat")
        fileAutoCorrelation = os.path.join(dis_path, fitsfile + "_auto.dat")
        fileUV = os.path.join(dis_path, fitsfile + "_uv.dat")

        self.baseline_data.tofile(fileCrossCorrelation)
        self.auto_correlation_data.tofile(fileAutoCorrelation)
        self.uvws_sum.tofile(fileUV)
        if self.debug:
            logger.info("Numpy format data saved.")
        return self.frequency, self.polarization, self.ra_sum, self.dec_sum


    def compute_UVW(self, obs_date, obs_time):
        Sun = self.pyuvfits.makeSource(name=self.obs_target)
        source = Sun

        obs = Observatory(lon=self.longitude, lat=self.latitude, altitude=self.altitude)

        array_geometry = self.pyuvfits.ant_array()
        antenna_array = Array(lat=self.latitude, long=self.longitude, elev=self.altitude, antennas=array_geometry)

        source.midnightJD, midnightMJD = self.pyuvfits.ephem.convert_date(obs_date, '00:00:00')
        # We should compute the target's position firstly
        source.compute(cobs=obs, cdate=obs_date, ctime=obs_time)

        uvws = []
        self.baseline = []
        bl_len = int(self.antennas * (self.antennas - 1) / 2)
        (bl_order, baselines) = self.pyuvfits.config_baseline_ID(bl_len)
        for baseline in baselines:
            vector = baseline[1]
            self.baseline.append(baseline[0])
            H, d = (source.gast - source.appra, source.appdec)
            uvws.append(self.pyuvfits.computeUVW(vector, H * 15., d))

        uvws = np.array(uvws)
        uvws_sum = uvws.reshape(bl_len, 3) / light_speed  # units: SECONDS

        return uvws_sum, source

    def get_data_info(self, start_time, end_time, integral_period, realtime=False):

        self.set_data_date_time(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute,
                                start_time.second, start_time.millisecond, start_time.microsecond,
                                start_time.nanosecond)
        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)

        # If cannot locate a proper frame, return with False
        if realtime:
            if self.search_frame_realtime() == False:
                logger.error("cannot find observational data.")
                return False, None, None, None
        else:
            if self.search_first_frame() == False:
                logger.error("cannot find observational data.")
                return False, None, None, None

        frame = [2500, 20625]  # 25,206.25 -> 2500, 20625, so integraltime * 1e8 instead of 1e6

        if self.is_loop_mode == False:
            integral_number = int(integral_period * 1e6 // 3125)  # 3.125 -> 3125
        else:
            if ((integral_period * 1e8) % frame[self.sub_array - 1]) == 0:
                integral_number = integral_period * 1e5 // frame[self.sub_array - 1]
            else:
                integral_number = int(integral_period * 1e5 // frame[self.sub_array - 1]) + 1

        if ((start_time.get_date_time() - end_time.get_date_time()).seconds * 1E5) % (
                    integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1])) == 0:
            total_loop_number = int(abs((end_time.get_date_time() - start_time.get_date_time()).seconds * 1E5) / (
                integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1])))
        else:
            total_loop_number = int(
                abs((end_time.get_date_time() - start_time.get_date_time()).seconds * 1E5) / (
                    integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1]))) + 1

        return self.current_frame_time.get_date_time(), integral_number, total_loop_number, self.is_loop_mode


    def get_frame_info(self, start_time, end_time):

        self.set_data_date_time(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute,
                                start_time.second, start_time.millisecond, start_time.microsecond,
                                start_time.nanosecond)
        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)

        # If cannot locate a proper frame, return with False
        if self.search_first_frame() == False:
            logger.error("cannot find observational data.")
            return False, None, None, None

        frame = [2500, 20625]  # 25,206.25 -> 2500, 20625, so integraltime * 1e8 instead of 1e6
        integral_number = 1

        if ((start_time.get_date_time() - end_time.get_date_time()).seconds * 1E5) % (
                    integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1])) == 0:
            total_loop_number = int(abs((end_time.get_date_time() - start_time.get_date_time()).seconds * 1E5) / (
                integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1])))
        else:
            total_loop_number = int(
                abs((end_time.get_date_time() - start_time.get_date_time()).seconds * 1E5) / (
                    integral_number * (312.5 if self.is_loop_mode == False else frame[self.sub_array - 1]))) + 1

        return total_loop_number



    def get_file_info(self, start_time, end_time): # 2015-11-01 11:34:00  2015-11-01 11:36:00

        file_info=[]
        self.set_data_date_time(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute,
                                start_time.second, start_time.millisecond, start_time.microsecond, start_time.nanosecond)

        if self.search_first_file() == False:
            print "Cannot find observational data."
            return

        if self.open_raw_file(self.current_file_name) == False:
            print "Cannot open observational data."
            return

        if self.read_one_frame() == False:
            print "Error reading frame."
            return

        file_info.append(self.current_file_name)

        while True:

            if self.open_next_file(1) == False:
                break
            self.read_one_frame()
            if self.current_frame_time.get_time_stamp() > end_time.get_time_stamp():
                break
            file_info.append(self.current_file_name)

        return file_info

    def get_visdata(self, filename, offset, repeat_number):

        self.in_file = open(filename, 'rb')
        self.in_file.seek(offset, 0)

        vis_file= ""
        uvw_file= ""
        date_file = ""
        uvws = []
        visdata = []
        date_FILE = np.ndarray(shape = (repeat_number, 7), dtype= float)

        for iLoop in range(0, repeat_number):

            # Read Observational Data
            self.read_one_frame()
            self.read_data()
            if self.debug:
                logger.info("Time: %s BAND:%5d POL: %-3s" % (
                self.current_frame_date_time, self.channel_group, "LL" if self.polarization == 1 else "RR"))

            # Skip the first frame which is right polarization in loop mode

            # Delay process and Strip stop
            if self.current_frame_header.strip_switch == 0xCCCCCCCC:
                if self.debug:
                    logger.debug("Strip rotation correction")
                self.delay_process('sun')

            if self.no_calibration == 1:
                # Load Calibration file from disk according to the observational date and time
                self.load_calibration_data()
                # Calibration
                self.calibration()

            visdata.append(self.baseline_data)

            if iLoop == 0:
                file_VIS = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d.vis') % (   #FILENAME VIS UVW
                    self.current_frame_time.get_detail_time())
                vis_file = self.env.vis_file(self.sub_array, file_VIS)

                file_UVW = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d.uvw') % (
                    self.current_frame_time.get_detail_time())
                uvw_file = self.env.uvw_file(self.sub_array, file_UVW)

                file_DATE = ('%4d%02d%02d-%02d%02d%02d_%03d%03d%03d.date') % (
                    self.current_frame_time.get_detail_time())
                date_file = self.env.uvw_file(self.sub_array, file_DATE)

            Sun = self.pyuvfits.makeSource(name=self.obs_target)
            self.source = Sun

            obs = Observatory(lon=self.longitude, lat=self.latitude, altitude=self.altitude)

            array_geometry = self.pyuvfits.ant_array()
            antenna_array = Array(lat=self.latitude, long=self.longitude, elev=self.altitude, antennas=array_geometry)
            self.source.midnightJD, midnightMJD = self.pyuvfits.ephem.convert_date(self.obs_date, '00:00:00')
            self.source.compute(cobs=obs, cdate=self.obs_date, ctime=self.obs_time)


            self.baseline = []
            bl_len = int(self.antennas * (self.antennas - 1) / 2)
            (bl_order, baselines) = self.pyuvfits.config_baseline_ID(bl_len)

            for baseline in baselines:
                vector = baseline[1]
                self.baseline.append(baseline[0])
                if self.hourangle==999 and self.declination ==999:
                    H, d = (self.source.gast - self.source.appra, self.source.appdec)
                else:
                    H, d = self.hourangle, self.declination

            if self.is_loop_mode == True:
                if iLoop %2 == 0:
                    uvws.append(self.pyuvfits.computeUVW(vector, H * 15., d) / light_speed )  # units: SECOND
                    self.obs_date_sum = self.source.midnightJD
                    self.obs_time_sum = self.source.JD - self.source.midnightJD
                    date_FILE[iLoop][0] = self.obs_date_sum
                    date_FILE[iLoop][1] = self.obs_time_sum
                    date_FILE[iLoop][2] = self.source.appra
                    date_FILE[iLoop][3] = self.source.appdec
                    date_FILE[iLoop][4] = self.source.topora
                    date_FILE[iLoop][5] = self.source.topodec
                    date_FILE[iLoop][6] = self.frequency

            else:
                uvws.append(self.pyuvfits.computeUVW(vector, H * 15., d) / light_speed )  # units: SECOND
                self.obs_date_sum = self.source.midnightJD
                self.obs_time_sum = self.source.JD - self.source.midnightJD
                date_FILE[iLoop][0] = self.obs_date_sum
                date_FILE[iLoop][1] = self.obs_time_sum

        UVW= np.array(uvws)
        VIS_DATA = np.array(visdata)
        UVW.tofile(vis_file)
        VIS_DATA.tofile(uvw_file)
        date_FILE.tofile(date_file)

        print "Writing to file: ", os.path.basename(vis_file),os.path.basename(uvw_file),os.path.basename(date_file)
        return [vis_file, uvw_file, date_file]

    def merge_one_big_uvfits(self, file_name, vis_file, uvw_file, date_file, repeatnumber):

        (self.t_len, self.chan_len, self.bl_len, self.pol_len, self.ri_len) = (
            1, self.sub_channels, self.antennas * (self.antennas - 1) / 2, 1, 2)
        vis_file_len = len(vis_file) # number of files need to read, and it's also task number
        UVW_DATA = []
        VIS_DATA = []
        DATE_DATA = []

        for len in range(0, vis_file_len):
            self.load_vis_data(vis_file[len])
            self.load_uvw_data(uvw_file[len])
            self.load_date_file(date_file[len])

            UVW_DATA.extend(self.uvw_data)
            VIS_DATA.extend(self.vis_data)
            DATE_DATA.extend(self.date)

        self.uvwdata = np.array(UVW_DATA).reshape(vis_file_len*repeatnumber*self.antennas * (self.antennas - 1) // 2, 3)

        if self.sub_array == 1:
            self.vis = np.array(VIS_DATA).reshape(vis_file_len*repeatnumber*self.antennas * (self.antennas - 1) // 2, 16)
        elif self.sub_array == 2:
            self.vis = np.array(VIS_DATA).reshape(vis_file_len*repeatnumber*self.antennas * (self.antennas - 1) // 2, 33)

        DATE = np.array(DATE_DATA).reshape(vis_file_len*repeatnumber*self.antennas * (self.antennas - 1) // 2, 7)
        self.appra = np.ndarray(shape=(repeatnumber),dtype=float)
        self.appdec = np.ndarray(shape=(repeatnumber),dtype=float)
        self.topora = np.ndarray(shape=(repeatnumber),dtype=float)
        self.topodec = np.ndarray(shape=(repeatnumber),dtype=float)
        self.date1 = DATE[:,0]
        self.date2 = DATE[:,1]
        self.appra[:] = DATE[:,2]
        self.appdec[:] = DATE[:,3]
        self.topora[:] = DATE[:,4]
        self.topodec[:] = DATE[:,5]
        self.frequency[:] = DATE[:,6]

        self.baseline = []
        self.baseline_merge = []
        bl_len = int(self.antennas * (self.antennas - 1) / 2)
        (bl_order, baselines) = self.pyuvfits.config_baseline_ID(self.bl_len)
        for baseline in baselines:
            vector = baseline[1]
            self.baseline.append(baseline[0])

        for i in range(0, repeatnumber):
            self.baseline_merge.extend(self.baseline)

        fitsfile = file_name
        if self.is_loop_mode == True:
            if self.repeat_number % 2 <> 0:
                self.repeat_number = self.repeat_number - 1
            self.pyuvfits.set_visibility_data(vis_file_len*self.repeat_number // 2)
        else:
            self.pyuvfits.set_visibility_data(vis_file_len*self.repeat_number)

        self.pyuvfits.config_merge_primary_big(vis_file_len, repeatnumber)
        self.pyuvfits.config_merger_source_big(repeatnumber)
        self.pyuvfits.config_frequency_big()

        hdu = self.pyuvfits.make_primary_big()
        tbl_frequency = self.pyuvfits.make_frequency_big()

        tbl_antenna = self.pyuvfits.make_antenna(num_rows=self.antennas)
        tbl_antenna = self.pyuvfits.config_antenna( tbl_antenna)
        tbl_source = self.pyuvfits.make_source_big()

        hdulist = pf.HDUList(
            [hdu,
             tbl_frequency,
             tbl_antenna,
             tbl_source,
             ])

        # hdulist.info()
        hdulist.verify()  # Verify all values in the instance. Output verification option.
        if (os.path.isfile(fitsfile)):
            os.remove(fitsfile)
        if self.debug:
            logger.info('Write big UVFITS file - %s' % os.path.basename(fitsfile))

        hdulist.writeto(fitsfile)
        return fitsfile





