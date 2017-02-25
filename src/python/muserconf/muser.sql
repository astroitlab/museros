-- --------------------------------------------------------
-- 主机:                           172.31.254.25
-- 服务器版本:                        5.1.73 - Source distribution
-- 服务器操作系统:                      redhat-linux-gnu
-- HeidiSQL 版本:                  9.3.0.4984
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

-- 导出  表 muser.p_antenna_delay 结构
CREATE TABLE IF NOT EXISTS `p_antenna_delay` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `refTime` datetime DEFAULT NULL,
  `freq` tinyint(4) DEFAULT '0',
  `theValue` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `refTime_freq` (`refTime`,`freq`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.p_antenna_delay 的数据：~5 rows (大约)
/*!40000 ALTER TABLE `p_antenna_delay` DISABLE KEYS */;
INSERT INTO `p_antenna_delay` (`id`, `refTime`, `freq`, `theValue`) VALUES
	(1, '2015-10-23 00:00:00', 1, '50,0,17,902,21,67,-13,459,3,32,-675,29,345,-65,-33,57,-9,61,-62,13,12,26,16,27,7,53,-5,63,342,37,30,313,655,12,-30,48,-16,14,-6,-1665,0,0,0,0'),
	(2, '2015-10-23 00:00:00', 2, '-17,0,7,-25,-21,474,22,-20,-16,-19,-1,29,143,0,-12,58,-64,-5,40,120,-22,-36,-17,78,386,292,353,4,149,-7,-23,-11,-22,-49,-33,142,-9,712,1880,111,-6,-31,-42,-12,233,29,-8,-81,-25,227,-4,190,-25,-14,-34,-56,6,-40,2,-27,0,1,0,0'),
	(3, '2016-04-06 23:59:59', 1, 'af'),
	(6, '2016-04-30 23:59:59', 1, '123.212 42 125.02 0.2312 3141 43 31431 1341 234.1 1.4 314.32 341.2 143.23'),
	(7, '2016-05-01 23:59:59', 1, '123 ');
/*!40000 ALTER TABLE `p_antenna_delay` ENABLE KEYS */;


-- 导出  表 muser.p_antenna_flag 结构
CREATE TABLE IF NOT EXISTS `p_antenna_flag` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `refTime` datetime DEFAULT NULL,
  `freq` tinyint(4) DEFAULT '0',
  `theValue` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `refTime_freq` (`refTime`,`freq`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.p_antenna_flag 的数据：~6 rows (大约)
/*!40000 ALTER TABLE `p_antenna_flag` DISABLE KEYS */;
INSERT INTO `p_antenna_flag` (`id`, `refTime`, `freq`, `theValue`) VALUES
	(1, '2015-11-01 00:00:00', 1, '8, 9,10, 11, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39'),
	(2, '2014-11-01 00:00:00', 1, '4, 7, 10, 11,12,13,16,17,18, 19, 24, 25, 26, 36, 38, 39'),
	(3, '2014-11-01 00:00:00', 2, '4, 7, 10, 11,12,13,16,17,18, 19, 24, 25, 26, 36, 38, 39'),
	(4, '2015-11-01 00:00:00', 2, '8, 9,10, 11, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39'),
	(5, '2016-04-05 23:59:59', 1, '12'),
	(6, '2016-05-01 23:59:59', 1, '2134');
/*!40000 ALTER TABLE `p_antenna_flag` ENABLE KEYS */;


-- 导出  表 muser.p_antenna_position 结构
CREATE TABLE IF NOT EXISTS `p_antenna_position` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `refTime` datetime DEFAULT NULL,
  `freq` tinyint(4) DEFAULT '0',
  `theValue` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `refTime_freq` (`refTime`,`freq`)
) ENGINE=InnoDB AUTO_INCREMENT=39 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.p_antenna_position 的数据：~16 rows (大约)
/*!40000 ALTER TABLE `p_antenna_position` DISABLE KEYS */;
INSERT INTO `p_antenna_position` (`id`, `refTime`, `freq`, `theValue`) VALUES
	(3, '2014-11-01 00:00:00', 1, '0.000,0.000,0.000,-0.022,7.988,0.000,-6.426,19.456,0.000,-21.025,28.661,0.000,-44.371,31.799,0.000,-79.806,-0.535,0.100,-92.243,-67.981,0.000,-49.823,-156.924,0.000,75.111,-226.202,0.000,283.605,-203.226,0.000,517.111,3.454,0.000,624.882,459.903,0.000,356.229,1120.404,5.500,-286.460,1817.530,10.000,-6.884,-4.112,0.000,-13.559,-15.394,0.000,-14.214,-32.499,0.000,-5.365,-54.364,0.000,40.334,-68.776,0.000,105.005,-45.994,0.000,160.745,35.210,0.000,158.353,178.005,0.000,34.311,347.220,0.000,-261.386,446.416,5.500,-710.941,311.107,10.000,-1148.753,-251.691,10.000,-1194.157,-1342.201,10.000,6.966,-3.990,0.000,20.100,-4.200,0.000,35.379,3.976,0.000,50.088,26.883,0.000,39.461,69.250,0.000,-12.702,113.930,0.000,-110.987,121.621,2.000,-233.431,47.856,2.000,-317.836,-143.928,2.000,-255.740,-449.563,2.000,86.093,-771.081,0.000,792.346,-868.716,10.000,1759.184,-362.106,-0.100'),
	(4, '2015-11-01 00:00:00', 1, '0.000,0.000,0.000,-0.022,7.988,0.000,-6.426,19.456,0.000,-21.025,28.661,0.000,-44.371,31.799,0.000,-79.806,-0.535,0.100,-92.243,-67.981,0.000,-49.823,-156.924,0.000,75.111,-226.202,0.000,283.605,-203.226,0.000,517.111,3.454,0.000,624.882,459.903,0.000,356.229,1120.404,5.500,-286.460,1817.530,10.000,-6.884,-4.112,0.000,-13.559,-15.394,0.000,-14.214,-32.499,0.000,-5.365,-54.364,0.000,40.334,-68.776,0.000,105.005,-45.994,0.000,160.745,35.210,0.000,158.353,178.005,0.000,34.311,347.220,0.000,-261.386,446.416,5.500,-710.941,311.107,10.000,-1148.753,-251.691,10.000,-1194.157,-1342.201,10.000,6.966,-3.990,0.000,20.100,-4.200,0.000,35.379,3.976,0.000,50.088,26.883,0.000,39.461,69.250,0.000,-12.702,113.930,0.000,-110.987,121.621,2.000,-233.431,47.856,2.000,-317.836,-143.928,2.000,-255.740,-449.563,2.000,86.093,-771.081,0.000,792.346,-868.716,10.000,1759.184,-362.106,-0.100'),
	(5, '2014-11-01 00:00:00', 2, '4.052,1.798,-0.002,8.088,9.444,0.002,5.843,20.412,0.004,-4.660,30.883,-0.001,-23.554,35.989,0.003,-48.180,30.299,0.004,-73.229,9.748,0.003,-90.606,-27.797,-0.010,-90.584,-80.753,-0.019,-62.380,-142.397,0.000,3.301,-199.676,-0.002,111.324,-232.669,-0.112,257.455,-214.313,-0.122,422.408,-114.359,-0.037,567.032,94.601,-0.232,628.851,425.688,-0.088,523.096,861.555,5.440,65.796,1398.847,10.030,-236.188,1814.218,9.994,-1.831,28.075,0.003,-3.626,2.556,0.001,-12.152,2.248,0.004,-20.571,-5.101,-0.007,-24.478,-19.521,-0.001,-19.370,-38.417,0.005,-2.245,-56.877,-0.005,28.211,-68.264,0.006,69.276,-64.495,0.006,115.201,-38.105,0.003,154.503,17.091,-0.012,171.306,102.749,0.005,145.956,212.634,0.008,57.228,330.170,0.023,-111.753,423.123,5.473,-365.237,443.832,5.472,-683.019,331.954,9.925,-1007.692,22.612,9.694,-1233.287,-532.519,9.929,-1203.442,-1314.658,9.866,-23.409,-15.635,-0.015,-0.455,-4.366,0.004,4.023,-11.688,0.008,14.689,-15.310,0.003,29.097,-11.404,-0.006,42.901,2.416,-0.003,49.788,22.208,0.003,44.973,58.494,-0.002,21.278,92.393,-0.006,-24.581,118.787,0.003,-92.185,125.235,1.986,-174.686,96.951,1.962,-257.143,19.786,1.983,-314.411,-115.657,1.968,-310.206,-308.568,1.965,-201.612,-538.394,1.778,54.277,-757.439,-0.357,484.575,-883.681,5.255,801.866,-871.843,9.753,1740.087,-383.930,0.332,25.217,-12.456,0.007'),
	(6, '2015-11-01 00:00:00', 2, '4.052,1.798,-0.002,8.088,9.444,0.002,5.843,20.412,0.004,-4.660,30.883,-0.001,-23.554,35.989,0.003,-48.180,30.299,0.004,-73.229,9.748,0.003,-90.606,-27.797,-0.010,-90.584,-80.753,-0.019,-62.380,-142.397,0.000,3.301,-199.676,-0.002,111.324,-232.669,-0.112,257.455,-214.313,-0.122,422.408,-114.359,-0.037,567.032,94.601,-0.232,628.851,425.688,-0.088,523.096,861.555,5.440,65.796,1398.847,10.030,-236.188,1814.218,9.994,-1.831,28.075,0.003,-3.626,2.556,0.001,-12.152,2.248,0.004,-20.571,-5.101,-0.007,-24.478,-19.521,-0.001,-19.370,-38.417,0.005,-2.245,-56.877,-0.005,28.211,-68.264,0.006,69.276,-64.495,0.006,115.201,-38.105,0.003,154.503,17.091,-0.012,171.306,102.749,0.005,145.956,212.634,0.008,57.228,330.170,0.023,-111.753,423.123,5.473,-365.237,443.832,5.472,-683.019,331.954,9.925,-1007.692,22.612,9.694,-1233.287,-532.519,9.929,-1203.442,-1314.658,9.866,-23.409,-15.635,-0.015,-0.455,-4.366,0.004,4.023,-11.688,0.008,14.689,-15.310,0.003,29.097,-11.404,-0.006,42.901,2.416,-0.003,49.788,22.208,0.003,44.973,58.494,-0.002,21.278,92.393,-0.006,-24.581,118.787,0.003,-92.185,125.235,1.986,-174.686,96.951,1.962,-257.143,19.786,1.983,-314.411,-115.657,1.968,-310.206,-308.568,1.965,-201.612,-538.394,1.778,54.277,-757.439,-0.357,484.575,-883.681,5.255,801.866,-871.843,9.753,1740.087,-383.930,0.332,25.217,-12.456,0.007'),
	(7, '2016-04-27 23:59:59', 1, '123.212'),
	(10, '2016-04-30 23:59:59', 1, '0.12 0.3'),
	(24, '2016-05-17 00:00:00', 1, 'd'),
	(25, '2016-04-18 00:00:00', 1, 'afd'),
	(26, '2016-05-01 23:59:59', 1, '23'),
	(29, '2016-03-18 00:00:00', 1, '12 23 34'),
	(30, '2016-04-09 00:00:00', 1, '23 12 32'),
	(31, '2015-09-27 00:00:00', 1, '23 23 56'),
	(32, '2016-03-04 00:00:00', 1, '12.21 23.21 2.21 23.1 45.21'),
	(33, '2016-02-14 00:00:00', 2, '12 23 89 97 79'),
	(34, '2015-12-06 00:00:00', 2, '12'),
	(38, '2015-12-22 00:00:00', 2, '12 23 34 56 78');
/*!40000 ALTER TABLE `p_antenna_position` ENABLE KEYS */;


-- 导出  表 muser.p_instrument_status 结构
CREATE TABLE IF NOT EXISTS `p_instrument_status` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `refTime` datetime DEFAULT NULL,
  `freq` tinyint(4) DEFAULT '0',
  `theValue` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `refTime_freq` (`refTime`,`freq`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.p_instrument_status 的数据：~2 rows (大约)
/*!40000 ALTER TABLE `p_instrument_status` DISABLE KEYS */;
INSERT INTO `p_instrument_status` (`id`, `refTime`, `freq`, `theValue`) VALUES
	(2, '2016-04-05 23:59:59', 1, 'running'),
	(3, '2016-05-01 23:59:59', 1, '12');
/*!40000 ALTER TABLE `p_instrument_status` ENABLE KEYS */;


-- 导出  表 muser.p_weather 结构
CREATE TABLE IF NOT EXISTS `p_weather` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `refTime` datetime DEFAULT NULL,
  `theValue` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `refTime` (`refTime`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.p_weather 的数据：~6 rows (大约)
/*!40000 ALTER TABLE `p_weather` DISABLE KEYS */;
INSERT INTO `p_weather` (`id`, `refTime`, `theValue`) VALUES
	(2, '2016-02-25 16:27:37', '{temperature:20,humidity:80,wind:\'\',rain:\'\'}'),
	(3, '2016-04-05 23:59:59', 'sum'),
	(6, '2016-05-01 23:59:59', 'cold'),
	(10, '2016-05-19 00:00:00', 'wind'),
	(11, '2016-05-31 00:00:00', 'wind'),
	(13, '2016-04-04 00:00:00', 'sum');
/*!40000 ALTER TABLE `p_weather` ENABLE KEYS */;


-- 导出  表 muser.t_calibration 结构
CREATE TABLE IF NOT EXISTS `t_calibration` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `ctime` date DEFAULT '1980-05-08',
  `priority` smallint(6) DEFAULT '1',
  `fileName` varchar(255) DEFAULT NULL,
  `offset` int(6) DEFAULT '100',
  `status` tinyint(4) DEFAULT '0',
  `freq` tinyint(4) DEFAULT '0',
  `theValue` mediumtext,
  `description` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_calibration 的数据：~8 rows (大约)
/*!40000 ALTER TABLE `t_calibration` DISABLE KEYS */;
INSERT INTO `t_calibration` (`id`, `ctime`, `priority`, `fileName`, `offset`, `status`, `freq`, `theValue`, `description`) VALUES
	(3, '2015-01-29', 0, '1', 100, 0, 0, '111111111111', NULL),
	(4, '2015-01-30', 0, '10', 10, 0, 1, NULL, NULL),
	(5, '2016-02-22', 0, NULL, 10, 0, 2, NULL, NULL),
	(6, '2016-04-06', 0, NULL, 10, 0, 1, NULL, NULL),
	(7, '2016-04-06', 0, NULL, 10, 0, 2, NULL, NULL),
	(8, '2016-04-08', 0, NULL, 10, 1, 1, '', NULL),
	(9, '2016-04-26', 0, NULL, 10, 1, 1, '', NULL),
	(10, '2016-06-28', 1, NULL, 10, 1, 1, '', NULL);
/*!40000 ALTER TABLE `t_calibration` ENABLE KEYS */;


-- 导出  表 muser.t_config 结构
CREATE TABLE IF NOT EXISTS `t_config` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `keyName` varchar(255) NOT NULL,
  `createTime` datetime NOT NULL,
  `theValue` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniKeyName` (`keyName`,`createTime`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_config 的数据：~5 rows (大约)
/*!40000 ALTER TABLE `t_config` DISABLE KEYS */;
INSERT INTO `t_config` (`id`, `keyName`, `createTime`, `theValue`) VALUES
	(1, 'altitude', '2011-12-01 00:00:00', '1365.0'),
	(2, 'longitude', '2011-12-01 00:00:00', '42.211833333'),
	(3, 'latitude', '2011-12-01 00:00:00', '115.2505'),
	(6, 'observatory', '2011-12-01 00:00:00', '115.2505 42.211833333 1365.0'),
	(7, 'altitude', '2016-05-04 01:01:01', '1');
/*!40000 ALTER TABLE `t_config` ENABLE KEYS */;


-- 导出  表 muser.t_config_key 结构
CREATE TABLE IF NOT EXISTS `t_config_key` (
  `keyName` varchar(255) NOT NULL,
  PRIMARY KEY (`keyName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_config_key 的数据：~4 rows (大约)
/*!40000 ALTER TABLE `t_config_key` DISABLE KEYS */;
INSERT INTO `t_config_key` (`keyName`) VALUES
	('altitude'),
	('latitude'),
	('longitude'),
	('observatory');
/*!40000 ALTER TABLE `t_config_key` ENABLE KEYS */;


-- 导出  表 muser.t_imaging 结构
CREATE TABLE IF NOT EXISTS `t_imaging` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `beginTime` datetime DEFAULT NULL,
  `endTime` datetime DEFAULT NULL,
  `status` tinyint(4) DEFAULT '0',
  `seconds` tinyint(4) DEFAULT '0' COMMENT '积分时间',
  `freq` tinyint(4) DEFAULT '0' COMMENT '1,2',
  `description` varchar(255) DEFAULT NULL,
  `job_id` varchar(50) DEFAULT NULL,
  `createTime` datetime DEFAULT NULL,
  `format` varchar(20) NOT NULL DEFAULT 'uvfits' COMMENT 'uvfits, fitsidid, votable',
  `is_specified_file` tinyint(4) DEFAULT '0',
  `specified_file` varchar(50) DEFAULT NULL COMMENT '指定处理的文件',
  `big_file` varchar(50) DEFAULT NULL COMMENT '处理后的大文件',
  `gen_result` varchar(10) DEFAULT 'all' COMMENT 'png,fits.all',
  `with_axis` tinyint(4) DEFAULT '0' COMMENT '0,1',
  `results` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `job_id` (`job_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8 COMMENT='成图任务表';

-- 正在导出表  muser.t_imaging 的数据：~1 rows (大约)
/*!40000 ALTER TABLE `t_imaging` DISABLE KEYS */;
INSERT INTO `t_imaging` (`id`, `beginTime`, `endTime`, `status`, `seconds`, `freq`, `description`, `job_id`, `createTime`, `format`, `is_specified_file`, `specified_file`, `big_file`, `gen_result`, `with_axis`, `results`) VALUES
	(1, '2015-11-01 10:00:00', '2015-11-02 11:00:00', 1, 1, 1, 're', 'Imaging-1', '2016-03-02 13:11:20', 'uvfits', 1, '/astrodata/archive/20151101/MUSER-1/20151101-1634', NULL, 'fits', 1, NULL);
/*!40000 ALTER TABLE `t_imaging` ENABLE KEYS */;


-- 导出  表 muser.t_integration 结构
CREATE TABLE IF NOT EXISTS `t_integration` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `beginTime` datetime DEFAULT NULL,
  `endTime` datetime DEFAULT NULL,
  `status` tinyint(4) DEFAULT '0' COMMENT '0:new,1:waiting,2,running,3:success，4:fail',
  `seconds` float DEFAULT '1' COMMENT '积分时间',
  `freq` tinyint(4) NOT NULL DEFAULT '1' COMMENT '1,2',
  `job_id` varchar(50) DEFAULT NULL,
  `createTime` datetime DEFAULT NULL,
  `format` varchar(20) NOT NULL DEFAULT 'uvfits' COMMENT 'uvfits, fitsidid, votable',
  `is_specified_file` tinyint(4) DEFAULT '0',
  `specified_file` varchar(50) DEFAULT NULL COMMENT '指定处理的文件',
  `big_file` varchar(50) DEFAULT NULL COMMENT '处理后的大文件',
  `description` varchar(255) DEFAULT NULL,
  `results` varchar(255) DEFAULT NULL,
  `task_num` int(11) DEFAULT '0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `job_id` (`job_id`)
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8 COMMENT='积分任务表';

-- 正在导出表  muser.t_integration 的数据：~11 rows (大约)
/*!40000 ALTER TABLE `t_integration` DISABLE KEYS */;
INSERT INTO `t_integration` (`id`, `beginTime`, `endTime`, `status`, `seconds`, `freq`, `job_id`, `createTime`, `format`, `is_specified_file`, `specified_file`, `big_file`, `description`, `results`, `task_num`) VALUES
	(1, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-1', '2016-03-02 09:24:46', '', 0, NULL, NULL, '', '/astrodata/work/temp/Integration-1.zip', 1),
	(2, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-2', '2016-03-02 09:24:48', 'fitsidi', 1, '/astrodata/archive/20151101/MUSER-1/20151101-1208', NULL, '', '/astrodata/work/temp/Integration-2.zip', 1),
	(3, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-3', '2016-03-02 09:24:43', 'uvfits', 0, '', NULL, '', '/astrodata/work/temp/Integration-3.zip', 1),
	(4, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-4', '2016-03-02 09:24:42', 'uvfits', 0, '', NULL, '', '/astrodata/work/temp/Integration-4.zip', 1),
	(5, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-5', '2016-03-02 09:24:36', 'uvfits', 0, NULL, NULL, NULL, '/astrodata/work/temp/Integration-5.zip', 1),
	(6, '2015-11-01 12:08:50', '2015-11-01 12:08:59', 3, 1, 1, 'Integration-6', '2016-03-02 09:24:47', 'uvfits', 0, NULL, NULL, NULL, NULL, 1),
	(9, '2015-11-01 12:08:50', '2015-11-01 12:08:52', 3, 1, 1, 'Integration-9', '2016-03-02 09:24:35', 'uvfits', 0, '', NULL, '', '/astrodata/work/temp/Integration-9.zip', 1),
	(14, '2015-11-01 12:08:50', '2015-11-01 12:08:53', 3, 1, 1, 'Integration-14', '2016-03-02 00:00:00', 'uvfits', 0, '', NULL, '', '', 1),
	(15, '2016-03-01 10:00:00', '2016-03-01 11:00:00', 4, 1, 1, 'Integration-15', '2016-03-02 00:00:00', 'uvfits', 0, '', NULL, '', 'cannot find observational data.', 1),
	(16, '2015-11-01 12:08:50', '2015-11-01 12:08:52', 3, 1, 1, 'Integration-16', '2016-03-10 00:00:00', 'uvfits', 1, '', NULL, '', '/astrodata/work/temp/Integration-16.zip', 1),
	(17, '2015-11-01 12:08:50', '2015-11-01 12:08:51', 2, 1, 1, 'Integration-17', '2016-04-06 15:53:36', 'uvfits', 1, '', NULL, '', 'cannot find observational data.', 1);
/*!40000 ALTER TABLE `t_integration` ENABLE KEYS */;


-- 导出  表 muser.t_integration_task 结构
CREATE TABLE IF NOT EXISTS `t_integration_task` (
  `task_id` varchar(50) NOT NULL,
  `timeStr` varchar(50) DEFAULT NULL,
  `status` tinyint(4) DEFAULT '0',
  `job_id` varchar(50) DEFAULT NULL,
  `results` text COMMENT '结果',
  `last_time` datetime DEFAULT NULL,
  `int_id` int(11) NOT NULL,
  `freq` int(11) NOT NULL DEFAULT '1',
  `int_number` int(11) NOT NULL,
  `repeat_num` int(11) NOT NULL DEFAULT '1',
  PRIMARY KEY (`task_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='积分子任务表';

-- 正在导出表  muser.t_integration_task 的数据：~52 rows (大约)
/*!40000 ALTER TABLE `t_integration_task` DISABLE KEYS */;
INSERT INTO `t_integration_task` (`task_id`, `timeStr`, `status`, `job_id`, `results`, `last_time`, `int_id`, `freq`, `int_number`, `repeat_num`) VALUES
	('Integration-1-20151101120850007259', '20151101120850.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-01 17:24:16', 1, 1, 40, 1),
	('Integration-1-20151101120851007259', '20151101120851.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-01 19:04:01', 1, 1, 40, 1),
	('Integration-1-20151101120852007259', '20151101120852.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_519745361I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_522870972I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_525995327I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_529120938I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_532245293I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_535370904I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_538495259I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-01 17:24:49', 1, 1, 40, 1),
	('Integration-1-20151101120853007259', '20151101120853.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_494786592I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_497910946I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_501036558I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_504160912I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_507286524I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_510410878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_513525803I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_516650786I.uvfits', '2016-03-01 16:05:40', 1, 1, 40, 1),
	('Integration-1-20151101120854007259', '20151101120854.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_494757279I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_497882891I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_501007245I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_504132857I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_507257211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_510382823I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_513497119I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_516622102I.uvfits', '2016-03-01 16:05:42', 1, 1, 40, 1),
	('Integration-1-20151101120855007259', '20151101120855.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_519718402I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_522843385I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_525968368I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_529093351I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_532218334I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_535343317I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_538468300I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_541593283I.uvfits', '2016-03-01 16:05:39', 1, 1, 40, 1),
	('Integration-1-20151101120856007259', '20151101120856.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_519741895I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_522866878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_525991861I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_529116844I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_532241827I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_535366810I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_538491793I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_541616776I.uvfits', '2016-03-01 16:05:42', 1, 1, 40, 1),
	('Integration-1-20151101120857007259', '20151101120857.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_494776211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_497900566I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_501026177I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_504150532I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_507276143I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_510400498I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_513515423I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_516640406I.uvfits', '2016-03-01 16:05:41', 1, 1, 40, 1),
	('Integration-1-20151101120858007259', '20151101120858.007259', 2, 'Integration-1', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_519742991I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_522868603I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_525992957I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_529118569I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_532242923I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_535368535I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_538492890I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_541618501I.uvfits', '2016-03-01 16:05:42', 1, 1, 40, 1),
	('Integration-14-20151101120850007259', '20151101120850.007259', 2, 'Integration-14', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-10 14:04:33', 14, 1, 40, 1),
	('Integration-14-20151101120851007259', '20151101120851.007259', 2, 'Integration-14', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-10 14:08:28', 14, 1, 40, 1),
	('Integration-14-20151101120852007259', '20151101120852.007259', 2, 'Integration-14', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-10 14:08:31', 14, 1, 40, 1),
	('Integration-16-20151101120850007259', '20151101120850.007259', 2, 'Integration-16', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-12 16:25:47', 16, 1, 40, 1),
	('Integration-16-20151101120851007259', '20151101120851.007259', 2, 'Integration-16', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-12 17:04:36', 16, 1, 40, 1),
	('Integration-2-20151101120850007259', '20151101120850.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120851007259', '20151101120851.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120852007259', '20151101120852.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_519745361I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_522870972I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_525995327I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_529120938I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_532245293I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_535370904I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_538495259I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120853007259', '20151101120853.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_494786592I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_497910946I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_501036558I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_504160912I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_507286524I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_510410878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_513525803I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_516650786I.uvfits', '2016-03-01 17:04:11', 2, 1, 40, 1),
	('Integration-2-20151101120854007259', '20151101120854.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_494757279I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_497882891I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_501007245I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_504132857I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_507257211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_510382823I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_513497119I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_516622102I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120855007259', '20151101120855.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_519718402I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_522843385I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_525968368I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_529093351I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_532218334I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_535343317I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_538468300I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_541593283I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120856007259', '20151101120856.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_519741895I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_522866878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_525991861I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_529116844I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_532241827I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_535366810I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_538491793I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_541616776I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120857007259', '20151101120857.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_494776211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_497900566I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_501026177I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_504150532I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_507276143I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_510400498I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_513515423I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_516640406I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-2-20151101120858007259', '20151101120858.007259', 2, 'Integration-2', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_519742991I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_522868603I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_525992957I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_529118569I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_532242923I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_535368535I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_538492890I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_541618501I.uvfits', '2016-03-01 17:05:58', 2, 1, 40, 1),
	('Integration-3-20151101120850007259', '20151101120850.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-01 17:08:01', 3, 1, 40, 1),
	('Integration-3-20151101120851007259', '20151101120851.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120852007259', '20151101120852.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_519745361I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_522870972I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_525995327I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_529120938I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_532245293I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_535370904I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_538495259I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120853007259', '20151101120853.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_494786592I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_497910946I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_501036558I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_504160912I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_507286524I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_510410878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_513525803I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_516650786I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120854007259', '20151101120854.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_494757279I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_497882891I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_501007245I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_504132857I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_507257211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_510382823I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_513497119I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_516622102I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120855007259', '20151101120855.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_519718402I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_522843385I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_525968368I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_529093351I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_532218334I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_535343317I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_538468300I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_541593283I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120856007259', '20151101120856.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_519741895I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_522866878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_525991861I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_529116844I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_532241827I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_535366810I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_538491793I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_541616776I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120857007259', '20151101120857.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_494776211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_497900566I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_501026177I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_504150532I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_507276143I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_510400498I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_513515423I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_516640406I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-3-20151101120858007259', '20151101120858.007259', 2, 'Integration-3', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_519742991I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_522868603I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_525992957I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_529118569I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_532242923I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_535368535I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_538492890I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_541618501I.uvfits', '2016-03-01 17:12:57', 3, 1, 40, 1),
	('Integration-4-20151101120850007259', '20151101120850.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-01 17:08:37', 4, 1, 40, 1),
	('Integration-4-20151101120851007259', '20151101120851.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-01 17:13:48', 4, 1, 40, 1),
	('Integration-4-20151101120852007259', '20151101120852.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_519745361I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_522870972I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_525995327I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_529120938I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_532245293I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_535370904I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_538495259I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120853007259', '20151101120853.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_494786592I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_497910946I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_501036558I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_504160912I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_507286524I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_510410878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_513525803I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_516650786I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120854007259', '20151101120854.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_494757279I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_497882891I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_501007245I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_504132857I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_507257211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_510382823I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_513497119I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_516622102I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120855007259', '20151101120855.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_519718402I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_522843385I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_525968368I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_529093351I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_532218334I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_535343317I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_538468300I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_541593283I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120856007259', '20151101120856.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_519741895I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_522866878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_525991861I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_529116844I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_532241827I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_535366810I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_538491793I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_541616776I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120857007259', '20151101120857.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_494776211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_497900566I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_501026177I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_504150532I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_507276143I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_510400498I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_513515423I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_516640406I.uvfits', '2016-03-01 17:12:57', 4, 1, 40, 1),
	('Integration-4-20151101120858007259', '20151101120858.007259', 2, 'Integration-4', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_519742991I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_522868603I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_525992957I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_529118569I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_532242923I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_535368535I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_538492890I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_541618501I.uvfits', '2016-03-01 17:13:48', 4, 1, 40, 1),
	('Integration-5-20151101120850007259', '20151101120850.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120851007259', '20151101120851.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120852007259', '20151101120852.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_519745361I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_522870972I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_525995327I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_529120938I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_532245293I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_535370904I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_538495259I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120852_541620871I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120853007259', '20151101120853.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_494786592I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_497910946I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_501036558I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_504160912I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_507286524I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_510410878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_513525803I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120853_516650786I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120854007259', '20151101120854.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_494757279I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_497882891I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_501007245I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_504132857I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_507257211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_510382823I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_513497119I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120854_516622102I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120855007259', '20151101120855.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_519718402I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_522843385I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_525968368I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_529093351I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_532218334I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_535343317I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_538468300I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120855_541593283I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120856007259', '20151101120856.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_519741895I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_522866878I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_525991861I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_529116844I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_532241827I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_535366810I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_538491793I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120856_541616776I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120857007259', '20151101120857.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_494776211I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_497900566I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_501026177I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_504150532I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_507276143I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_510400498I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_513515423I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120857_516640406I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-5-20151101120858007259', '20151101120858.007259', 2, 'Integration-5', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_519742991I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_522868603I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_525992957I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_529118569I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_532242923I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_535368535I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_538492890I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120858_541618501I.uvfits', '2016-03-01 17:13:48', 5, 1, 40, 1),
	('Integration-9-20151101120850007259', '20151101120850.007259', 2, 'Integration-9', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_494755087I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_497879441I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_501005053I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_504129407I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_507255019I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_510379374I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_513494298I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120850_516619281I.uvfits', '2016-03-12 17:04:58', 9, 1, 40, 1),
	('Integration-9-20151101120851007259', '20151101120851.007259', 2, 'Integration-9', '/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_519715581I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_522840564I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_525965547I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_529090530I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_532215513I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_535340496I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_538465479I.uvfits,/astrodata/archive/20151101/MUSER-1/uvfits/20151101-120851_541590462I.uvfits', '2016-03-12 17:04:58', 9, 1, 40, 1);
/*!40000 ALTER TABLE `t_integration_task` ENABLE KEYS */;


-- 导出  表 muser.t_int_queue 结构
CREATE TABLE IF NOT EXISTS `t_int_queue` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `stime` datetime DEFAULT NULL,
  `int_num` int(6) DEFAULT '1' COMMENT '多少个帧积分',
  `int_toal` int(6) DEFAULT '1' COMMENT '总共多少个积分帧',
  `fileName` varchar(255) DEFAULT NULL,
  `offset` int(6) DEFAULT '100',
  `integration_id` int(11) DEFAULT '0',
  `freq` tinyint(4) DEFAULT '0',
  `task_id` varchar(50) NOT NULL DEFAULT '0',
  `task_start_time` datetime DEFAULT NULL COMMENT '任务开始时间',
  `task_end_time` datetime DEFAULT NULL COMMENT '任务结束时间',
  `status` tinyint(4) DEFAULT '0' COMMENT '任务状态',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_int_queue 的数据：~0 rows (大约)
/*!40000 ALTER TABLE `t_int_queue` DISABLE KEYS */;
/*!40000 ALTER TABLE `t_int_queue` ENABLE KEYS */;


-- 导出  表 muser.t_job 结构
CREATE TABLE IF NOT EXISTS `t_job` (
  `job_id` varchar(50) NOT NULL DEFAULT '0',
  `status` tinyint(4) DEFAULT '0' COMMENT '任务状态',
  `mode` varchar(50) DEFAULT NULL,
  `create_time` datetime DEFAULT NULL,
  PRIMARY KEY (`job_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_job 的数据：~1 rows (大约)
/*!40000 ALTER TABLE `t_job` DISABLE KEYS */;
INSERT INTO `t_job` (`job_id`, `status`, `mode`, `create_time`) VALUES
	('MUSERRealTimeManager', 1, 'factory', '2016-02-26 21:10:32');
/*!40000 ALTER TABLE `t_job` ENABLE KEYS */;


-- 导出  表 muser.t_raw_file 结构
CREATE TABLE IF NOT EXISTS `t_raw_file` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `startTime` datetime NOT NULL,
  `freq` tinyint(4) NOT NULL DEFAULT '0' COMMENT '1,2',
  `path` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `rawFileStartTime` (`startTime`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_raw_file 的数据：~3 rows (大约)
/*!40000 ALTER TABLE `t_raw_file` DISABLE KEYS */;
INSERT INTO `t_raw_file` (`id`, `startTime`, `freq`, `path`) VALUES
	(2, '2014-05-12 13:14:53', 1, '/astrodata/CSRH_20140512-131453_178059171'),
	(3, '2015-11-01 12:08:00', 1, '/astrodata/archive/20151101/MUSER-1/20151101-1208'),
	(4, '2015-11-01 16:34:00', 1, '/astrodata/archive/20151101/MUSER-1/20151101-1634');
/*!40000 ALTER TABLE `t_raw_file` ENABLE KEYS */;


-- 导出  表 muser.t_task 结构
CREATE TABLE IF NOT EXISTS `t_task` (
  `task_id` varchar(50) NOT NULL,
  `task_desc` text NOT NULL,
  `task_start_time` datetime DEFAULT NULL COMMENT '任务开始时间',
  `task_end_time` datetime DEFAULT NULL COMMENT '任务结束时间',
  `status` tinyint(4) DEFAULT '1' COMMENT '任务状态',
  `priority` tinyint(4) DEFAULT '10' COMMENT '1-10',
  `job_id` varchar(50) DEFAULT '10',
  `result` text,
  PRIMARY KEY (`task_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 正在导出表  muser.t_task 的数据：~10 rows (大约)
/*!40000 ALTER TABLE `t_task` DISABLE KEYS */;
INSERT INTO `t_task` (`task_id`, `task_desc`, `task_start_time`, `task_end_time`, `status`, `priority`, `job_id`, `result`) VALUES
	('20151101120849.354161-20160228214447', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214447\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:58:36', '2016-02-28 21:58:38', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120849.354161-20160228214452', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214452\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:58:36', '2016-02-28 21:58:37', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120849.354161-20160228214522', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214522\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:58:57', '2016-02-28 21:58:57', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120849.354161-20160228214533', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214533\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:08', '2016-02-28 21:59:08', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120849.354161-20160228214548', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214548\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:20', '2016-02-28 21:59:20', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120849.354161-20160228214553', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x081\\x05s\\xa6\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120849.354161-20160228214553\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:35', '2016-02-28 21:59:36', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120849_357286240_0000qa†q.'),
	('20151101120854.354161-20160228214517', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x086\\x05s\\x8b\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120854.354161-20160228214517\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:58:57', '2016-02-28 21:58:57', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120854_357259880_0000qa†q.'),
	('20151101120854.354161-20160228214528', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x086\\x05s\\x8b\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120854.354161-20160228214528\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:08', '2016-02-28 21:59:08', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120854_357259880_0000qa†q.'),
	('20151101120854.354161-20160228214538', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x086\\x05s\\x8b\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120854.354161-20160228214538\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:19', '2016-02-28 21:59:19', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120854_357259880_0000qa†q.'),
	('20151101120854.354161-20160228214543', '(iopencluster.item\nTask\np1\n(dp2\nS\'workDir\'\np3\nS\'/astrodata/wsl/museros/ocscripts\'\np4\nsS\'tried\'\np5\nI0\nsS\'state_time\'\np6\nI0\nsS\'priority\'\np7\nI1\nsS\'workerClass\'\np8\nS\'realTimeWorker.RealTimeWorker\'\np9\nsS\'state\'\np10\nI6\nsS\'result\'\np11\nNsS\'warehouse\'\np12\nS\'OpenCluster1\'\np13\nsS\'data\'\np14\nccopy_reg\n_reconstructor\np15\n(copencluster.item\nObjValue\np16\nc__builtin__\ndict\np17\n(dp18\nS\'freq\'\np19\nI1\nsS\'firstFrameTime\'\np20\ncdatetime\ndatetime\np21\n(S\'\\x07\\xdf\\x0b\\x01\\x0c\\x086\\x05s\\x8b\'\ntRp22\nstRp23\nsS\'id\'\np24\nS\'20151101120854.354161-20160228214543\'\np25\nsS\'resources\'\np26\n(dp27\nS\'mem\'\np28\nI500\nsS\'gpus\'\np29\nI0\nsS\'cpus\'\np30\nI1\nssb.', '2016-02-28 21:59:19', '2016-02-28 21:59:19', 2, 1, 'MUSERRealTimeManager', '\\\0\0\0ðM€UTrueq]qXC\0\0\0/astrodata/work/temp/display/MUSER-1/20151101-120854_357259880_0000qa†q.');
/*!40000 ALTER TABLE `t_task` ENABLE KEYS */;


-- 导出  函数 muser.f_integration_check 结构
DELIMITER //
CREATE DEFINER=`muser`@`%` FUNCTION `f_integration_check`() RETURNS varchar(255) CHARSET latin1
BEGIN
declare notrun,running ,finished ,failed,totalRec int Default 0;
declare v_job_id varchar(50); 
declare job_ids varchar(255) Default ''; 
declare stop int default 0; 
declare cur cursor for select job_id from t_integration where status='2'; 
declare CONTINUE HANDLER FOR SQLSTATE '02000' SET stop=1; 
open cur; 
fetch cur into v_job_id;
while stop <> 1 do 
	
	select count(*) into totalRec from t_integration_task where job_id=v_job_id;
	
	select sum(case when status='0' then 1 else 0 end) into notrun from t_integration_task where job_id=v_job_id;
	
	select sum(case when status='1' then 1 else 0 end) into running from t_integration_task where job_id=v_job_id;
	
	select sum(case when status='2' then 1 else 0 end) into finished from t_integration_task where job_id=v_job_id;
	
	select sum(case when status='3' then 1 else 0 end) into failed from t_integration_task where job_id=v_job_id;
	
	if totalRec=finished then
		update t_integration set status='3' where job_id=v_job_id;
		set job_ids=concat(job_ids,v_job_id,',');
	end if;
	if totalRec=finished+failed and failed>0 then
		update t_integration set status='4' where job_id=v_job_id;
	end if;
	fetch cur into v_job_id;
end while; -- 循环结束 
close cur; -- 关闭游标

return job_ids;
END//
DELIMITER ;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
