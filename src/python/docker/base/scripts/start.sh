#!/bin/sh

if [ -z "${APP}" ]
then
    echo "APP must be one of [MuserConfWeb,OCWeb,RealTime,Integration,Factory,FactoryMesos]"
    exit 1
fi


case .${APP} in
    .MuserConfWeb)
        python /opt/work/museros/python/muserconf/browser.py
	;;
    .OCWeb)
        python /opt/work/museros/python/sbin/startfactoryweb.py
	;;
    .RealTime)
        cd /opt/work/museros/python
        python ./ocscripts/realTimeManager.py -m mesos -l 15 -v
	;;
    .Integration)
        python /opt/work/museros/python/ocscripts/integrationTask.py -m factory
	;;
    .Factory)
        python /opt/work/museros/python/sbin/startfactory.py
	;;
    .FactoryMesos)
        python /opt/work/museros/python/sbin/startfactorymesos.py
	;;
esac

