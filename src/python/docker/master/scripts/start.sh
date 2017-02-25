#!/bin/sh

MASTER_IP=$(ip addr | awk '/inet/ && /eth0/{sub(/\/.*$/,"",$2); print $2}')
dnsmasq &
/opt/mesosphere/zookeeper/bin/zkServer.sh start
mesos-master --cluster=opencluster-mesos --work_dir=/var/lib/mesos --zk=zk://${MASTER_IP}:2181/mesos --log_dir=/var/logs/mesos --quorum=1 &
marathon &
/usr/sbin/sshd -D
python /astrodata/museros/python/sbin/startfactory.py --factory=${MASTER_IP}:6666 --master=zk://${MASTER_IP}:2181/mesos &
python /astrodata/museros/python/sbin/startfactoryweb.py --factory=${MASTER_IP}:6666 --master=zk://${MASTER_IP}:2181/mesos &
