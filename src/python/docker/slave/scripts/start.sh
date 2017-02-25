#!/bin/sh

if [ -z "${MASTER_IP}" ]
then
    exit 1
fi
SLAVE_IP=$(ip addr | awk '/inet/ && /eth0/{sub(/\/.*$/,"",$2); print $2}')
mesos-slave --master=${MASTER_IP}:5050 --hostname=${SLAVE_IP} &
/usr/sbin/sshd
