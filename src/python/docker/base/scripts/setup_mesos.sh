#!/bin/bash

cd /work
rpm -Uvh http://muser.cnlab.net/package/mesos/mesosphere-el-repo-7-1.noarch.rpm
yum -y install mesos

