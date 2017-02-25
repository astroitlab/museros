from opencluster:base
MAINTAINER Shoulin Wei "wsl@cnlab.net"

#mesos
RUN systemctl disable mesos-slave && systemctl disable mesos-master

#RUN rpm -Uvh http://repos.mesosphere.io/el/7/noarch/RPMS/mesosphere-el-repo-7-1.noarch.rpm
#RUN yum install -y --enablerepo=mesosphere marathon mesosphere-zookeeper

ADD http://muser.cnlab.net/package/mesosphere-zookeeper-3.4.6-0.1.20141204175332.centos7.x86_64.rpm /soft/mesosphere-zookeeper-3.4.6-0.1.20141204175332.centos7.x86_64.rpm
RUN cd /soft && yum -y install mesosphere-zookeeper-3.4.6-0.1.20141204175332.centos7.x86_64.rpm && rm -f /soft/mesosphere-zookeeper-3.4.6-0.1.20141204175332.centos7.x86_64.rpm
RUN touch /var/lib/zookeeper/myid && echo 1 > /var/lib/zookeeper/myid
RUN cp -r /etc/zookeeper/conf /opt/mesosphere/zookeeper/conf

ADD http://muser.cnlab.net/package/marathon-0.11.0-1.0.425.el7.x86_64.rpm /soft/marathon-0.11.0-1.0.425.el7.x86_64.rpm
RUN cd /soft && yum -y install marathon-0.11.0-1.0.425.el7.x86_64.rpm && rm -f /soft/marathon-0.11.0-1.0.425.el7.x86_64.rpm
RUN systemctl disable marathon

RUN yum install -y dnsmasq

ADD etc /etc
ADD scripts /data/scripts
RUN chmod a+x /data/scripts/start.sh
EXPOSE 22 5050 2181 2888 3888 8080 9080 6666
CMD ["/bin/bash","/data/scripts/start.sh"]





