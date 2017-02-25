#!/bin/sh
if [ -z $1 ]
then
    SLAVE_NUM=1
else
    SLAVE_NUM=$1
fi

echo $SLAVES
docker run -d -v /astrodata -v /astrodata:/astrodata --name astrodata centos:latest
MASTER=$(docker run --privileged=true --dns 127.0.0.1 -d --volumes-from astrodata opencluster:master)
MASTER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' $MASTER)

echo MASTER IP : ${MASTER_IP}

SLAVES=()
for i in $(seq ${SLAVE_NUM})
do
    SLAVES+=($(docker run --privileged=true --dns ${MASTER_IP} -e MASTER_IP=${MASTER_IP} -d --volumes-from astrodata opencluster:slave))
done

trap cleanup 2

cleanup()
{
    docker kill ${MASTER}
    for SLAVE in ${SLAVES}
    do
        docker kill ${SLAVE}
    done
    docker rm ${MASTER}
    for SLAVE in ${SLAVES}
    do
        docker rm ${SLAVE}
    done
    exit 0
}

sleep 3
echo password is muser
ssh -l muser ${MASTER_IP}

cleanup

