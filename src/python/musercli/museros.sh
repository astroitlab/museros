#!/bin/bash

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#export MUSEROS_HOME=$DIR
#export MUSEROS_WORK=$DIR"/work"

if [ "$1" = "-gui" ];
then
 echo "GUI Environment"
 exec muserqt.py --gui --client
else
 exec 'museros.py'
fi
