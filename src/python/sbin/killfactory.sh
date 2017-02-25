#!/bin/sh

processID=`ps aux | grep python | grep startfactory.py | awk '{print $2}'`

if [ "$processID"x = ""x ]; then
	echo "not found factory"
else
	echo "about to kill factory....."
	`kill -9 $processID`
fi


