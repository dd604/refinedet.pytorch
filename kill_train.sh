ps -ef | grep train.py | grep -v grep | awk '{print $2}'|xargs kill -9
