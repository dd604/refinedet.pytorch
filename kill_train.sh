ps -ef | grep train_refinedet.py | grep -v grep | awk '{print $2}'|xargs kill -9
