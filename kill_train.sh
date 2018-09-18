ps -ef | grep train_refinedet_voc.py | grep -v grep | awk '{print $2}'|xargs kill -9
