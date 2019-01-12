nohup python train_refinedet_voc.py --dataset VOC --input_size 320 --network resnet101 --basenet resnet101.pth --save_folder "weights/resnet101" > voc_320_resnet101_nohup.out 2>&1 &
