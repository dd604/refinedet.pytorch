nohup python train_refinedet.py --dataset voc --input_size 512 --network resnet101 --basenet resnet101.pth --save_folder "weights/resnet101" > resnet101_voc512_nohup.out 2>&1 &
