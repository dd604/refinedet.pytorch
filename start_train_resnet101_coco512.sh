nohup python train_refinedet_voc.py --dataset COCO --input_size 512 --network resnet101 --basenet resnet101.pth --save_folder "weights/resnet101" > coco_512_resnet101_nohup.out 2>&1 &
