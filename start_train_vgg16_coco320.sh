nohup python -u train_refinedet.py --dataset coco --input_size 320 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16" > vgg16_coco320_nohup.out 2>&1 &
