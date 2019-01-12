nohup python -u train_refinedet_voc.py --dataset COCO --input_size 512 --batch_size 16 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16" &
