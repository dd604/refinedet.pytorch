nohup python -u train_refinedet_voc.py --dataset VOC --input_size 512 --batch_size 16 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16" >> voc_512_vgg16_nohup.out 2>&1 &
