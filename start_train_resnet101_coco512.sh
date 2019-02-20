nohup python -u train_refinedet.py --dataset coco --input_size 512 --batch_size 16 --network resnet101 --basenet resnet101.pth --save_folder "weights/resnet101" > resnet101_coco512_nohup.out 2>&1 &
