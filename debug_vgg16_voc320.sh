CUDA_VISIBLE_DEVICES=0 python -m pdb train_refinedet.py --dataset voc --input_size 320 --batch_size 4 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16"
