#!/bin/bash
output=result.txt

python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0000 --data cifar10 > $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0004 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0008 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0012 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0016 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0020 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0024 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0028 --data cifar10 >> $output
python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0032 --data cifar10 >> $output
