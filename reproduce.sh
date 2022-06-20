#!/bin/bash
output=result.txt
python3 my_verify_stride.py --net convSmall --pth_file ./parameter/convSmall_fgsm_mnist.pth --epsilon 0.03 --data mnist > $output
python3 my_verify_stride.py --net convMed --pth_file ./parameter/convMed_pgd_mnist.pth --epsilon 0.03 --data mnist >> $output
python3 my_verify_stride.py --net convBig --pth_file ./parameter/convBig_normal_mnist.pth --epsilon 0.01 --data mnist >> $output

python3 my_verify_stride.py --net convSmallCIFAR10 --pth_file ./parameter/convSmall_normal_cifar10.pth --epsilon 0.0020 --data cifar10 >> $output
python3 my_verify_stride.py --net convMedCIFAR10 --pth_file ./parameter/convMed_fgsm_cifar10.pth --epsilon 0.0028 --data cifar10 >> $output
python3 my_verify_stride.py --net convBigCIFAR10 --pth_file ./parameter/convBig_normal_cifar10.pth --epsilon 0.0006 --data cifar10 >> $output
