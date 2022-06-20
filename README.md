# CAPM
This repository is used for verifying the maxpool-based CNN.

## Requirements

* python version 3.8.5
* packages:
  * numpy==1.19.2
  * torch == 1.8.0
  * torchvision == 0.9.0
  * psutil == 5.7.2
  * pandas == 1.1.3
  
* These packages can be automatically installed by typing: 
  ```
  pip3 install requirements.txt
  ```

## Usage
  ```
  python3 my_verify_stride.py --net <select network> --pth_file <path to pth file> --epsilon <float between 0 and 1> --mean <float(s)> --std <float(s)> --data <mnist/cifar10> -d <debug mode on/off>
  ```
  There are 10 nets can be chosen : `smallNet`, `variant_smallNet`, `largeNet`, `variant_largeNet`, `convSmall`, `convMed`, `convBig`, `convSmallCIFAR10`, `convMedCIFAR10`, `convBigCIFAR10`. The difference between original nets, e.g., `smallNet`, and variant nets, e.g., `variant_smallNet`, is the setting of maxpool. In original net, kernel size and stride are set equally; however, in variant net, the kernel size and stride do not have any setting restrictions.
  
  * `smallNet`, `variant_smallNet`, `convSmall`, `convMed`, and `convBig`: are trained on the MNIST dataset.
  * `largeNet`,  `variant_smallNet`,`convSmallCIFAR10`, `convMedCIFAR10`, and `convBigCIFAR10` : are trained on the CIFAR 10 dataset.
  * `-d`: when -d is set, CAPM will print out the result per sample. 

## Example
`python3 my_verify_stride.py --net convSmall --pth_file ./parameter/convSmall_normal_mnist.pth --epsilon 0.02` gives us the verify result of epsilon 0.0020 of convSmall in MNIST dataset.

## Customize the input net

If you want to verify your net instaed of our default net, you should define your net in ```my_verify_stride.py``` before verifying part (start at ```line 333```), then load your net and pth file at `line 404`. Note that the net you input should follow this operation oder: ```conv``` >> ```ReLU``` >> ```maxpool``` in convolution part and ```linear``` >> ```ReLU``` in fully-connected part.

## Pretrained model and reproduce our work
All model used in our paper is stored in `parameter/` folder. One can use `bash reproduce.sh` to reproduce our result in Table 6.
