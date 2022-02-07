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
  python my_verify_stride.py [input the net]
  ```
  There are 4 nets can be chosen : ```smallNet, variant_smallNet, largeNet, variant_largeNet```. The difference between original nets, e.g., smallNet, and variant nets, e.g., variant_smallNet, is the setting of maxpool. In original net, kernel size and stride should be equal; however, in variant net, the kernel size and stride are without any restrictions.
  
  * ```smallNet```  and  ```variant_smallNet``` : are trained on the MNIST dataset.
  * ```largeNet```  and  ```variant_smallNet``` : are trained on the CIFAR 10 dataset.

## Modify the input net

If you want to verify your net instaed of our default net, you should load your net in ```my_verify_stride.py``` before verifying part (start at ```line 222```). Note that the net you input should follow this operation oder: ```conv``` >> ```ReLU``` >> ```maxpool``` in convolution part and ```linear``` >> ```ReLU``` in fully-connected part.
