# CAPM
This repository is used to verify the maxpool-based CNN.

## Introduction
**Problem Statement**: In the past few years, convolution neural networks have reached unprecedented performance in various tasks such as face recognition and self-driving cars, to name
a few. However, these networks are vulnerable to malicious modification of the pixels in input images, known as adversarial examples, such as FGSM, PGD, One Pixel Attack, Deepfool, EAD, GAP, MaF and many others. Because of the threat posed by adversarial examples, how to protect neural networks from being tricked by adversarial examples has become an emerging research topic. Therefore, the need for guaranteed robustness assessments has led to the developing of verification mechanisms for a neural network. These verify specific properties of neural networks, such as robustness against norm-bounded perturbation.  

![image](https://github.com/jiahaubai/CAPM/blob/main/images/attck.png)

**CAPM** (Convex Adversarial Polytope for Maxpool-based CNN) is a verification tool that provides verified bounds for a Maxpool-based CNN, assuming l∞ norm-bounded input perturbations.

![image](https://github.com/jiahaubai/CAPM/blob/main/images/verification.png)

## Experiment
We evaluate the verified robustness and average verified time of CAPM against [DeepZ](https://papers.nips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html), [DeepPoly](https://dl.acm.org/doi/10.1145/3290354), and [PRIMA(SOTA in 2022)](https://dl.acm.org/doi/abs/10.1145/3498704) with l∞ norm-bounded perturbation of various budgets under various attack schemes, such as [FGSM](https://arxiv.org/pdf/1412.6572) and [PGD](https://arxiv.org/abs/1706.06083). All experiments are conducted on a 2.6 GHz 14 cores Intel(R) Xeon(R) CPU E5-2690 v4 with a main memory of 512 GB.

### Settings
- **Dataset**: We trained our models on MNIST and CIFAR10 datasets, where the images are normalized following the default setting in DeepPoly. In MNIST, the mean and standard deviation are 0.5 and 0.5, respectively. In CIFAR 10, the mean and standard deviation of the RGB channels are (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225), respectively.
  
- **Architecture of neural networks**: Since there are no empirical verification results on maxpool-based CNNs, we add maxpool layers to common benchmark networks convSmall and convBig in [DiffAI](https://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf). We change the
parameters of striding and padding to achieve a similar number of parameters in the literature. CNNSmall is trained using the MNIST dataset, while CNNBig is trained using the CIFAR10 dataset. 

![image](https://github.com/jiahaubai/CAPM/blob/main/images/net_architecture.png)
  
- **Performance metrics**: The performance of neural network verification is often evaluated through the following metrics,
   - **verified robustness**: The number of images verified to be resilient to adversary example attack, divided by the total number of accurate 
     images. This ratio indicates the analysis precision of a verifier when the neural network is applied to the test image dataset that 
     encounters an adversarial example attack.
   - **average verified time**: The total time spent by the verification algorithm on the verified images divided by the total number of images.
### Result
* CNNSmall
![image](https://github.com/jiahaubai/CAPM/blob/main/images/CNN_small.png)
CAPM can achieve comparable validation bounds to RefinePoly but with significantly less computational time. Specifically, when epsilon is 0.09, CAPM is 40 times faster than RefinePoly(PRIMA).

* CNNBig
![image](https://github.com/jiahaubai/CAPM/blob/main/images/CNN_big.png)
CAPM outperforms DeepZ, DeepPoly, and RefinePoly(PRIMA) in both precision and computation cost.

More detailed settings and results are currently being submitted and will not be made public.
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
  pip3 install - r requirements.txt
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
All model used in our paper is stored in `parameter/` folder. One can use `bash reproduce.sh` to reproduce our result in Fig 11 convSmall CIFAR10 normal train's result.
