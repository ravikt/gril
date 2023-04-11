##     Gaze Regularized Imitation Learning:
### Learning Continuous Control from Human Gaze

<img src="misc/gril.png" width="600">


This repository conatins implementationof gaze regualrized imitation learning. The trained model is tested in AisrSim with a quadrotor to perform tasks like search and navigation autonomously. At present the repository contains implementation of GRIL with baseline algorithms. The below image shows GRIL's generalization on movning target scenario.


<img src="misc/gril.gif" width="600">

### Requirements

The code has been tested on Ubuntu 18.04. The development environment is based on Python3 and Tensorflow 2.0+. This environment can be setup in a virtual environment using the requirements file included in the repository. After creating virtual environment, run the following. 
```bash
pip3 install -r requirements.txt
```

### Reference

"[Gaze Regularized Imitation Learning: Learning Continuous Control from Human Gaze](https://arxiv.org/pdf/2102.13008.pdf)" arXiv preprint arXiv:2102.13008 (2023)



