##     Gaze Regularized Imitation Learning:
### Learning Continuous Control from Human Gaze

<img src="misc/gril.png" width="600">


This repository contains the implementation of gaze-regularized imitation learning. The trained model is tested in AirSim with a quadrotor to perform tasks like search and navigation autonomously. At present, the repository contains implementation of GRIL with baseline algorithms. The GIF below shows GRIL's generalization on a moving target scenario.


<img src="misc/gril.gif" width="600">

### Requirements

The code has been tested on Ubuntu 18.04. The development environment is based on Python3 and Tensorflow 2.0+. This environment can be setup in a virtual environment using the requirements file included in the repository. After creating virtual environment, run the following. 
```bash
pip3 install -r requirements.txt
```

### Reference

Paper: [Imitation Learning with Human Eye Gaze via Multi-Objective Prediction](https://openreview.net/pdf?id=MF1Tcmk2YB) ILHF Workshop ICML 2023 
Video: [Quadrotor following a moving truck](https://www.youtube.com/watch?v=vFiT293FR1Q)



