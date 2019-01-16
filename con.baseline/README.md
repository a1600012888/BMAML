## Requiresments:
* pytorch==0.4.1
* tensorboardX

## How to run
* creat an empty directory named *logs* aside direcotry *code*
* python3 Dataset.py to create simulated data.
* python3 main.py to train
* tensorborad: `cd log; tensorboard --logdir ./ runs`


## Hyper-param tuning

* *meta-lr*: line 62 in main.py
* *inner-lr*: line 19 in main.py
* number of iterations for the inner fitting:  line 20 in main.py
* *learing rate* for optimizing the rnn kernel: line 75 in main.py
* *Time steps*:  line 24 in main.py
* *Bandwidth* for the RBF kernel:  line 70 in main.py
* number of initial particles:  line 16 in main.py
* How long to train the kernel once:   line 23 in main.py
* number of total training tasks: line 22 in main.py
* How big is the rnn(number of hindden state):  line 72 in config.py

You can use tensorboard to see the loss curve.

## How to visulize
* using jupyter to run visual.ipynb
* run all the cells
* you can run the last cell as many times as you like
