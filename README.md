# Ranger - Custom Optimizer for MXnet

Unofficial implementation of 'Ranger - Synergistic combination of RAdam + LookAhead for the best of both' using **mxnet gluon**.

Original medium article: https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

Official implementation: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer


### How to use

Import 'ranger_optimizer' file
```
import ranger_optimizer
``` 

Create optimizer object
```
optimizer = ranger_optimizer.Ranger(learning_rate=0.001, wd=0)
``` 

Create trainer object
```
trainer = gluon.Trainer(net.collect_params(), optimizer)
```

### Hyper Parameters
learning_rate : float, default 0.001<br>
        The initial learning rate. If None, the optimization will use the<br>
        learning rate from lr_scheduler. If not None, it will overwrite<br>
        the learning rate in lr_scheduler. If None and lr_scheduler<br>
        is also None, then it will be set to 0.001 by default.<br><br>
    wd : float, default 0<br>
        Weight decay value.<br><br>
    alpha : float, default 0.5<br>
        Scale the subtraction between 'fast learner' and 'slow learner'.<br><br>
    k : int, default 6<br>
        Update 'slow learner' after 6 updates.<br><br>
    beta1 : float, default 0.9<br>
        Exponential decay rate for the first moment estimates.<br><br>
    beta2 : float, default 0.999<br>
        Exponential decay rate for the second moment estimates.<br><br>
    epsilon : float, default 1e-8<br>
        Small value to avoid division by 0.<br><br>
    use_gc : bool, default True<br>
        Center the gradients by subtract mean<br><br>
    gc_conv_only : bool, default False<br>
        Whether to center only convolution layers or everything<br><br>