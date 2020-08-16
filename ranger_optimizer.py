from mxnet import nd
from mxnet.optimizer import Optimizer, register
import math


@register
class Ranger(Optimizer):
    """
    Parameters
    ----------
    learning_rate : float, default 0.001
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.001 by default.
    alpha : float, default 0.5
        Scale the subtraction between 'fast learner' and 'slow learner'.
    k : int, default 6
        Update 'slow learner' after 6 updates.
    beta1 : float, default 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    use_gc : bool, default True
        Center the gradients by subtract mean
    gc_conv_only : bool, default False
        Whether to center only convolution layers or everything
    """
    def __init__(self, learning_rate=0.001,
                 alpha=0.5, k=6, n_sma_threshhold=5,  # Ranger options
                 beta1=0.9, beta2=0.999, epsilon=1e-8, # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False,
                 **kwargs):
        super(Ranger, self).__init__(learning_rate=learning_rate, **kwargs)

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not learning_rate > 0:
            raise ValueError(f'Invalid Learning Rate: {learning_rate}')
        if not epsilon > 0:
            raise ValueError(f'Invalid eps: {epsilon}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # adjustable threshold
        self.n_sma_threshhold = n_sma_threshhold

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for _ in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def create_state(self, index, weight):
        return {'step': 0,
                'exp_avg': nd.zeros(weight.shape, weight.context, dtype=weight.dtype),
                'exp_avg_sq': nd.zeros(weight.shape, weight.context, dtype=weight.dtype),
                'slow_buffer': nd.array(weight, ctx=weight.context)}

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        state['step'] += 1

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        grad += wd * weight

        # Gradient Centralization operation for Conv layers and FC layers
        if self.use_gc and len(grad.shape) > self.gc_gradient_threshold:
            grad = grad - grad.mean(axis=tuple(range(1, len(grad.shape))), keepdims=True)

        # compute mean moving avg and variance moving avg
        state['exp_avg'] = (state['exp_avg'] * self.beta1) + ((1 - self.beta1) * grad)
        state['exp_avg_sq'] = (state['exp_avg_sq'] * self.beta2) + ((1 - self.beta2) * grad * grad)

        buffered = self.radam_buffer[int(state['step'] % 10)]

        if state['step'] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = state['step']
            beta2_t = self.beta2 ** state['step']
            N_sma_max = 2 / (1 - self.beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma
            if N_sma > self.n_sma_threshhold:
                step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                        N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - self.beta1 ** state['step'])
            else:
                step_size = 1.0 / (1 - self.beta1 ** state['step'])
            buffered[2] = step_size
            self.radam_buffer[int(state['step'] % 10)] = buffered

        # apply lr
        new_lr = -step_size * lr
        if N_sma > self.n_sma_threshhold:
            denom = state['exp_avg_sq'].sqrt() + self.epsilon
            weight[:] += new_lr * (state['exp_avg'] / denom)
        else:
            weight[:] += new_lr * state['exp_avg']

        # integrated look ahead
        if state['step'] % self.k == 0:
            state['slow_buffer'] += (weight - state['slow_buffer']) * self.alpha
            weight[:] = state['slow_buffer']
