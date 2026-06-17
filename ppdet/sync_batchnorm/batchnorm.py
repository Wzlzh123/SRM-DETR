import paddle
import paddle.nn.functional as F
from paddle import nn
import collections

from paddle.nn import BatchNorm1D, BatchNorm2D, BatchNorm3D


# SyncMaster is expected to be implemented elsewhere for synchronization
from .comm import SyncMaster

__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']

def _sum_ft(tensor):
    """Sum over the first and last dimension"""
    return paddle.sum(tensor, axis=0).sum(axis=-1)

def _unsqueeze_ft(tensor):
    """Add new dimensions at the front and the tail"""
    return paddle.unsqueeze(paddle.unsqueeze(tensor, 0), -1)

_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(nn.BatchNorm1D):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        
        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use Paddle's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.weight, self.bias, self.running_mean, self.running_var, 
                                self.training, self.momentum, self.eps)
        
        # Resize the input to (B, C, -1).
        input_shape = paddle.shape(input)
        input = paddle.reshape(input, (input_shape[0], self.num_features, -1))

        # Compute the sum and square-sum.
        sum_size = input.shape[0] * input.shape[2]
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return paddle.reshape(output, input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.place)

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.place for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = paddle.distributed.all_reduce([i for i in to_reduce])

        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = paddle.distributed.broadcast(mean, target_gpus)
        broadcasted_std = paddle.distributed.broadcast(inv_std, target_gpus)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(broadcasted[i], broadcasted_std[i])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var

        return mean, paddle.maximum(bias_var, self.eps) ** -0.5

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(SynchronizedBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError(f"Expected 4D input (got {len(input.shape)}D input)")
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)
