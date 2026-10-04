"""Parameter-wise optimizer steps after all expected gradient contributions."""


class OptimizerInBackward:
    """Wrap one optimizer per trainable leaf tensor.

    Call begin() before each optimizer step's forwards/backwards and finish()
    afterwards. Counts are post-accumulate hook calls, not tensor uses: ordinary
    backward contributes once, while separate reentrant checkpoints contribute
    independently. Include every microbatch and checkpoint use in the counts.
    Parameters must not be needed by another forward/recomputation after their
    final contribution. Global clipping and GradScaler unscaling are unsupported.
    """

    def __init__(self, optimizers):
        self.optimizers = {}
        self.parameters = []
        for optimizer in optimizers:
            params = (list(optimizer.params) if hasattr(optimizer, 'params') else
                      [p for group in optimizer.param_groups for p in group['params']])
            if len(params) != 1:
                raise ValueError('Optimizer-in-backward requires one parameter per optimizer')
            parameter = params[0]
            if not parameter.is_leaf or not parameter.requires_grad:
                raise ValueError('Optimizer-in-backward requires trainable leaf tensors')
            if id(parameter) in self.optimizers:
                raise ValueError('A parameter cannot belong to multiple optimizers')
            self.parameters.append(parameter)
            self.optimizers[id(parameter)] = optimizer
        self.remaining = None
        self.handles = [p.register_post_accumulate_grad_hook(self.update) for p in self.parameters]

    def begin(self, contributions=(), *, default_contributions=1):
        """Set expected calls, with optional (parameter, count) overrides.

        For gradient accumulation, default_contributions is the microbatch count.
        Checkpointed/shared parameters may need larger explicit counts.
        """
        if self.remaining is not None:
            raise RuntimeError('Call finish() before beginning another optimizer step')
        if any(p.grad is not None for p in self.parameters):
            raise RuntimeError('Optimizer-in-backward requires cleared gradients at begin()')
        if not isinstance(default_contributions, int) or default_contributions < 1:
            raise ValueError('Contribution counts must be positive integers')
        remaining = {id(p): default_contributions for p in self.parameters}
        for parameter, count in contributions:
            if id(parameter) not in remaining:
                raise ValueError('Contribution override refers to an unmanaged parameter')
            if not isinstance(count, int) or count < 1:
                raise ValueError('Contribution counts must be positive integers')
            remaining[id(parameter)] = count
        self.remaining = remaining

    def update(self, parameter):
        if self.remaining is None:
            raise RuntimeError('Call begin() before backward')
        key = id(parameter)
        self.remaining[key] -= 1
        if self.remaining[key] < 0:
            raise RuntimeError('Unexpected extra gradient contribution in optimizer-in-backward')
        if self.remaining[key] == 0:
            self.optimizers[key].step()
            parameter.grad = None

    def finish(self):
        if self.remaining is None:
            raise RuntimeError('Call begin() before finish()')
        if any(self.remaining.values()) or any(p.grad is not None for p in self.parameters):
            raise RuntimeError('Optimizer-in-backward did not consume every parameter gradient')
        self.remaining = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
