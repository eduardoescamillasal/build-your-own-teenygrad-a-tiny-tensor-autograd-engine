"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
import math
def prod(shape):
    return math.prod(shape)

# Step 2 - argsort
def argsort(values):
    return [i for i, _ in sorted(enumerate(values), key=lambda pair: pair[1])]

# Step 3 - make_op_enums
from enum import Enum, auto

def make_op_enums():
    class UnaryOps(Enum):
        NEG = auto()
        RELU = auto()
        LOG = auto()
        EXP = auto()
        SQRT = auto()
        SIGMOID = auto()
    
    class BinaryOps(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
        CMPLT = auto()
        MAX = auto()

    class ReduceOps(Enum):
        SUM = auto()
        MAX = auto()

    class MovementOps(Enum):
        RESHAPE = auto()
        EXPAND = auto()
        PERMUTE = auto()

    return (UnaryOps, BinaryOps, ReduceOps, MovementOps)

# Step 4 - LazyBuffer
import numpy as np
class LazyBuffer:
    def __init__(self, np_array):
        np_array = np.asarray(np_array)
        self._np = np_array
        self.shape = np_array.shape
        self.dtype = np_array.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    arr = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(arr)
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    rng = np.random.RandomState(seed)
    data = rng.random(shape).astype(np.float32)
    return LazyBuffer(data)

# Step 7 - lazybuffer_unary_e
def e(self, op, *srcs):
    x = self._np
    name = op.name

    if name == "NEG":
        out = -x
    elif name == "RELU":
        out = np.maximum(x, 0)
    elif name == "LOG":
        out = np.log(x)
    elif name == "EXP":
        out = np.exp(x)
    elif name == "SQRT":
        out = np.sqrt(x)
    elif name == "SIGMOID":
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(f"Unsopported unary op: {op}")

    return LazyBuffer(out)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    x = self._np
    y = other._np
    name = op.name

    if name == "ADD":
        out = x + y
    elif name == "SUB":
        out = x - y
    elif name == "MUL":
        out = x * y
    elif name == "DIV":
        out = x / y
    elif name == "CMPLT":
        out = (x < y).astype(np.float32)
    elif name == "MAX":
        out = np.maximum(x, y)
    else:
        raise ValueError(f"Unsupported binary op: {op}")
    
    return LazyBuffer(out)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    name = getattr(op, 'name', None)
    x = self._np

    if name == "SUM":
        out = x.sum(axis=axis, keepdims=True)
    elif name == "MAX":
        out = x.max(axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsopported reduce op: {op}")
    
    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    return LazyBuffer(self._np.reshape(new_shape))

LazyBuffer.reshape = reshape

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    target_shape = tuple(int(dim) for dim in new_shape)
    view = np.broadcast_to(self._np, target_shape)
    return LazyBuffer(np.array(view))

# Step 12 - lazybuffer_permute
def permute(self, order):
    x = self._np
    axes = tuple(int(axis) for axis in order)
    return LazyBuffer(x.transpose(axes))

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        self.needs_input_grad = [t.requires_grad for t in tensors]

        if any(flag is True for flag in self.needs_input_grad):
            self.requires_grad = True
        elif None in self.needs_input_grad:
            self.requires_grad = None
        else:
            self.requires_grad = False
        
        if self.requires_grad:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")
        

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    ctx = cls(*tensors)
    bufs = [t.lazydata for t in tensors]
    out_bufs = ctx.forward(*bufs, **kwargs)
    out = Tensor(out_bufs, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx
    return out


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
UnaryOps, BinaryOps, _, _ = make_op_enums()
class Relu(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        zero = LazyBuffer.const(0, self.ret._np.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

