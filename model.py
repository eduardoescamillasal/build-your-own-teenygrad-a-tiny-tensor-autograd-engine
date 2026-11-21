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

# Step 18 - Log
(UnaryOps, BinaryOps, _, _) = make_op_enums()
class Log(Function):
    def forward(self, x):
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        one = LazyBuffer.const(1, self.x._np.shape)
        mask = lazybuffer_binary_e(one, BinaryOps.DIV, self.x)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 19 - Exp
UnaryOps, BinaryOps, _, _ = make_op_enums()
class Exp(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)

# Step 20 - Sqrt
UnaryOps, BinaryOps, _, _ = make_op_enums()

class Sqrt(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        two = LazyBuffer.const(2, self.ret._np.shape)
        denom = lazybuffer_binary_e(self.ret, BinaryOps.MUL, two)
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, denom)

# Step 21 - Sigmoid
UnaryOps, BinaryOps, _, _ = make_op_enums()

class Sigmoid(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        one = LazyBuffer.const(1, self.ret._np.shape)
        one_minus_sig = lazybuffer_binary_e(one, BinaryOps.SUB, self.ret)
        dx = lazybuffer_binary_e(self.ret, BinaryOps.MUL, one_minus_sig)
        return lazybuffer_binary_e(dx, BinaryOps.MUL, grad_output)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        grad_x = grad_output if self.needs_input_grad[0] else None
        grad_y = grad_output if self.needs_input_grad[1] else None
        return (grad_x, grad_y)

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()
        grad_x = grad_output if self.needs_input_grad[0] else None
        grad_y = grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        return (grad_x, grad_y)

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        (needsx, needsy) = self.needs_input_grad
        grad_x = lazybuffer_binary_e(self.y, BinaryOps.MUL, grad_output) if needsx else None
        grad_y = lazybuffer_binary_e(self.x, BinaryOps.MUL , grad_output) if needsy else None
        return (grad_x, grad_y)

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
       
        (needsx, needsy) = self.needs_input_grad
        grad_x = lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.y) if needsx else None
        grad_y = lazybuffer_binary_e(grad_output.e(UnaryOps.NEG), BinaryOps.MUL, self.x)
        y_SQ = lazybuffer_binary_e(self.y, BinaryOps.MUL, self.y)
        grad_y = lazybuffer_binary_e(grad_y, BinaryOps.DIV, y_SQ) if needsy else None
        return (grad_x, grad_y)

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        self.input_shape = x._np.shape
        self.axis = axis
        result = x._np.sum(axis=axis, keepdims=True)
        return LazyBuffer(result)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    view = np.broadcast_to(grad_output._np, self.input_shape)
    return LazyBuffer(np.ascontiguousarray(view))
    # return expand(grad_output, self.input_shape)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        self.x = x
        self.axis = axis
        out = x._np.max(axis=axis, keepdims=True)
        self.ret = LazyBuffer(out)
        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    _, BinaryOps, ReduceOps, _ = make_op_enums()
    shape = self.x._np.shape
    ret_exp = expand(self.ret, shape)
    ones = LazyBuffer.const(1.0, shape)
    lt_max = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, ret_exp)
    is_max = lazybuffer_binary_e(ones, BinaryOps.SUB, lt_max)
    cnts_max = expand(r(is_max, ReduceOps.SUM, self.axis), shape)
    division = lazybuffer_binary_e(is_max, BinaryOps.DIV, cnts_max)
    grad_expand = expand(grad_output, shape)
    return lazybuffer_binary_e(division, BinaryOps.MUL, grad_expand)


Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return LazyBuffer(x._np.reshape(shape))

    def backward(self, grad_output):
        return LazyBuffer(grad_output._np.reshape(self.input_shape))

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    ctx.input_shape = x.shape
    new_shape = tuple(int(d) for d in shape)
    out = np.broadcast_to(x._np, new_shape)
    return LazyBuffer(np.ascontiguousarray(out))

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    _, _, ReduceOps, _ = make_op_enums()
    in_shape = ctx.input_shape
    grad_shape = grad_output._np.shape
    axes = tuple(i for i in range(len(in_shape)) if in_shape[i] == 1 and grad_shape[i] != 1)
    return r(grad_output, ReduceOps.SUM, axes )

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

