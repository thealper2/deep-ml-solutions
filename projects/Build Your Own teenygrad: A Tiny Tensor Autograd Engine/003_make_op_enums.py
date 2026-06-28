from enum import Enum

def make_op_enums():
    class UnaryOps(Enum):
        NEG = 1
        RELU = 2
        LOG = 3
        EXP = 4
        SQRT = 5
        SIGMOID = 6

    class BinaryOps(Enum):
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        CMPLT = 5
        MAX = 6

    class ReduceOps(Enum):
        SUM = 1
        MAX = 2

    class MovementOps(Enum):
        RESHAPE = 1
        EXPAND = 2
        PERMUTE = 3

    return UnaryOps, BinaryOps, ReduceOps, MovementOps
