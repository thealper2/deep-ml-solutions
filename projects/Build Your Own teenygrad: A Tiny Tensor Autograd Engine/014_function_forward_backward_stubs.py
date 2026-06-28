def function_forward_backward_stubs():
    def forward_stub(self, *args):
        raise NotImplementedError("Subclasses must implement forward()")

    def backward_stub(self, *grad_output):
        raise NotImplementedError("Subclasses must implement backward()")

    Function.forward = forward_stub
    Function.backward = backward_stub
