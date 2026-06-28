class Function:
    def __init__(self, *inputs):
        self.needs_input_grad = []
        parent_tensors = []
        requires_grad = False
        
        for inp in inputs:
            if hasattr(inp, 'requires_grad'):
                needs = inp.requires_grad
                self.needs_input_grad.append(needs)
                if needs:
                    parent_tensors.append(inp)
                    requires_grad = True
            else:
                self.needs_input_grad.append(None)
                if inp is None:
                    requires_grad = None
        
        if any(g is True for g in self.needs_input_grad):
            self.requires_grad = True
        elif any(g is None for g in self.needs_input_grad):
            self.requires_grad = None
        else:
            self.requires_grad = False
        
        if self.requires_grad:
            self.parents = parent_tensors
