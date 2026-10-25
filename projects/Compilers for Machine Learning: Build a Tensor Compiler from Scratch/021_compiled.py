import numpy as np


class Compiled:
    def __init__(self, prog):
        self.prog = prog

        # Render every kernel with placeholder name K, deduplicate sources.
        sources = []
        kernel_to_unique = []
        unique_of = {}
        for k, names in prog.kernels:
            src = render_kernel(k)
            placeholder = src.replace(f"void {k.name}(", "void K(")
            if placeholder not in unique_of:
                unique_of[placeholder] = len(sources)
                sources.append(placeholder)
            kernel_to_unique.append(unique_of[placeholder])

        self.names = [f"k{i}" for i in range(len(sources))]
        self.unique = len(sources)
        self.kernel_to_unique = kernel_to_unique

        # Replace the placeholder with the assigned names and compile all at once.
        parts = []
        for i, src in enumerate(sources):
            parts.append(src.replace("void K(", f"void {self.names[i]}(", 1))
        combined = "\n".join(parts)
        self.lib = compile_c(combined)

    def run(self, inputs):
        bufs = {}
        for name, (shape, dtype, kind) in self.prog.buffers.items():
            if kind == "input":
                arr = np.ascontiguousarray(inputs[name], dtype=dtype.np).reshape(shape)
                bufs[name] = arr
            else:
                bufs[name] = np.zeros(shape, dtype=dtype.np)

        for (k, names), uniq in zip(self.prog.kernels, self.kernel_to_unique):
            ordered = [bufs[n] for n in names]
            call_kernel(self.lib, self.names[uniq], ordered)

        return {name: bufs[name] for name, (_, _, kind) in self.prog.buffers.items()
                if kind == "output"}


def run_program_np(prog, inputs):
    bufs = {}
    for name, (shape, dtype, kind) in prog.buffers.items():
        if kind == "input":
            arr = np.ascontiguousarray(inputs[name], dtype=dtype.np).reshape(shape)
            bufs[name] = arr
        else:
            bufs[name] = np.zeros(shape, dtype=dtype.np)

    for k, names in prog.kernels:
        ordered = [bufs[n] for n in names]
        run_kernel_np(k, ordered)

    return {name: bufs[name] for name, (_, _, kind) in prog.buffers.items()
            if kind == "output"}