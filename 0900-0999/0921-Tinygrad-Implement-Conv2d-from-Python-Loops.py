from tinygrad import Tensor

def conv2d_loop(x, W, b):
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape

    out_h = H - kH + 1
    out_w = W_in - kW + 1

    rows = []

    for i in range(out_h):
        cols = []

        for j in range(out_w):
            patch = x[:, :, i:i+kH, j:j+kW]

            out = (
                patch.unsqueeze(1)
                * W.unsqueeze(0)
            ).sum(axis=(2, 3, 4))

            out = out + b.reshape(1, -1)

            cols.append(out.unsqueeze(-1))

        row = cols[0]
        for col in cols[1:]:
            row = row.cat(col, dim=2)

        rows.append(row.unsqueeze(2))

    output = rows[0]
    for row in rows[1:]:
        output = output.cat(row, dim=2)

    return output