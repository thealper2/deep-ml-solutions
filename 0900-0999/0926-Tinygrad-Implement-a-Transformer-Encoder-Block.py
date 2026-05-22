from tinygrad import Tensor, nn

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.mlp_fc1 = nn.Linear(d_model, d_ff)
        self.mlp_fc2 = nn.Linear(d_ff, d_model)

    def _reshape_heads(self, x, B, T):
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x, B, T):
        return x.transpose(1, 2).reshape(B, T, self.d_model)

    def __call__(self, x):
        B, T, D = x.shape

        norm_x = self.norm1(x)

        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)

        q = self._reshape_heads(q, B, T)
        k = self._reshape_heads(k, B, T)
        v = self._reshape_heads(v, B, T)

        attn = q.scaled_dot_product_attention(k, v)

        attn = self._merge_heads(attn, B, T)
        attn = self.attn_out(attn)

        x = x + attn.dropout(self.dropout_p)

        norm_x = self.norm2(x)

        mlp = self.mlp_fc1(norm_x).gelu()
        mlp = self.mlp_fc2(mlp)

        x = x + mlp.dropout(self.dropout_p)
        return x