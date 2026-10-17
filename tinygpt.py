"""从零实现的微型 GPT（char-level，无任何外部模型代码依赖）。

架构就是 GPT-2 的最小化复刻：
    token embedding + position embedding
    -> N x [LayerNorm -> 因果多头注意力 -> LayerNorm -> MLP]（残差连接）
    -> LayerNorm -> lm head（输出词表分布）

和 nanoGPT / minGPT 的教学实现同构，这里是逐行手写版，
参数量约 70 万，CPU 可训。规模小了五个数量级，但一个字都没少：
注意力、因果 mask、残差、layernorm、dropout 全都有。
"""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个位置只能看到自己和过去。"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 因果 mask：上三角为 -inf，禁止看到未来
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.dropout(att.softmax(dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block：LN -> attention -> 残差 -> LN -> MLP -> 残差。"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """微型 GPT + 可选的世界模型潜状态注入通道（L2 半耦合）。

    潜状态注入：如果传入 state（B, state_dim），先经 state_proj 投影到
    (B, d_model)，再加到输入序列中 [STATE] token 位置的 embedding 上。
    这样语言模型回答感知类问题时，是真的"看到"了世界模型的内部表征，
    而不是靠关键词触发外部查询（L1）。没有 [STATE] token 的样本
    （普通文本）mask 全零，state 自动失效，行为与纯 GPT 一致。
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3, block_size=128,
                 dropout=0.1, state_dim=0, state_token_id=None):
        super().__init__()
        self.block_size = block_size
        self.state_dim = state_dim
        self.state_token_id = state_token_id
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        if state_dim > 0:
            # 投影层：单层 Linear。
            # 曾经试过 MLP（Linear→GELU→Linear），冷启动参数从 2 万涨到 23 万，
            # 4 epochs 学不动，[STATE] 位置被注入随机噪声，感知通道整体失效
            # （回答退化成最常见文本模式）。MLP 版存档为"表达力强但冷启动难"的教训。
            self.state_proj = nn.Linear(state_dim, d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, state=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        # 潜状态注入：只加到 [STATE] token 位置上
        if state is not None and self.state_dim > 0 and self.state_token_id is not None:
            s = self.state_proj(state)                          # (B, d_model)
            mask = (idx == self.state_token_id).unsqueeze(-1).to(x.dtype)  # (B, T, 1)
            x = x + s.unsqueeze(1) * mask
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20, stop_ids=None, state=None):
        """自回归生成：带 temperature + top-k 采样，遇到 stop_ids 停止。

        state 非空时每步都注入（[STATE] token 在 prompt 前部，不会滑出窗口）。
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond, state=state)
            logits = logits[:, -1, :] / max(temperature, 1e-3)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if stop_ids is not None and next_id.item() in stop_ids:
                break
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TinyGPT(vocab_size=1500)
    print("params:", count_params(m))
    x = torch.randint(0, 1500, (2, 32))
    logits, loss = m(x, x)
    print("logits:", tuple(logits.shape), "loss:", round(loss.item(), 3))
    out = m.generate(x, max_new_tokens=20)
    print("generated:", tuple(out.shape))
