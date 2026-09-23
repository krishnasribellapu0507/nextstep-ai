from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 256

    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Create Query, Key and Value projections ourselves.
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False,
        )

        self.output = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

        # Causal mask:
        # token 5 may see tokens 0..5,
        # but may NOT see future tokens.
        mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                config.context_length,
                config.context_length,
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :T, :T],
            float("-inf"),
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        out = attention @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T, C)

        return self.output(out)


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.d_model,
                config.d_ff,
            ),
            nn.GELU(),
            nn.Linear(
                config.d_ff,
                config.d_model,
            ),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x):

        # Residual connection around attention
        x = x + self.attention(
            self.ln1(x)
        )

        # Residual connection around FFN
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


class NextStepLM(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Tie input and output embeddings.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        tokens,
        targets=None,
    ):

        B, T = tokens.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(
            T,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.config.vocab_size,
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens=100,
        temperature=1.0,
        top_k=40,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            context = tokens[
                :,
                -self.config.context_length:
            ]

            logits, _ = self(context)

            logits = logits[:, -1, :]

            logits = logits / max(
                temperature,
                1e-5,
            )

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1),
                    ),
                )

                cutoff = values[:, [-1]]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

        return tokens
