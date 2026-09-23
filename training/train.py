import random
from pathlib import Path

import torch

from model.transformer import NextStepLM, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 16
SEQUENCE_LENGTH = 128
TRAINING_STEPS = 500
LEARNING_RATE = 3e-4

DATA_FILE = Path("data/train.txt")
CHECKPOINT_FILE = Path("checkpoints/nextstep-v0.pt")


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("NEXTSTEP TRAINING")
print("=================")
print("Device:", device)


# -----------------------------
# Tokenizer + Dataset
# -----------------------------

tokenizer = ByteTokenizer()

text = DATA_FILE.read_text(encoding="utf-8")

tokens = tokenizer.encode(text)

# Repeat our tiny experimental corpus so we have
# enough sequence positions for random batches.
tokens = tokens * 20

data = torch.tensor(
    tokens,
    dtype=torch.long,
)

print("Training bytes:", len(data))


# -----------------------------
# Model
# -----------------------------

config = ModelConfig()

model = NextStepLM(config).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {parameter_count:,}")


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# -----------------------------
# Batch generator
# -----------------------------

def get_batch():

    max_start = len(data) - SEQUENCE_LENGTH - 1

    starts = [
        random.randint(0, max_start)
        for _ in range(BATCH_SIZE)
    ]

    x = torch.stack(
        [
            data[
                start:
                start + SEQUENCE_LENGTH
            ]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            data[
                start + 1:
                start + SEQUENCE_LENGTH + 1
            ]
            for start in starts
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


# -----------------------------
# Training
# -----------------------------

model.train()

for step in range(1, TRAINING_STEPS + 1):

    x, y = get_batch()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits, loss = model(
        x,
        targets=y,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step == 1 or step % 25 == 0:

        print(
            f"Step {step:4d}/{TRAINING_STEPS} "
            f"| Loss: {loss.item():.4f}"
        )


# -----------------------------
# Save OUR model weights
# -----------------------------

CHECKPOINT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "training_steps": TRAINING_STEPS,
    },
    CHECKPOINT_FILE,
)

print()
print("TRAINING COMPLETE")
print("=================")
print(
    f"Saved model to: {CHECKPOINT_FILE}"
)
