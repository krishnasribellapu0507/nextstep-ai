import random
import re
from pathlib import Path

import torch

from model.transformer import NextStepLM, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer


# =========================================================
# NEXTSTEP V1 CONFIGURATION
# =========================================================

BATCH_SIZE = 8
SEQUENCE_LENGTH = 256

TRAINING_STEPS = 1500
LEARNING_RATE = 3e-4

EVAL_EVERY = 100
EVAL_BATCHES = 20

TRAIN_FILE = Path("data/v1/train.txt")
VALIDATION_FILE = Path("data/v1/validation.txt")

CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT = CHECKPOINT_DIR / "nextstep-v1-best.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "nextstep-v1-final.pt"

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("NEXTSTEP V1 TRAINING")
print("====================")
print("Device:", device)


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = ByteTokenizer()


# =========================================================
# LOAD DATA
# =========================================================

train_text = TRAIN_FILE.read_text(
    encoding="utf-8"
)

validation_text = VALIDATION_FILE.read_text(
    encoding="utf-8"
)

train_tokens = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long,
)

validation_tokens = torch.tensor(
    tokenizer.encode(validation_text),
    dtype=torch.long,
)

print(
    f"Training bytes:   {len(train_tokens):,}"
)

print(
    f"Validation bytes: {len(validation_tokens):,}"
)


# =========================================================
# MODEL
# =========================================================

config = ModelConfig(
    vocab_size=256,
    context_length=256,
    d_model=192,
    n_heads=6,
    n_layers=4,
    d_ff=768,
)

model = NextStepLM(config).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters:       {parameter_count:,}"
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)


# =========================================================
# BATCH GENERATOR
# =========================================================

def get_batch(data, batch_size=BATCH_SIZE):

    max_start = (
        len(data)
        - SEQUENCE_LENGTH
        - 1
    )

    if max_start <= 0:
        raise ValueError(
            "Dataset is too small for the "
            "configured sequence length."
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[
                i:
                i + SEQUENCE_LENGTH
            ]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:
                i + SEQUENCE_LENGTH + 1
            ]
            for i in starts
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


# =========================================================
# VALIDATION LOSS
# =========================================================

@torch.no_grad()
def estimate_validation_loss():

    model.eval()

    losses = []

    for _ in range(EVAL_BATCHES):

        x, y = get_batch(
            validation_tokens
        )

        _, loss = model(
            x,
            targets=y,
        )

        losses.append(
            loss.item()
        )

    model.train()

    return (
        sum(losses)
        / len(losses)
    )


# =========================================================
# SAVE CHECKPOINT
# =========================================================

def save_checkpoint(path, step, train_loss, val_loss):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "config":
                vars(config),

            "step":
                step,

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss,
        },
        path,
    )


# =========================================================
# TRAINING
# =========================================================

best_validation_loss = float("inf")

model.train()

print()
print("Training started...")
print()

for step in range(
    1,
    TRAINING_STEPS + 1
):

    x, y = get_batch(
        train_tokens
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(
        x,
        targets=y,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if (
        step == 1
        or step % 25 == 0
    ):

        print(
            f"Step {step:4d}/{TRAINING_STEPS}"
            f" | Train Loss: "
            f"{loss.item():.4f}"
        )

    if step % EVAL_EVERY == 0:

        validation_loss = (
            estimate_validation_loss()
        )

        print(
            f"         "
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            save_checkpoint(
                BEST_CHECKPOINT,
                step,
                loss.item(),
                validation_loss,
            )

            print(
                "         "
                "✓ New best model saved"
            )


# =========================================================
# FINAL CHECKPOINT
# =========================================================

final_validation_loss = (
    estimate_validation_loss()
)

save_checkpoint(
    FINAL_CHECKPOINT,
    TRAINING_STEPS,
    loss.item(),
    final_validation_loss,
)


print()
print("NEXTSTEP V1 TRAINING COMPLETE")
print("=============================")

print(
    "Best validation loss:",
    round(
        best_validation_loss,
        4
    ),
)

print(
    "Final validation loss:",
    round(
        final_validation_loss,
        4
    ),
)

print()
print(
    "Best model:",
    BEST_CHECKPOINT,
)

print(
    "Final model:",
    FINAL_CHECKPOINT,
)
