import re
from pathlib import Path

import torch
import torch.nn as nn

from learning_engine.action_model import ACTIONS, NextActionModel


TRAIN_FILE = Path("data/v1/train.txt")
VAL_FILE = Path("data/v1/validation.txt")
OUTPUT = Path("checkpoints/nextstep-action-v1.pt")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_examples(path):
    text = path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"mastery=([0-9.]+)\n"
        r"mistake_rate=([0-9.]+)\n"
        r"forgetting_risk=([0-9.]+)\n"
        r"exam_days=([0-9]+)\n\n"
        r"<ACTION>\n"
        r"([a-z_]+)"
    )

    X = []
    Y = []

    for match in pattern.finditer(text):
        mastery = float(match.group(1))
        mistakes = float(match.group(2))
        forgetting = float(match.group(3))
        exam_days = int(match.group(4))
        action = match.group(5)

        X.append([
            mastery,
            mistakes,
            forgetting,
            min(exam_days, 30) / 30.0,
        ])

        Y.append(ACTIONS.index(action))

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.long),
    )


train_x, train_y = load_examples(TRAIN_FILE)
val_x, val_y = load_examples(VAL_FILE)

print()
print("NEXTSTEP ACTION NETWORK")
print("=======================")
print("Device:", device)
print("Training examples:", len(train_x))
print("Validation examples:", len(val_x))

model = NextActionModel().to(device)

train_x = train_x.to(device)
train_y = train_y.to(device)
val_x = val_x.to(device)
val_y = val_y.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
)

criterion = nn.CrossEntropyLoss()

best_accuracy = 0.0

for epoch in range(1, 201):

    model.train()

    optimizer.zero_grad()

    logits = model(train_x)

    loss = criterion(
        logits,
        train_y,
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:

        model.eval()

        with torch.no_grad():
            val_logits = model(val_x)

            predictions = torch.argmax(
                val_logits,
                dim=1,
            )

            accuracy = (
                predictions == val_y
            ).float().mean().item()

        print(
            f"Epoch {epoch:3d}/200"
            f" | Loss: {loss.item():.4f}"
            f" | Val Accuracy: {accuracy * 100:.1f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            OUTPUT.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "actions": ACTIONS,
                    "validation_accuracy": accuracy,
                },
                OUTPUT,
            )

print()
print("TRAINING COMPLETE")
print("=================")
print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.1f}%"
)
print("Saved:", OUTPUT)
