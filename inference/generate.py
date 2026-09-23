import sys
from pathlib import Path

import torch

from model.transformer import NextStepLM, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer


CHECKPOINT = Path("checkpoints/nextstep-v0.pt")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running NextStep on: {device}")


# Load our trained checkpoint
checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(
    **checkpoint["config"]
)

model = NextStepLM(config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


tokenizer = ByteTokenizer()


if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    prompt = (
        "Student: I do not understand matrices.\n"
        "Tutor:"
    )


tokens = tokenizer.encode(prompt)

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device,
)


with torch.no_grad():
    output = model.generate(
        x,
        max_new_tokens=100,
        temperature=0.35,
        top_k=5,
    )


generated = tokenizer.decode(
    output[0].tolist()
)


print()
print("NEXTSTEP")
print("========")
print(generated)
