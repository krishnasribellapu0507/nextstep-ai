from pathlib import Path

import torch

from model.transformer import NextStepLM, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer


CHECKPOINT = Path("checkpoints/nextstep-v1-best.pt")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = ByteTokenizer()


# ---------------------------------------------------------
# Load our NextStep-v1 model
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------

@torch.no_grad()
def predict_action(prompt, max_tokens=40):

    input_tokens = tokenizer.encode(prompt)

    tokens = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=device,
    )

    generated = []

    for _ in range(max_tokens):

        context = tokens[
            :,
            -config.context_length:
        ]

        logits, _ = model(context)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()

        # Stop when the model produces a newline
        if token_id == 10 and generated:
            break

        generated.append(token_id)

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    text = tokenizer.decode(generated)

    return text.strip()


# ---------------------------------------------------------
# Truly hand-written test cases
# ---------------------------------------------------------

tests = [

    {
        "student":
            "I barely know how determinants work.",
        "topic": "determinants",
        "mastery": 0.18,
        "mistakes": 0.35,
        "forgetting": 0.20,
        "exam_days": 14,
        "expected": "teach_foundation",
    },

    {
        "student":
            "I understand the basics but need more practice.",
        "topic": "probability",
        "mastery": 0.52,
        "mistakes": 0.35,
        "forgetting": 0.25,
        "exam_days": 7,
        "expected": "guided_practice",
    },

    {
        "student":
            "I keep making the same sign mistake.",
        "topic": "eigenvalues",
        "mastery": 0.55,
        "mistakes": 0.88,
        "forgetting": 0.20,
        "exam_days": 5,
        "expected": "correct_misconception",
    },

    {
        "student":
            "I learned this before but now I cannot recall it.",
        "topic": "pn junction",
        "mastery": 0.65,
        "mistakes": 0.20,
        "forgetting": 0.91,
        "exam_days": 10,
        "expected": "retrieval_practice",
    },

    {
        "student":
            "My exam is tomorrow and I am still weak here.",
        "topic": "rectifiers",
        "mastery": 0.40,
        "mistakes": 0.45,
        "forgetting": 0.40,
        "exam_days": 1,
        "expected": "exam_priority",
    },

    {
        "student":
            "The normal questions are becoming too easy.",
        "topic": "matrices",
        "mastery": 0.90,
        "mistakes": 0.10,
        "forgetting": 0.10,
        "exam_days": 20,
        "expected": "increase_difficulty",
    },

    {
        "student":
            "I am totally lost with Python functions.",
        "topic": "functions in python",
        "mastery": 0.22,
        "mistakes": 0.50,
        "forgetting": 0.30,
        "exam_days": 6,
        "expected": "teach_foundation",
    },

    {
        "student":
            "I repeatedly choose the wrong formula.",
        "topic": "differentiation",
        "mastery": 0.48,
        "mistakes": 0.82,
        "forgetting": 0.25,
        "exam_days": 4,
        "expected": "correct_misconception",
    },

    {
        "student":
            "I studied it well but it has been many days.",
        "topic": "data structures",
        "mastery": 0.72,
        "mistakes": 0.12,
        "forgetting": 0.83,
        "exam_days": 15,
        "expected": "retrieval_practice",
    },

    {
        "student":
            "I know most of this topic and rarely make mistakes.",
        "topic": "loops in python",
        "mastery": 0.86,
        "mistakes": 0.09,
        "forgetting": 0.15,
        "exam_days": 12,
        "expected": "increase_difficulty",
    },

]


correct = 0

print()
print("NEXTSTEP V1 ACTION EVALUATION")
print("=============================")
print()


for number, test in enumerate(tests, 1):

    prompt = f"""<STUDENT>
{test["student"]}

<STATE>
topic={test["topic"]}
mastery={test["mastery"]:.2f}
mistake_rate={test["mistakes"]:.2f}
forgetting_risk={test["forgetting"]:.2f}
exam_days={test["exam_days"]}

<ACTION>
"""

    predicted = predict_action(prompt)

    passed = (
        predicted == test["expected"]
    )

    if passed:
        correct += 1

    symbol = "PASS" if passed else "FAIL"

    print(
        f"{number:02d}. {symbol}"
    )

    print(
        f"    Expected:  {test['expected']}"
    )

    print(
        f"    Predicted: {predicted}"
    )

    print()


accuracy = correct / len(tests) * 100

print("=============================")
print(
    f"Accuracy: {correct}/{len(tests)} "
    f"({accuracy:.1f}%)"
)
