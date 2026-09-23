import torch

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)


CHECKPOINT = "checkpoints/nextstep-action-v1.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)


model = NextActionModel().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


tests = [
    {
        "name": "Very weak beginner",
        "mastery": 0.12,
        "mistakes": 0.25,
        "forgetting": 0.15,
        "exam_days": 14,
        "expected": "teach_foundation",
    },

    {
        "name": "Moderate understanding",
        "mastery": 0.55,
        "mistakes": 0.30,
        "forgetting": 0.25,
        "exam_days": 8,
        "expected": "guided_practice",
    },

    {
        "name": "Repeated misconception",
        "mastery": 0.58,
        "mistakes": 0.91,
        "forgetting": 0.20,
        "exam_days": 5,
        "expected": "correct_misconception",
    },

    {
        "name": "Forgotten old topic",
        "mastery": 0.72,
        "mistakes": 0.16,
        "forgetting": 0.93,
        "exam_days": 12,
        "expected": "retrieval_practice",
    },

    {
        "name": "Exam tomorrow and weak",
        "mastery": 0.39,
        "mistakes": 0.41,
        "forgetting": 0.45,
        "exam_days": 1,
        "expected": "exam_priority",
    },

    {
        "name": "Ready for challenge",
        "mastery": 0.91,
        "mistakes": 0.08,
        "forgetting": 0.10,
        "exam_days": 20,
        "expected": "increase_difficulty",
    },

    {
        "name": "Weak but many mistakes",
        "mastery": 0.24,
        "mistakes": 0.82,
        "forgetting": 0.20,
        "exam_days": 10,
        "expected": "correct_misconception",
    },

    {
        "name": "Exam tomorrow but strong",
        "mastery": 0.86,
        "mistakes": 0.12,
        "forgetting": 0.20,
        "exam_days": 1,
        "expected": "increase_difficulty",
    },

    {
        "name": "Needs revision",
        "mastery": 0.60,
        "mistakes": 0.20,
        "forgetting": 0.85,
        "exam_days": 7,
        "expected": "retrieval_practice",
    },

    {
        "name": "Almost understands",
        "mastery": 0.62,
        "mistakes": 0.34,
        "forgetting": 0.30,
        "exam_days": 10,
        "expected": "guided_practice",
    },
]


correct = 0


print()
print("NEXTSTEP POLICY TEST")
print("====================")
print()


for i, test in enumerate(tests, 1):

    state = state_tensor(
        test["mastery"],
        test["mistakes"],
        test["forgetting"],
        test["exam_days"],
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(state)

        probs = torch.softmax(
            logits,
            dim=-1,
        )[0]

    index = torch.argmax(probs).item()

    predicted = ACTIONS[index]
    confidence = probs[index].item()

    passed = predicted == test["expected"]

    if passed:
        correct += 1

    print(
        f"{i:02d}. "
        f"{'PASS' if passed else 'FAIL'} "
        f"- {test['name']}"
    )

    print(
        f"    Expected:   {test['expected']}"
    )

    print(
        f"    Predicted:  {predicted}"
    )

    print(
        f"    Confidence: {confidence:.1%}"
    )

    print()


accuracy = correct / len(tests)


print("====================")
print(
    f"Independent accuracy: "
    f"{correct}/{len(tests)} "
    f"({accuracy:.1%})"
)
