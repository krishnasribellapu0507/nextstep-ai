import random
from pathlib import Path


random.seed(42)

OUT = Path("data/v1")
OUT.mkdir(parents=True, exist_ok=True)


topics = [
    "matrices",
    "determinants",
    "eigenvalues",
    "limits",
    "differentiation",
    "probability",
    "pn junction",
    "rectifiers",
    "transistors",
    "loops in python",
    "functions in python",
    "data structures",
]


student_messages = [
    "I am struggling with {topic}.",
    "I keep making mistakes in {topic}.",
    "I studied {topic}, but I am not confident.",
    "What should I do next for {topic}?",
    "I do not understand {topic} properly.",
    "I need help preparing {topic}.",
]


def decide_action(mastery, mistakes, forgetting, exam_days):
    if exam_days <= 1 and mastery < 0.60:
        return "exam_priority"

    if mistakes >= 0.70:
        return "correct_misconception"

    if forgetting >= 0.70 and mastery >= 0.45:
        return "retrieval_practice"

    if mastery < 0.35:
        return "teach_foundation"

    if mastery < 0.65:
        return "guided_practice"

    return "increase_difficulty"


def tutor_response(action, topic):
    responses = {
        "exam_priority":
            f"Your exam is very close, so we should focus on the highest-value parts of {topic}, then immediately test recall.",

        "correct_misconception":
            f"You are repeatedly making mistakes in {topic}. The next step is to identify the exact misconception before doing more questions.",

        "retrieval_practice":
            f"You have learned {topic} before, but forgetting risk is high. Do not reread yet. First try to recall the main idea from memory.",

        "teach_foundation":
            f"Your mastery of {topic} is still low. The next step is a short foundation lesson followed by one simple check question.",

        "guided_practice":
            f"You understand part of {topic}. The next step is one guided example followed by an independent problem.",

        "increase_difficulty":
            f"Your mastery of {topic} is strong enough to move forward. The next step is a harder problem that tests deeper understanding.",
    }

    return responses[action]


examples = []

for _ in range(4000):
    topic = random.choice(topics)

    mastery = round(random.uniform(0.10, 0.95), 2)
    mistakes = round(random.uniform(0.05, 0.95), 2)
    forgetting = round(random.uniform(0.05, 0.95), 2)

    exam_days = random.choice([
        1, 2, 3, 5, 7, 14, 30
    ])

    message = random.choice(
        student_messages
    ).format(topic=topic)

    action = decide_action(
        mastery,
        mistakes,
        forgetting,
        exam_days,
    )

    response = tutor_response(
        action,
        topic,
    )

    example = f"""<STUDENT>
{message}

<STATE>
topic={topic}
mastery={mastery:.2f}
mistake_rate={mistakes:.2f}
forgetting_risk={forgetting:.2f}
exam_days={exam_days}

<ACTION>
{action}

<TUTOR>
{response}

<END>

"""

    examples.append(example)


random.shuffle(examples)

split = int(len(examples) * 0.85)

train_examples = examples[:split]
validation_examples = examples[split:]


(OUT / "train.txt").write_text(
    "".join(train_examples),
    encoding="utf-8",
)

(OUT / "validation.txt").write_text(
    "".join(validation_examples),
    encoding="utf-8",
)


print("NEXTSTEP V1 DATASET")
print("===================")
print("Total examples:", len(examples))
print("Training:", len(train_examples))
print("Validation:", len(validation_examples))

print()
print("Saved:")
print("data/v1/train.txt")
print("data/v1/validation.txt")
