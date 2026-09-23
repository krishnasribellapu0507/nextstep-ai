from pathlib import Path

import torch

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)

from memory.student_model import StudentModel

from model.transformer import (
    NextStepLM,
    ModelConfig,
)

from tokenizer.byte_tokenizer import ByteTokenizer


POLICY_CHECKPOINT = Path(
    "checkpoints/nextstep-action-v1.pt"
)

LANGUAGE_CHECKPOINT = Path(
    "checkpoints/nextstep-v1-best.pt"
)

CONFIDENCE_THRESHOLD = 0.55

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = ByteTokenizer()

student_model = StudentModel()


# =========================================================
# LOAD POLICY BRAIN
# =========================================================

policy_data = torch.load(
    POLICY_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

policy_model = NextActionModel().to(device)

policy_model.load_state_dict(
    policy_data["model_state_dict"]
)

policy_model.eval()


# =========================================================
# LOAD LANGUAGE BRAIN
# =========================================================

language_data = torch.load(
    LANGUAGE_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(
    **language_data["config"]
)

language_model = NextStepLM(
    config
).to(device)

language_model.load_state_dict(
    language_data["model_state_dict"]
)

language_model.eval()


# =========================================================
# POLICY DECISION
# =========================================================

@torch.no_grad()
def choose_action(
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
):

    state = state_tensor(
        mastery,
        mistake_rate,
        forgetting_risk,
        exam_days,
    )

    state = state.unsqueeze(0).to(device)

    logits = policy_model(state)

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    best_index = torch.argmax(
        probabilities
    ).item()

    action = ACTIONS[best_index]

    confidence = probabilities[
        best_index
    ].item()

    return (
        action,
        confidence,
        probabilities,
    )


# =========================================================
# LANGUAGE GENERATION
# =========================================================

@torch.no_grad()
def generate_tutor_response(
    message,
    topic,
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
    action,
):

    prompt = f"""<STUDENT>
{message}

<STATE>
topic={topic}
mastery={mastery:.2f}
mistake_rate={mistake_rate:.2f}
forgetting_risk={forgetting_risk:.2f}
exam_days={exam_days}

<ACTION>
{action}

<TUTOR>
"""

    ids = tokenizer.encode(prompt)

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    output = language_model.generate(
        x,
        max_new_tokens=120,
        temperature=0.35,
        top_k=5,
    )

    generated = output[
        0,
        len(ids):
    ].tolist()

    response = tokenizer.decode(
        generated
    )

    for marker in [
        "<END>",
        "<STUDENT>",
        "<STATE>",
        "<ACTION>",
    ]:
        if marker in response:
            response = response.split(
                marker
            )[0]

    return response.strip()


# =========================================================
# DISPLAY STUDENT STATE
# =========================================================

def print_student_state(progress):

    print()
    print("STUDENT MODEL")
    print("=============")

    print(
        f"Topic:           {progress.topic}"
    )

    print(
        f"Mastery:         "
        f"{progress.mastery:.0%}"
    )

    print(
        f"Mistake rate:    "
        f"{progress.mistake_rate():.0%}"
    )

    print(
        f"Forgetting risk: "
        f"{progress.forgetting_risk():.0%}"
    )

    print(
        f"Attempts:        "
        f"{progress.attempts}"
    )


# =========================================================
# MAIN NEXTSTEP LOOP
# =========================================================

def run_nextstep():

    print()
    print("NEXTSTEP AI V3")
    print("==============")
    print(
        "Adaptive learning with persistent "
        "student memory"
    )

    print()

    topic = input(
        "What topic are you studying? "
    ).strip()

    exam_days = int(
        input(
            "Days until exam: "
        )
    )

    progress = student_model.get_topic(
        topic
    )

    while True:

        print_student_state(
            progress
        )

        message = input(
            "\nTell NextStep what is happening: "
        ).strip()

        if message.lower() in {
            "quit",
            "exit",
            "stop",
        }:
            print(
                "\nProgress saved. See you next time."
            )
            break


        mastery = progress.mastery

        mistake_rate = (
            progress.mistake_rate()
        )

        forgetting_risk = (
            progress.forgetting_risk()
        )


        (
            action,
            confidence,
            probabilities,
        ) = choose_action(
            mastery,
            mistake_rate,
            forgetting_risk,
            exam_days,
        )


        print()
        print("NEXTSTEP DECISION")
        print("=================")

        print(
            f"Action:     {action}"
        )

        print(
            f"Confidence: {confidence:.1%}"
        )


        if confidence < CONFIDENCE_THRESHOLD:

            print()
            print(
                "I need a little more evidence "
                "before choosing your next step."
            )

            print(
                "Try one question from this topic "
                "without notes."
            )

        else:

            response = generate_tutor_response(
                message,
                topic,
                mastery,
                mistake_rate,
                forgetting_risk,
                exam_days,
                action,
            )

            print()
            print("NEXTSTEP")
            print("========")

            print(response)


        # =================================================
        # RECORD LEARNING RESULT
        # =================================================

        print()
        print("ATTEMPT RESULT")
        print("==============")

        result = input(
            "Was your latest attempt correct? "
            "(y/n/skip): "
        ).strip().lower()

        if result == "skip":
            continue

        if result not in {
            "y",
            "n",
        }:
            print(
                "Result not recorded."
            )
            continue


        hint = input(
            "Did you use a hint? (y/n): "
        ).strip().lower()

        correct = result == "y"

        hint_used = hint == "y"


        progress = student_model.record_attempt(
            topic,
            correct=correct,
            hint_used=hint_used,
        )


        print()
        print("PROGRESS UPDATED")
        print("================")

        print(
            f"New mastery: "
            f"{progress.mastery:.0%}"
        )

        print(
            f"Mistake rate: "
            f"{progress.mistake_rate():.0%}"
        )


if __name__ == "__main__":
    run_nextstep()
