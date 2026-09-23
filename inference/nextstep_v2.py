from pathlib import Path

import torch

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)

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
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD NEXT ACTION BRAIN
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


tokenizer = ByteTokenizer()


# =========================================================
# CHOOSE NEXT ACTION
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

    index = torch.argmax(
        probabilities
    ).item()

    action = ACTIONS[index]

    confidence = probabilities[
        index
    ].item()

    scores = {
        ACTIONS[i]:
            probabilities[i].item()
        for i in range(len(ACTIONS))
    }

    return (
        action,
        confidence,
        scores,
    )


# =========================================================
# GENERATE TUTOR RESPONSE
# =========================================================

@torch.no_grad()
def generate_response(
    student_message,
    topic,
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
    action,
):

    prompt = f"""<STUDENT>
{student_message}

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

    input_ids = tokenizer.encode(prompt)

    x = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    output = language_model.generate(
        x,
        max_new_tokens=140,
        temperature=0.35,
        top_k=5,
    )

    new_tokens = output[
        0,
        len(input_ids):
    ].tolist()

    response = tokenizer.decode(
        new_tokens
    )

    # Stop if another structured section starts.
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
# NEXTSTEP CORE
# =========================================================

def nextstep(
    student_message,
    topic,
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
):

    (
        action,
        confidence,
        scores,
    ) = choose_action(
        mastery,
        mistake_rate,
        forgetting_risk,
        exam_days,
    )

    print()
    print("NEXTSTEP ANALYSIS")
    print("=================")

    print(
        f"Topic:       {topic}"
    )

    print(
        f"Mastery:     {mastery:.0%}"
    )

    print(
        f"Next action: {action}"
    )

    print(
        f"Confidence:  {confidence:.1%}"
    )

    print()
    print("Policy probabilities:")

    for name, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {name:24} "
            f"{score:.1%}"
        )


    # -----------------------------------------------------
    # UNCERTAINTY HANDLING
    # -----------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print()
        print("NEXTSTEP")
        print("========")

        print(
            "I am not confident enough to choose "
            "your next learning action yet."
        )

        print(
            "Diagnostic question: "
            "Can you try one problem from this topic "
            "without notes and tell me exactly where "
            "you get stuck?"
        )

        return


    # -----------------------------------------------------
    # LANGUAGE GENERATION
    # -----------------------------------------------------

    response = generate_response(
        student_message,
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


# =========================================================
# INTERACTIVE TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("NEXTSTEP AI V2")
    print("==============")

    student_message = input(
        "What are you struggling with? "
    )

    topic = input(
        "Topic: "
    )

    mastery = float(
        input(
            "Current mastery 0-100: "
        )
    ) / 100

    mistake_rate = float(
        input(
            "Recent mistake rate 0-100: "
        )
    ) / 100

    forgetting_risk = float(
        input(
            "Forgetting risk 0-100: "
        )
    ) / 100

    exam_days = int(
        input(
            "Days until exam: "
        )
    )

    nextstep(
        student_message,
        topic,
        mastery,
        mistake_rate,
        forgetting_risk,
        exam_days,
    )
