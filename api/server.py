import os
import uuid
from pathlib import Path

import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from evaluation.answer_checker import evaluate_answer

from knowledge.adaptive_engine import AdaptiveKnowledgeEngine
from knowledge.engine import KnowledgeEngine
from knowledge.normalization import normalize_topic

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)

from memory.concept_memory import ConceptMemory


POLICY_CHECKPOINT = Path(
    "checkpoints/nextstep-action-v1.pt"
)

CONFIDENCE_THRESHOLD = 0.55

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if not POLICY_CHECKPOINT.exists():
    raise RuntimeError(
        "Missing checkpoints/nextstep-action-v1.pt"
    )


policy_data = torch.load(
    POLICY_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

policy = NextActionModel().to(device)

policy.load_state_dict(
    policy_data["model_state_dict"]
)

policy.eval()


knowledge = KnowledgeEngine()

ACTIVE_QUESTIONS = {}


app = FastAPI(
    title="NextStep AI",
    version="1.0.0",
    description=(
        "Adaptive study intelligence "
        "powered by NextStep's own learning policy."
    ),
)


class StartRequest(BaseModel):
    student_id: str = "local"
    topic: str
    exam_days: int = Field(
        default=30,
        ge=0,
        le=3650,
    )


class AnswerRequest(BaseModel):
    student_id: str = "local"
    question_id: str
    answer: str
    hint_used: bool = False


def get_memory(student_id):
    return ConceptMemory(
        student_id=student_id
    )


def prepare_topic(
    memory,
    topic,
):
    topic = normalize_topic(topic)

    concepts = knowledge.concepts_for(
        topic
    )

    if not concepts:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No knowledge map exists "
                f"for '{topic}'."
            ),
        )

    memory.ensure_topic(
        topic,
        concepts,
    )

    return topic


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
    ).unsqueeze(0).to(device)

    logits = policy(state)

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    index = torch.argmax(
        probabilities
    ).item()

    scores = {
        ACTIONS[i]:
            float(probabilities[i].item())
        for i in range(
            len(ACTIONS)
        )
    }

    return (
        ACTIONS[index],
        float(
            probabilities[index].item()
        ),
        scores,
    )


def guidance_for(
    action,
    concept_name,
    confidence,
):
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "I am not confident enough to make "
            "a strong intervention yet, so I am "
            "using a diagnostic question first."
        )

    messages = {
        "teach_foundation":
            f"Let's strengthen {concept_name} "
            f"from the foundation before moving on.",

        "guided_practice":
            f"You are ready for guided practice "
            f"on {concept_name}.",

        "correct_misconception":
            f"I detected a pattern of mistakes. "
            f"Let's isolate the misconception in "
            f"{concept_name}.",

        "retrieval_practice":
            f"This concept is at risk of being "
            f"forgotten. Let's retrieve "
            f"{concept_name} from memory.",

        "exam_priority":
            f"Your exam is close, so we are "
            f"prioritizing high-value practice "
            f"in {concept_name}.",

        "increase_difficulty":
            f"Your current level supports a harder "
            f"challenge in {concept_name}.",
    }

    return messages.get(
        action,
        f"Let's work on {concept_name}.",
    )


def mastery_map(
    memory,
    topic,
):
    states = memory.topic_progress(
        topic
    )

    return [
        {
            "concept_id":
                state.concept_id,

            "concept":
                state.concept_name,

            "mastery":
                round(
                    state.mastery,
                    4,
                ),

            "attempts":
                state.attempts,

            "mistake_rate":
                round(
                    state.mistake_rate(),
                    4,
                ),

            "forgetting_risk":
                round(
                    state.forgetting_risk(),
                    4,
                ),

            "misconceptions":
                state.misconceptions,
        }
        for state in states
    ]


def build_next_step(
    student_id,
    topic,
    exam_days,
):
    memory = get_memory(
        student_id
    )

    topic = prepare_topic(
        memory,
        topic,
    )

    (
        mastery,
        mistake_rate,
        forgetting_risk,
    ) = memory.topic_summary(
        topic
    )

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

    action_for_question = action

    if confidence < CONFIDENCE_THRESHOLD:
        action_for_question = (
            "guided_practice"
        )

    adaptive = AdaptiveKnowledgeEngine(
        memory
    )

    concept, question = (
        adaptive.generate_question(
            topic,
            action_for_question,
        )
    )

    if concept is None or question is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate the next "
                "learning question."
            ),
        )

    question_id = uuid.uuid4().hex

    ACTIVE_QUESTIONS[
        question_id
    ] = {
        "student_id": student_id,
        "topic": topic,
        "exam_days": exam_days,
        "question": question,
    }

    return {
        "student_id":
            student_id,

        "topic":
            topic,

        "overall_mastery":
            round(mastery, 4),

        "mistake_rate":
            round(
                mistake_rate,
                4,
            ),

        "forgetting_risk":
            round(
                forgetting_risk,
                4,
            ),

        "action":
            action,

        "confidence":
            round(
                confidence,
                4,
            ),

        "policy_scores":
            scores,

        "guidance":
            guidance_for(
                action,
                concept["name"],
                confidence,
            ),

        "concept": {
            "id":
                concept["id"],

            "name":
                concept["name"],

            "difficulty":
                concept["difficulty"],

            "prerequisites":
                concept.get(
                    "prerequisites",
                    [],
                ),
        },

        "question": {
            "id":
                question_id,

            "prompt":
                question.prompt,

            "difficulty":
                question.difficulty,
        },

        "mastery_map":
            mastery_map(
                memory,
                topic,
            ),
    }


@app.get(
    "/health"
)
def health():
    return {
        "status": "ok",
        "name": "NextStep AI",
        "device": str(device),
        "policy_loaded": True,
        "topics": list(
            knowledge.topics.keys()
        ),
    }


@app.post(
    "/api/start"
)
def start(
    request: StartRequest
):
    return build_next_step(
        request.student_id,
        request.topic,
        request.exam_days,
    )


@app.post(
    "/api/answer"
)
def answer(
    request: AnswerRequest
):
    active = ACTIVE_QUESTIONS.get(
        request.question_id
    )

    if active is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Question expired or unknown. "
                "Start a new learning step."
            ),
        )

    if (
        active["student_id"]
        != request.student_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Question belongs to another student.",
        )

    question = active[
        "question"
    ]

    result = evaluate_answer(
        question,
        request.answer,
    )

    memory = get_memory(
        request.student_id
    )

    topic = active["topic"]

    prepare_topic(
        memory,
        topic,
    )

    updated = memory.record_attempt(
        topic=topic,
        concept_id=question.concept,
        correct=result.correct,
        hint_used=request.hint_used,
        misconception=(
            None
            if result.correct
            else result.reason
        ),
    )

    ACTIVE_QUESTIONS.pop(
        request.question_id,
        None,
    )

    next_step = build_next_step(
        request.student_id,
        topic,
        active["exam_days"],
    )

    return {
        "correct":
            result.correct,

        "detected_issue":
            (
                None
                if result.correct
                else result.reason
            ),

        "expected":
            (
                None
                if result.correct
                else result.expected
            ),

        "updated_concept": {
            "concept":
                updated.concept_name,

            "mastery":
                round(
                    updated.mastery,
                    4,
                ),

            "attempts":
                updated.attempts,

            "mistake_rate":
                round(
                    updated.mistake_rate(),
                    4,
                ),

            "misconceptions":
                updated.misconceptions,
        },

        "next":
            next_step,
    }


@app.get(
    "/api/state/{student_id}/{topic}"
)
def state(
    student_id: str,
    topic: str,
):
    memory = get_memory(
        student_id
    )

    topic = prepare_topic(
        memory,
        topic,
    )

    (
        mastery,
        mistakes,
        forgetting,
    ) = memory.topic_summary(
        topic
    )

    return {
        "student_id":
            student_id,

        "topic":
            topic,

        "overall_mastery":
            mastery,

        "mistake_rate":
            mistakes,

        "forgetting_risk":
            forgetting,

        "mastery_map":
            mastery_map(
                memory,
                topic,
            ),
    }


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    path = Path(
        "web/index.html"
    )

    return path.read_text(
        encoding="utf-8"
    )
