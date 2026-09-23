import torch
import torch.nn as nn

ACTIONS = [
    "teach_foundation",
    "guided_practice",
    "correct_misconception",
    "retrieval_practice",
    "exam_priority",
    "increase_difficulty",
]


class NextActionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, len(ACTIONS)),
        )

    def forward(self, state):
        return self.network(state)


def state_tensor(
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
):
    exam_value = min(exam_days, 30) / 30.0

    return torch.tensor(
        [
            mastery,
            mistake_rate,
            forgetting_risk,
            exam_value,
        ],
        dtype=torch.float32,
    )
