import random
from dataclasses import dataclass


@dataclass
class GeneratedQuestion:
    topic: str
    concept: str
    prompt: str
    answer_type: str
    expected: object
    misconception: str
    difficulty: int


class QuestionGenerator:

    def determinant_2x2(self):
        a = random.randint(1, 7)
        b = random.randint(1, 7)
        c = random.randint(1, 7)
        d = random.randint(1, 7)

        answer = a * d - b * c

        return GeneratedQuestion(
            topic="eigenvalues",
            concept="determinant_2x2",
            prompt=f"Find the determinant of [[{a}, {b}], [{c}, {d}]].",
            answer_type="number",
            expected=answer,
            misconception="determinant_2x2_error",
            difficulty=1,
        )

    def diagonal_eigenvalues(self):
        a = random.randint(-5, 7)
        d = random.randint(-5, 7)

        return GeneratedQuestion(
            topic="eigenvalues",
            concept="diagonal_eigenvalues",
            prompt=(
                f"Find the eigenvalues of "
                f"A = [[{a}, 0], [0, {d}]]. "
                f"Enter them separated by commas."
            ),
            answer_type="number_set",
            expected=[a, d],
            misconception="diagonal_entries_not_recognized",
            difficulty=2,
        )

    def triangular_eigenvalues(self):
        a = random.randint(-5, 7)
        b = random.randint(1, 6)
        d = random.randint(-5, 7)

        return GeneratedQuestion(
            topic="eigenvalues",
            concept="triangular_eigenvalues",
            prompt=(
                f"Find the eigenvalues of the upper triangular matrix "
                f"A = [[{a}, {b}], [0, {d}]]."
            ),
            answer_type="number_set",
            expected=[a, d],
            misconception="triangular_eigenvalue_error",
            difficulty=3,
        )

    def general_2x2_eigenvalues(self):
        m = random.randint(-4, 4)
        n = random.choice([-3, -2, -1, 1, 2, 3])

        a = m + n
        b = n
        c = n
        d = m + n

        return GeneratedQuestion(
            topic="eigenvalues",
            concept="general_2x2_eigenvalues",
            prompt=(
                f"Find the eigenvalues of "
                f"A = [[{a}, {b}], [{c}, {d}]]."
            ),
            answer_type="number_set",
            expected=[m, m + 2 * n],
            misconception="characteristic_polynomial_error",
            difficulty=4,
        )

    def generate(self, concept_id: str):
        generators = {
            "determinant_2x2": self.determinant_2x2,
            "diagonal_eigenvalues": self.diagonal_eigenvalues,
            "triangular_eigenvalues": self.triangular_eigenvalues,
            "general_2x2_eigenvalues": self.general_2x2_eigenvalues,
        }

        generator = generators.get(concept_id)

        if generator is None:
            raise ValueError(
                f"No question generator exists for concept: {concept_id}"
            )

        return generator()
