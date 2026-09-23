from dataclasses import dataclass


@dataclass
class Question:
    topic: str
    prompt: str
    answer_type: str
    expected: object
    misconception: str
    difficulty: int = 1


QUESTIONS = [

    Question(
        topic="eigenvalues",
        prompt=(
            "Find the eigenvalues of the matrix "
            "[[2, 0], [0, 3]]. "
            "Enter them separated by commas."
        ),
        answer_type="number_set",
        expected=[2, 3],
        misconception="basic_eigenvalue_identification",
        difficulty=1,
    ),

    Question(
        topic="eigenvalues",
        prompt=(
            "For A = [[4, 0], [0, 1]], "
            "what are the eigenvalues?"
        ),
        answer_type="number_set",
        expected=[4, 1],
        misconception="basic_eigenvalue_identification",
        difficulty=1,
    ),

    Question(
        topic="eigenvalues",
        prompt=(
            "For A = [[2, 1], [0, 2]], "
            "what is the repeated eigenvalue?"
        ),
        answer_type="number_set",
        expected=[2],
        misconception="repeated_eigenvalue",
        difficulty=2,
    ),

    Question(
        topic="determinants",
        prompt=(
            "Find det([[2, 3], [1, 4]])."
        ),
        answer_type="number",
        expected=5,
        misconception="determinant_expansion",
        difficulty=1,
    ),

]
