"""
Logic Reasoning Arena - A diverse reasoning environment for continuous learning

This module creates challenging reasoning tasks that test:
1. Logical deduction
2. Mathematical reasoning
3. Sequential pattern recognition
4. Causal reasoning
"""

import random
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ReasoningTask:
    """A single reasoning task with question and answer"""
    task_type: str
    question: str
    answer: str
    difficulty: int  # 1-5
    explanation: str


class LogicReasoningArena:
    """Generates diverse reasoning challenges"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_generators = {
            'comparison': self._generate_comparison_task,
            'sequence': self._generate_sequence_task,
            'causal': self._generate_causal_task,
            'math_word': self._generate_math_word_task,
            'logic_grid': self._generate_logic_grid_task,
        }

    def generate_task(self, task_type: str = None) -> ReasoningTask:
        """Generate a random reasoning task"""
        if task_type is None:
            task_type = random.choice(list(self.task_generators.keys()))
        return self.task_generators[task_type]()

    def generate_batch(self, n: int = 10, mix: bool = True) -> List[ReasoningTask]:
        """Generate a batch of reasoning tasks"""
        tasks = []
        for _ in range(n):
            task_type = None if mix else random.choice(list(self.task_generators.keys()))
            tasks.append(self.generate_task(task_type))
        return tasks

    def _generate_comparison_task(self) -> ReasoningTask:
        """Generate comparison reasoning task (A > B > C type)"""
        names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
        attributes = [
            ('tall', 'taller', 'tallest', 'shortest'),
            ('old', 'older', 'oldest', 'youngest'),
            ('fast', 'faster', 'fastest', 'slowest'),
            ('smart', 'smarter', 'smartest', 'least smart'),
        ]

        n_people = random.randint(3, 4)
        people = random.sample(names, n_people)
        attr, comp, superlative, opposite = random.choice(attributes)

        # Create ordering
        ordering = people.copy()
        random.shuffle(ordering)

        # Generate comparisons
        comparisons = []
        for i in range(len(ordering) - 1):
            comparisons.append(f"{ordering[i]} is {comp} than {ordering[i+1]}")

        random.shuffle(comparisons)

        # Generate question
        question_types = [
            (f"Who is the {superlative}?", ordering[0]),
            (f"Who is the {opposite}?", ordering[-1]),
        ]

        if len(ordering) >= 3:
            question_types.append((f"Who is {comp}, {ordering[0]} or {ordering[-1]}?", ordering[0]))

        q_text, answer = random.choice(question_types)

        question = f"{' '.join(comparisons)}. {q_text}"

        return ReasoningTask(
            task_type='comparison',
            question=question,
            answer=answer,
            difficulty=len(ordering),
            explanation=f"Ordering: {' > '.join(ordering)}"
        )

    def _generate_sequence_task(self) -> ReasoningTask:
        """Generate sequence pattern recognition task"""
        patterns = [
            # Arithmetic sequences
            lambda x: x + 2,
            lambda x: x + 3,
            lambda x: x * 2,
            lambda x: x * 2 + 1,
            # Fibonacci-like
        ]

        pattern_type = random.randint(0, 3)

        if pattern_type == 0:  # Arithmetic
            start = random.randint(1, 10)
            diff = random.choice([2, 3, 5, 7])
            seq = [start + i * diff for i in range(5)]
            answer = str(seq[-1] + diff)
            explanation = f"Arithmetic sequence with difference {diff}"

        elif pattern_type == 1:  # Geometric
            start = random.randint(2, 5)
            ratio = random.choice([2, 3])
            seq = [start * (ratio ** i) for i in range(4)]
            answer = str(seq[-1] * ratio)
            explanation = f"Geometric sequence with ratio {ratio}"

        elif pattern_type == 2:  # Fibonacci
            seq = [1, 1]
            for i in range(3):
                seq.append(seq[-1] + seq[-2])
            answer = str(seq[-1] + seq[-2])
            explanation = "Fibonacci sequence"

        else:  # Alternating
            seq = [random.randint(1, 5)]
            op1, op2 = random.choice([('+2', '+3'), ('*2', '+1'), ('+5', '-1')])
            for i in range(4):
                if i % 2 == 0:
                    seq.append(eval(f"{seq[-1]}{op1}"))
                else:
                    seq.append(eval(f"{seq[-1]}{op2}"))
            answer = str(eval(f"{seq[-1]}{op1}") if len(seq) % 2 == 1 else eval(f"{seq[-1]}{op2}"))
            explanation = f"Alternating pattern: {op1}, {op2}"

        seq_str = ', '.join(map(str, seq))
        question = f"What comes next in this sequence: {seq_str}, ?"

        return ReasoningTask(
            task_type='sequence',
            question=question,
            answer=answer,
            difficulty=3,
            explanation=explanation
        )

    def _generate_causal_task(self) -> ReasoningTask:
        """Generate causal reasoning task"""
        scenarios = [
            {
                'setup': "If it rains, the ground gets wet. If the ground is wet, plants grow. It rained yesterday.",
                'question': "Did plants grow?",
                'answer': "Yes",
                'explanation': "Rain → wet ground → plants grow"
            },
            {
                'setup': "The alarm rings when motion is detected. Motion was detected at 3am. The dog was outside.",
                'question': "Did the alarm ring?",
                'answer': "Yes",
                'explanation': "Motion detected → alarm rings"
            },
            {
                'setup': "Coffee makes John alert. John is tired. John did not drink coffee.",
                'question': "Is John alert?",
                'answer': "No",
                'explanation': "No coffee → not alert (still tired)"
            },
            {
                'setup': "Studying improves test scores. Alice studied for 10 hours. Bob did not study.",
                'question': "Who likely scored higher?",
                'answer': "Alice",
                'explanation': "Studying → better scores"
            },
        ]

        scenario = random.choice(scenarios)

        return ReasoningTask(
            task_type='causal',
            question=f"{scenario['setup']} {scenario['question']}",
            answer=scenario['answer'],
            difficulty=2,
            explanation=scenario['explanation']
        )

    def _generate_math_word_task(self) -> ReasoningTask:
        """Generate math word problem"""
        problem_types = [
            # Addition
            lambda: {
                'q': f"Alice has {(a := random.randint(5, 20))} apples. Bob gives her {(b := random.randint(3, 10))} more. How many does she have?",
                'a': str(a + b),
                'e': f"{a} + {b} = {a + b}"
            },
            # Subtraction
            lambda: {
                'q': f"There are {(a := random.randint(20, 50))} birds. {(b := random.randint(5, 15))} fly away. How many remain?",
                'a': str(a - b),
                'e': f"{a} - {b} = {a - b}"
            },
            # Multiplication
            lambda: {
                'q': f"Each box has {(a := random.randint(3, 8))} items. There are {(b := random.randint(4, 10))} boxes. How many items total?",
                'a': str(a * b),
                'e': f"{a} × {b} = {a * b}"
            },
        ]

        problem = random.choice(problem_types)()

        return ReasoningTask(
            task_type='math_word',
            question=problem['q'],
            answer=problem['a'],
            difficulty=2,
            explanation=problem['e']
        )

    def _generate_logic_grid_task(self) -> ReasoningTask:
        """Generate logic grid puzzle"""
        people = random.sample(['Alice', 'Bob', 'Charlie'], 3)
        colors = random.sample(['red', 'blue', 'green'], 3)

        # Assign colors to people
        assignment = dict(zip(people, colors))

        # Create clues
        clues = []
        clues.append(f"{people[0]} likes {colors[0]}")
        clues.append(f"{people[1]} does not like {colors[0]}")

        # Question: what color does person[2] like?
        answer = assignment[people[2]]

        question = f"{'. '.join(clues)}. What color does {people[2]} like?"

        return ReasoningTask(
            task_type='logic_grid',
            question=question,
            answer=answer,
            difficulty=3,
            explanation=f"Assignment: {assignment}"
        )


if __name__ == "__main__":
    # Test the arena
    arena = LogicReasoningArena()

    print("=== Logic Reasoning Arena Test ===\n")
    for task_type in ['comparison', 'sequence', 'causal', 'math_word', 'logic_grid']:
        task = arena.generate_task(task_type)
        print(f"Type: {task.task_type}")
        print(f"Q: {task.question}")
        print(f"A: {task.answer}")
        print(f"Difficulty: {task.difficulty}")
        print(f"Explanation: {task.explanation}")
        print("-" * 60)

    print("\n=== Random Batch Test ===\n")
    batch = arena.generate_batch(5, mix=True)
    for i, task in enumerate(batch, 1):
        print(f"{i}. [{task.task_type}] {task.question}")
        print(f"   Answer: {task.answer}\n")
