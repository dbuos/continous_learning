"""
Advanced Reasoning Arena - More Challenging Environment

Inspired by GSM8K, MATH, ARC, and recent reasoning benchmarks.
Includes:
- Multi-step reasoning problems
- Difficulty levels (1-5)
- Curriculum learning support
- More diverse problem types
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ReasoningTask:
    """A reasoning task with metadata"""
    task_type: str
    question: str
    answer: str
    difficulty: int  # 1=easy, 5=very hard
    explanation: str
    num_steps: int  # Number of reasoning steps required


class AdvancedReasoningArena:
    """
    Advanced reasoning arena with challenging multi-step problems
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_generators = {
            'multi_step_math': self._generate_multi_step_math,
            'algebraic_reasoning': self._generate_algebraic_reasoning,
            'logical_deduction': self._generate_logical_deduction,
            'counterfactual': self._generate_counterfactual,
            'analogy': self._generate_analogy,
            'code_reasoning': self._generate_code_reasoning,
            'probability': self._generate_probability,
            'geometry': self._generate_geometry,
        }

    def generate_task(
        self,
        task_type: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> ReasoningTask:
        """Generate a reasoning task with optional constraints"""
        if task_type is None:
            task_type = random.choice(list(self.task_generators.keys()))

        task = self.task_generators[task_type]()

        # Filter by difficulty if specified
        if difficulty is not None:
            while task.difficulty != difficulty:
                task = self.task_generators[task_type]()

        return task

    def generate_curriculum(
        self,
        n_tasks: int = 100,
        start_difficulty: int = 1,
        end_difficulty: int = 5,
    ) -> List[ReasoningTask]:
        """
        Generate tasks with curriculum learning
        Difficulty gradually increases
        """
        tasks = []
        for i in range(n_tasks):
            # Linearly increase difficulty
            progress = i / n_tasks
            target_difficulty = int(start_difficulty + (end_difficulty - start_difficulty) * progress)
            target_difficulty = max(1, min(5, target_difficulty))

            task = self.generate_task(difficulty=target_difficulty)
            tasks.append(task)

        return tasks

    def generate_batch(
        self,
        n: int = 10,
        mix: bool = True,
        difficulty_range: Tuple[int, int] = (1, 5),
    ) -> List[ReasoningTask]:
        """Generate a batch of tasks"""
        tasks = []
        for _ in range(n):
            task_type = None if mix else random.choice(list(self.task_generators.keys()))
            task = self.generate_task(task_type)

            # Retry if outside difficulty range
            while task.difficulty < difficulty_range[0] or task.difficulty > difficulty_range[1]:
                task = self.generate_task(task_type)

            tasks.append(task)
        return tasks

    # ============= Task Generators =============

    def _generate_multi_step_math(self) -> ReasoningTask:
        """
        Multi-step math word problems (inspired by GSM8K)
        Requires 2-4 steps of arithmetic
        """
        templates = [
            # Level 2-3: Two-step problems
            lambda: self._gen_two_step_purchase(),
            lambda: self._gen_age_problem(),
            lambda: self._gen_distance_time(),

            # Level 3-4: Three-step problems
            lambda: self._gen_profit_loss(),
            lambda: self._gen_work_rate(),

            # Level 4-5: Four-step problems
            lambda: self._gen_complex_ratio(),
        ]

        generator = random.choice(templates)
        return generator()

    def _gen_two_step_purchase(self) -> ReasoningTask:
        """Two-step purchase problem"""
        item = random.choice(['apples', 'books', 'pens', 'notebooks'])
        price = random.randint(3, 15)
        quantity = random.randint(5, 20)
        discount = random.randint(10, 30)

        total_before = price * quantity
        discount_amount = total_before * discount // 100
        final_price = total_before - discount_amount

        question = f"Each {item[:-1]} costs ${price}. John buys {quantity} {item}. " \
                   f"He gets a {discount}% discount. How much does he pay in total?"

        explanation = f"Step 1: {quantity} × ${price} = ${total_before}\n" \
                      f"Step 2: {discount}% of ${total_before} = ${discount_amount}\n" \
                      f"Step 3: ${total_before} - ${discount_amount} = ${final_price}"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=str(final_price),
            difficulty=3,
            explanation=explanation,
            num_steps=3
        )

    def _gen_age_problem(self) -> ReasoningTask:
        """Age reasoning problem"""
        age_a = random.randint(20, 40)
        diff = random.randint(5, 15)
        age_b = age_a + diff

        question = f"Alice is {age_a} years old. Bob is {diff} years older than Alice. " \
                   f"What will be the sum of their ages in 5 years?"

        future_a = age_a + 5
        future_b = age_b + 5
        total = future_a + future_b

        explanation = f"Step 1: Bob's current age = {age_a} + {diff} = {age_b}\n" \
                      f"Step 2: Alice in 5 years = {age_a} + 5 = {future_a}\n" \
                      f"Step 3: Bob in 5 years = {age_b} + 5 = {future_b}\n" \
                      f"Step 4: Sum = {future_a} + {future_b} = {total}"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=str(total),
            difficulty=2,
            explanation=explanation,
            num_steps=4
        )

    def _gen_distance_time(self) -> ReasoningTask:
        """Distance, speed, time problem"""
        speed1 = random.randint(40, 80)
        speed2 = random.randint(40, 80)
        time = random.randint(2, 5)

        dist1 = speed1 * time
        dist2 = speed2 * time
        total = dist1 + dist2

        question = f"Two cars start from the same point and travel in opposite directions. " \
                   f"Car A travels at {speed1} mph and Car B at {speed2} mph. " \
                   f"How far apart are they after {time} hours?"

        explanation = f"Step 1: Car A distance = {speed1} × {time} = {dist1} miles\n" \
                      f"Step 2: Car B distance = {speed2} × {time} = {dist2} miles\n" \
                      f"Step 3: Total distance = {dist1} + {dist2} = {total} miles"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=str(total),
            difficulty=3,
            explanation=explanation,
            num_steps=3
        )

    def _gen_profit_loss(self) -> ReasoningTask:
        """Profit/loss calculation"""
        cost = random.randint(50, 200)
        markup = random.randint(20, 50)
        discount = random.randint(10, 30)

        selling_before_discount = cost + (cost * markup // 100)
        discount_amount = selling_before_discount * discount // 100
        final_selling = selling_before_discount - discount_amount
        profit = final_selling - cost

        question = f"A shopkeeper buys an item for ${cost}. He marks it up by {markup}% " \
                   f"but then gives a {discount}% discount. What is his profit?"

        explanation = f"Step 1: Marked price = ${cost} + {markup}% = ${selling_before_discount}\n" \
                      f"Step 2: Discount = {discount}% of ${selling_before_discount} = ${discount_amount}\n" \
                      f"Step 3: Selling price = ${selling_before_discount} - ${discount_amount} = ${final_selling}\n" \
                      f"Step 4: Profit = ${final_selling} - ${cost} = ${profit}"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=str(profit),
            difficulty=4,
            explanation=explanation,
            num_steps=4
        )

    def _gen_work_rate(self) -> ReasoningTask:
        """Work rate problem"""
        days_a = random.choice([6, 8, 10, 12])
        days_b = random.choice([8, 10, 12, 15])

        # Work rate: A does 1/days_a per day, B does 1/days_b per day
        # Together: 1/days_a + 1/days_b = combined rate
        # Days = 1 / combined_rate

        combined_rate_num = days_a + days_b
        combined_rate_den = days_a * days_b
        days_together = combined_rate_den / combined_rate_num

        # Simplify for clean answer
        gcd = math.gcd(combined_rate_den, combined_rate_num)
        days_together_num = combined_rate_den // gcd
        days_together_den = combined_rate_num // gcd

        question = f"Worker A can complete a job in {days_a} days. Worker B can complete " \
                   f"the same job in {days_b} days. How many days will it take if they work together?"

        if days_together_num % days_together_den == 0:
            answer = str(days_together_num // days_together_den)
            explanation = f"Step 1: A's rate = 1/{days_a} per day\n" \
                          f"Step 2: B's rate = 1/{days_b} per day\n" \
                          f"Step 3: Combined rate = 1/{days_a} + 1/{days_b} = {combined_rate_num}/{combined_rate_den}\n" \
                          f"Step 4: Days = 1 ÷ ({combined_rate_num}/{combined_rate_den}) = {answer}"
        else:
            answer = f"{days_together_num}/{days_together_den}"
            explanation = f"Combined rate = 1/{days_a} + 1/{days_b}\nDays together = {answer}"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=answer,
            difficulty=4,
            explanation=explanation,
            num_steps=4
        )

    def _gen_complex_ratio(self) -> ReasoningTask:
        """Complex ratio problem"""
        ratio_a = random.randint(2, 5)
        ratio_b = random.randint(2, 5)
        total = random.randint(50, 150)

        # Make sure total is divisible
        sum_ratio = ratio_a + ratio_b
        total = (total // sum_ratio) * sum_ratio

        share_a = total * ratio_a // sum_ratio
        share_b = total * ratio_b // sum_ratio
        diff = abs(share_a - share_b)

        question = f"${total} is divided between Alice and Bob in the ratio {ratio_a}:{ratio_b}. " \
                   f"How much more does the person with the larger share get?"

        explanation = f"Step 1: Total parts = {ratio_a} + {ratio_b} = {sum_ratio}\n" \
                      f"Step 2: Alice's share = {ratio_a}/{sum_ratio} × ${total} = ${share_a}\n" \
                      f"Step 3: Bob's share = {ratio_b}/{sum_ratio} × ${total} = ${share_b}\n" \
                      f"Step 4: Difference = ${diff}"

        return ReasoningTask(
            task_type='multi_step_math',
            question=question,
            answer=str(diff),
            difficulty=4,
            explanation=explanation,
            num_steps=4
        )

    def _generate_algebraic_reasoning(self) -> ReasoningTask:
        """Algebraic reasoning (solve for x)"""
        # Simple: ax + b = c
        a = random.randint(2, 10)
        b = random.randint(5, 30)
        x = random.randint(1, 20)
        c = a * x + b

        question = f"Solve for x: {a}x + {b} = {c}"
        explanation = f"Step 1: {a}x = {c} - {b} = {c - b}\n" \
                      f"Step 2: x = {c - b} / {a} = {x}"

        return ReasoningTask(
            task_type='algebraic_reasoning',
            question=question,
            answer=str(x),
            difficulty=2,
            explanation=explanation,
            num_steps=2
        )

    def _generate_logical_deduction(self) -> ReasoningTask:
        """Multi-premise logical deduction"""
        names = ['Alice', 'Bob', 'Charlie', 'Diana']
        selected = random.sample(names, 3)

        # Create a complex ordering with multiple clues
        order = selected.copy()
        random.shuffle(order)

        clues = [
            f"{order[0]} is smarter than {order[1]}",
            f"{order[1]} is smarter than {order[2]}",
        ]
        random.shuffle(clues)

        questions = [
            (f"Who is the smartest?", order[0]),
            (f"Who is in the middle?", order[1]),
            (f"Who is the least smart?", order[2]),
        ]

        q_text, answer = random.choice(questions)
        question = f"{'. '.join(clues)}. {q_text}"

        return ReasoningTask(
            task_type='logical_deduction',
            question=question,
            answer=answer,
            difficulty=3,
            explanation=f"Ordering: {' > '.join(order)}",
            num_steps=2
        )

    def _generate_counterfactual(self) -> ReasoningTask:
        """Counterfactual reasoning"""
        scenarios = [
            {
                'setup': "If Alice studied harder, she would have scored 90 instead of 75. Bob scored 80.",
                'question': "Who scored higher in reality?",
                'answer': "Bob",
                'difficulty': 3,
            },
            {
                'setup': "The train would arrive at 3pm if it wasn't delayed. It was delayed by 2 hours. Charlie arrives at the station at 4pm.",
                'question': "Will Charlie miss the train?",
                'answer': "No",
                'difficulty': 4,
            },
        ]

        scenario = random.choice(scenarios)

        return ReasoningTask(
            task_type='counterfactual',
            question=f"{scenario['setup']} {scenario['question']}",
            answer=scenario['answer'],
            difficulty=scenario['difficulty'],
            explanation="Requires distinguishing actual from counterfactual",
            num_steps=2
        )

    def _generate_analogy(self) -> ReasoningTask:
        """Analogy reasoning (A:B::C:?)"""
        analogies = [
            ("cat", "kitten", "dog", "puppy", 2),
            ("hot", "cold", "day", "night", 2),
            ("book", "read", "music", "listen", 3),
            ("doctor", "hospital", "teacher", "school", 3),
            ("engine", "car", "processor", "computer", 4),
        ]

        a, b, c, d, diff = random.choice(analogies)

        question = f"Complete the analogy: {a} is to {b} as {c} is to ?"

        return ReasoningTask(
            task_type='analogy',
            question=question,
            answer=d,
            difficulty=diff,
            explanation=f"{a}:{b} :: {c}:{d}",
            num_steps=1
        )

    def _generate_code_reasoning(self) -> ReasoningTask:
        """Simple code reasoning"""
        code_snippets = [
            {
                'code': "x = 5\ny = 3\nz = x + y * 2",
                'question': "What is the value of z?",
                'answer': "11",
                'explanation': "y * 2 = 6, then x + 6 = 11 (order of operations)",
                'difficulty': 3,
            },
            {
                'code': "for i in range(3):\n    print(i)",
                'question': "How many numbers are printed?",
                'answer': "3",
                'explanation': "range(3) generates 0, 1, 2",
                'difficulty': 2,
            },
        ]

        snippet = random.choice(code_snippets)

        question = f"Given this code:\n```\n{snippet['code']}\n```\n{snippet['question']}"

        return ReasoningTask(
            task_type='code_reasoning',
            question=question,
            answer=snippet['answer'],
            difficulty=snippet['difficulty'],
            explanation=snippet['explanation'],
            num_steps=2
        )

    def _generate_probability(self) -> ReasoningTask:
        """Basic probability"""
        total = random.choice([6, 10, 20, 52])
        favorable = random.randint(1, total // 2)

        items = random.choice(['cards', 'balls', 'marbles'])
        question = f"A bag contains {total} {items}. {favorable} are red. " \
                   f"What is the probability of picking a red {items[:-1]}?"

        # Simplify fraction
        gcd = math.gcd(favorable, total)
        num = favorable // gcd
        den = total // gcd

        if den == 1:
            answer = str(num)
        else:
            answer = f"{num}/{den}"

        return ReasoningTask(
            task_type='probability',
            question=question,
            answer=answer,
            difficulty=3,
            explanation=f"P(red) = {favorable}/{total} = {answer}",
            num_steps=1
        )

    def _generate_geometry(self) -> ReasoningTask:
        """Basic geometry"""
        shape_problems = [
            lambda: self._gen_rectangle_area(),
            lambda: self._gen_triangle_area(),
            lambda: self._gen_circle_circumference(),
        ]

        generator = random.choice(shape_problems)
        return generator()

    def _gen_rectangle_area(self) -> ReasoningTask:
        length = random.randint(5, 20)
        width = random.randint(5, 20)
        area = length * width

        question = f"A rectangle has length {length} and width {width}. What is its area?"

        return ReasoningTask(
            task_type='geometry',
            question=question,
            answer=str(area),
            difficulty=1,
            explanation=f"Area = length × width = {length} × {width} = {area}",
            num_steps=1
        )

    def _gen_triangle_area(self) -> ReasoningTask:
        base = random.randint(4, 20)
        height = random.randint(4, 20)
        area = (base * height) // 2

        question = f"A triangle has base {base} and height {height}. What is its area?"

        return ReasoningTask(
            task_type='geometry',
            question=question,
            answer=str(area),
            difficulty=2,
            explanation=f"Area = (base × height) / 2 = ({base} × {height}) / 2 = {area}",
            num_steps=2
        )

    def _gen_circle_circumference(self) -> ReasoningTask:
        radius = random.randint(3, 15)
        # Use 3.14 for simplicity
        circumference = round(2 * 3.14 * radius, 1)

        question = f"A circle has radius {radius}. What is its circumference? (Use π ≈ 3.14)"

        return ReasoningTask(
            task_type='geometry',
            question=question,
            answer=str(circumference),
            difficulty=2,
            explanation=f"C = 2πr = 2 × 3.14 × {radius} = {circumference}",
            num_steps=1
        )


if __name__ == "__main__":
    # Test the advanced arena
    arena = AdvancedReasoningArena()

    print("=== Advanced Reasoning Arena Test ===\n")

    # Test each task type
    for task_type in arena.task_generators.keys():
        task = arena.generate_task(task_type)
        print(f"Type: {task.task_type} | Difficulty: {task.difficulty}/5 | Steps: {task.num_steps}")
        print(f"Q: {task.question[:100]}...")
        print(f"A: {task.answer}")
        print(f"Explanation: {task.explanation[:100]}...")
        print("-" * 70)

    print("\n=== Curriculum Learning Test ===\n")
    curriculum = arena.generate_curriculum(n_tasks=10, start_difficulty=1, end_difficulty=5)
    for i, task in enumerate(curriculum, 1):
        print(f"{i}. [{task.task_type}] Difficulty: {task.difficulty}, Steps: {task.num_steps}")
