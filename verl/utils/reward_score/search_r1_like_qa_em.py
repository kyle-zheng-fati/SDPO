# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the final answer from a planner TERMINATE response.

    Implements the JSON-`answer`-key contract: the response must contain a
    JSON object with an ``"answer"`` key whose value is a non-empty string.
    Returns the string on success, or ``None`` on contract failure (which
    ``compute_score`` translates to reward 0).

    The implementation is intentionally self-contained — verl is a
    git submodule and must not depend on our project's ``src/`` package.
    The semantics mirror :func:`src.utils.answer_extraction.extract_answer_strict`
    but without raising or carrying role/step-number metadata (verl's
    scorer never propagated those).
    """
    import json

    if not solution_str or not str(solution_str).strip():
        return None

    # Strip <think>...</think> blocks (closed and unclosed-leading) before
    # JSON scanning so reasoning content cannot pollute extraction.
    cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
    if "<think>" in cleaned and "</think>" not in cleaned:
        idx = cleaned.find("<think>")
        cleaned = cleaned[:idx]

    # Strip markdown fences and Python-style string concatenation.
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r'"\s*\+\s*"', "", cleaned)

    # Try straight JSON parse first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            ans = parsed["answer"]
            if ans is not None and isinstance(ans, (str, int, float)) and str(ans).strip():
                return str(ans).strip()
    except json.JSONDecodeError:
        pass

    # Right-to-left brace scan for the last balanced {...} containing "answer".
    depth = 0
    end = None
    for i in range(len(cleaned) - 1, -1, -1):
        ch = cleaned[i]
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end is not None:
                candidate = cleaned[i:end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        ans = parsed["answer"]
                        if ans is not None and isinstance(ans, (str, int, float)) and str(ans).strip():
                            return str(ans).strip()
                except json.JSONDecodeError:
                    pass
                end = None  # continue scanning leftward

    return None


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
