import math
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from typing import Any, Optional, Union

SYSTEM_PROMPTS = {
    "DeepSeekZero": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
""",
    "default": """Given a problem, your task is to answer the question by thinking step-by-step in a clear and specific manner.
Once you have thought about the reasoning process, provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
""",
    "thinking": """You are a helpful assistant that solves problems step by step.

IMPORTANT: You MUST format your response as follows:
1. First, think through the problem inside <think>...</think> tags
2. Then, provide ONLY your final answer inside <answer>...</answer> tags

Example format:
<think>
[Your reasoning here]
</think>
<answer>[Your final answer here, no explanation]</answer>

The answer tags are REQUIRED - do not skip them. Match the exact format requested in the problem.
""",
    "simple": "You are a helpful assistant that answers questions accurately and concisely. When asked to solve a problem, show your work step by step. Provide your final answer between <answer> and </answer> tags.",
    "direct": "Answer the question directly. Provide your answer between <answer> and </answer> tags. Do not return any preamble, explanation, or reasoning.",
    "chain_of_draft": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response enclosed in <answer> </answer> tags.",
}


def extract_answer(completion: str, tag_name: str = "answer", strip: bool = True) -> Optional[str]:
    """Extract answer from model completion with multiple fallback strategies.

    Strategy order:
    1. Look for <answer>...</answer> tags (standard format)
    2. Look for content after </think> tag (common with thinking models)
    3. Look for JSON/list/grid at the end of response
    """
    if not completion:
        return None

    # Strategy 1: Standard <answer> tags
    regex = f"<{tag_name}>\\s?(.*?)\\s?</{tag_name}>"
    matches = list(
        re.finditer(
            regex,
            completion,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        answer = matches[-1].group(1)
        if strip:
            answer = answer.strip()
        return answer

    # Strategy 2: Content after </think> tag
    think_end_match = re.search(r'</think>\s*(.+)$', completion, flags=re.DOTALL | re.IGNORECASE)
    if think_end_match:
        after_think = think_end_match.group(1).strip()
        if after_think:
            # Check if it's wrapped in other tags we should extract from
            inner_match = re.search(r'<(\w+)>\s*(.*?)\s*</\1>$', after_think, flags=re.DOTALL)
            if inner_match:
                return inner_match.group(2).strip() if strip else inner_match.group(2)
            return after_think

    # Strategy 3: Look for structured data at the end (JSON, lists, grids)
    # Try to find JSON object/array at end
    json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])\s*$', completion)
    if json_match:
        return json_match.group(1).strip() if strip else json_match.group(1)

    # Look for grid-like output (multiple lines of similar structure at the end)
    lines = completion.strip().split('\n')
    if len(lines) >= 2:
        # Check if last few lines look like a grid (numbers/chars separated by spaces)
        potential_grid = []
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[\d\s\.,\-]+$', line) or re.match(r'^[A-Za-z\s]+$', line):
                potential_grid.insert(0, line)
            else:
                break
        if len(potential_grid) >= 2:
            return '\n'.join(potential_grid)

    return None


def format_number(num: Union[int, float], max_decimals: int = 2, round_if_needed: bool = False) -> str:
    """Convert a number to string representation with controlled decimal places.

    Args:
        num: Number to format
        max_decimals: Maximum allowed decimal places
        round_if_needed: If True, round the number to max_decimals instead of raising an error

    Returns:
        String representation of the number

    Raises:
        ValueError: If number requires more decimal places than allowed and round_if_needed is False
    """
    if isinstance(num, int) or num.is_integer():
        return str(int(num))

    # Convert to Decimal for exact decimal arithmetic
    d = Decimal(str(num))

    # Find required decimals by removing trailing zeros
    str_val = f"{d:f}"
    str_val = str_val.rstrip("0").rstrip(".")
    if "." in str_val:
        required_decimals = len(str_val.split(".")[1])
        if required_decimals > max_decimals and not round_if_needed:
            raise ValueError(f"Number {num} requires {required_decimals} decimals but only {max_decimals} allowed")

    # Format with required decimals (will round if needed)
    result = f"{num:.{max_decimals}f}".rstrip("0").rstrip(".")

    # Verify result parses back to original value (skip verification if rounding was applied)
    if not round_if_needed:
        try:
            parsed = float(result)
            if not math.isclose(parsed, num, rel_tol=1e-9):
                raise ValueError(f"String representation {result} does not match original value {num}")
        except (ValueError, InvalidOperation) as e:
            raise ValueError(f"Failed to verify string representation: {e}")

    return result


def is_integer(obj: Any) -> bool:
    if isinstance(obj, (int, float)):
        return isinstance(obj, int) or obj.is_integer()
    elif isinstance(obj, Fraction):
        return obj.denominator == 1
    return False


def compute_decimal_reward(answer: Optional[str], oracle_answer: str, strip_commas: bool = True) -> float:
    """Compute the reward for a given answer compared to the oracle answer.
    Tolerate sign, leading zeros and trailing decimals, optionally strip commas ("+01,000.00" == "1000")

    Args:
        answer: Answer provided by model
        oracle_answer: Correct answer to the question
        strip_commas: Whether to remove commas from answers e.g "1,000" = "1000"

    Returns:
        Reward value between 0.0 and 1.0
    """
    reward = 0.0
    if answer is not None and len(answer) > 0:
        try:
            if strip_commas:
                answer = answer.replace(",", "")
                oracle_answer = oracle_answer.replace(",", "")

            if Decimal(answer) == Decimal(oracle_answer):
                return 1.0
        except:
            pass

        if oracle_answer in answer:
            reward = len(oracle_answer) / len(answer)

    return reward


class StrEnum(str, Enum):
    """
    Taken from Python 3.11 StrEnum implementation, moved here to support Python 3.10.
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
