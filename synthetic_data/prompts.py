"""
Prompt templates and variation matrix for synthetic data generation.
"""

from dataclasses import dataclass
from typing import Iterator


# Prompt templates for diversity in reasoning styles
# All prompts MUST:
# 1. Instruct the model to WRITE OUT its reasoning (not just think internally)
# 2. End with <answer></answer> tags AT THE VERY END
PROMPTS = {
    "step_by_step": """Solve this problem by writing out your reasoning step by step.

RESPONSE FORMAT - You MUST follow this structure:
1. First, write out your step-by-step reasoning
2. Then, at the VERY END, put your final answer in <answer></answer> tags

Example structure:
Step 1: [reasoning]
Step 2: [reasoning]
...
Step N: [final reasoning]

<answer>final answer here</answer>

CRITICAL: The <answer></answer> tags must appear at the END of your response, AFTER all reasoning. Do NOT put the answer first.""",

    "chain_of_thought": """Solve this problem using chain of thought reasoning.

RESPONSE FORMAT - Follow this structure exactly:
1. **Understanding**: What is the problem asking?
2. **Given Information**: What facts and constraints do we have?
3. **Reasoning**: Work through the logic step by step
4. **Verification**: Check your answer makes sense
5. **Final Answer**: <answer>your answer</answer>

CRITICAL: You MUST end your response with <answer></answer> tags. The answer tags must be the LAST thing in your response, after all your reasoning.""",

    "analytical": """Analyze this problem systematically.

RESPONSE FORMAT:
- Problem breakdown: Identify key components
- Analysis: Examine each component
- Deductions: Draw conclusions
- Solution: Put the pieces together

Then conclude with your answer in tags.

CRITICAL: Your response MUST end with <answer>final answer</answer> tags. Do NOT put the answer anywhere else.""",

    "concise": """Solve this problem efficiently. Show key reasoning steps, then give your answer.

RESPONSE FORMAT:
1. Brief approach and key reasoning steps
2. Final answer in tags at the end

CRITICAL: End with <answer>answer</answer> - these tags MUST appear at the END of your response.""",

    "teaching": """Explain your solution as if teaching someone.

RESPONSE FORMAT:
- Start with what we know
- Walk through each logical step
- Explain WHY each step follows
- End with your answer in tags

CRITICAL: Your response MUST conclude with <answer>your answer</answer> tags at the very end. All reasoning comes BEFORE the answer tags.""",
}

# Temperature settings for output variety
# 0.4 = more focused/deterministic
# 0.7 = more diverse/creative
TEMPERATURES = [0.4, 0.7]


@dataclass
class PromptVariation:
    """Represents a single prompt variation."""

    template_id: str
    system_prompt: str
    temperature: float


def get_prompt_template(template_id: str) -> str:
    """Get a prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        The prompt template string

    Raises:
        ValueError: If template_id not found
    """
    if template_id not in PROMPTS:
        raise ValueError(f"Unknown prompt template: {template_id}")
    return PROMPTS[template_id]


def get_all_variations() -> list[PromptVariation]:
    """Get all prompt variations (template x temperature matrix).

    Returns:
        List of PromptVariation objects (5 templates x 2 temperatures = 10 variations)
    """
    variations = []
    for template_id, system_prompt in PROMPTS.items():
        for temp in TEMPERATURES:
            variations.append(
                PromptVariation(
                    template_id=template_id,
                    system_prompt=system_prompt,
                    temperature=temp,
                )
            )
    return variations


def iter_variations(
    template_ids: list[str] | None = None,
    temperatures: list[float] | None = None,
) -> Iterator[PromptVariation]:
    """Iterate over prompt variations with optional filtering.

    Args:
        template_ids: Optional list of template IDs to include
        temperatures: Optional list of temperatures to include

    Yields:
        PromptVariation objects
    """
    use_templates = template_ids or list(PROMPTS.keys())
    use_temps = temperatures or TEMPERATURES

    for template_id in use_templates:
        if template_id not in PROMPTS:
            raise ValueError(f"Unknown prompt template: {template_id}")
        system_prompt = PROMPTS[template_id]
        for temp in use_temps:
            yield PromptVariation(
                template_id=template_id,
                system_prompt=system_prompt,
                temperature=temp,
            )


def count_variations(
    template_ids: list[str] | None = None,
    temperatures: list[float] | None = None,
) -> int:
    """Count the number of variations.

    Args:
        template_ids: Optional list of template IDs to include
        temperatures: Optional list of temperatures to include

    Returns:
        Number of variations
    """
    num_templates = len(template_ids) if template_ids else len(PROMPTS)
    num_temps = len(temperatures) if temperatures else len(TEMPERATURES)
    return num_templates * num_temps
