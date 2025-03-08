"""
A collection of utility functions supported by regular expressions.
"""

import re

import regex


def extract_figures(
    text_str: str,
    paired_format="$",
):
    """Extract the figure results from the paired_format."""
    # This pattern is used to extract the target result from the given
    # `target_format`
    # For example, when target_format is $
    # This pattern can extract
    # $6$, $14$ -> 6, 14
    # $6.5$, $14.88$ -> 6.5, 14.88
    # $6, 7$ -> 6, 7
    # $6.5, 6.7$ -> 6.5, 6.7
    # $7.000.222$, $1000,00,0$ -> 7.000.222, 1000,00,0

    pattern = rf"\{paired_format}?(\d[\d,.]*(?:\.\d*)?)\{paired_format}?,?"

    # Find all matches in the text
    matches = re.findall(pattern, text_str)

    if not matches:
        return None

    return matches


def extract_content(text, marker, content_pattern=r"(\d+)"):
    """
    Extracts content, specifically the digital numbers, presented
    after the `marker` in a text.

    For example:
        when marker is #### and the text is "The answer is #### 1234",
        we will have 1234.
    """
    # Build the regex pattern by escaping the marker so it is taken literally.
    # Then, we allow optional whitespace and capture the desired content.
    pattern = re.escape(marker) + r"\s*" + content_pattern
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def extract_format_equations(
    text_str: str, equation_format="=", target_format="\\boxed"
):
    """
    Extract the equations in the format defined by `target_format` from
    the content after the equation_format.
    """
    # First extract the equation
    splitted_eq = text_str.split(equation_format)
    right_eq = splitted_eq[-1]
    escaped_marker = re.escape(target_format)
    pattern_str = rf"{escaped_marker}\{{(?P<content>(?:[^{{}}]|\{{(?&content)\}})*)\}}"
    pattern = regex.compile(pattern_str)
    # Extract the target result within the target_format
    matches = pattern.findall(right_eq)
    # pattern = rf"{re.escape(target_format)}{{((?:[^{{}}]+|{{[^{{}}]+}})+)}}"
    # matches = re.findall(pattern, right_eq)
    if not matches:
        return None
    return matches
