"""
A collection of rewards that are computed for the visual task based on some common rules.
"""

import re

import torchvision


def iou_reward(completions, solution, **kwargs):
    """
    Compute the IOU rewards by comparing the bounding boxes predicted by the model and the ground truth.

    IOU>0.5, reward=1.0; otherwise, reward=0.0.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]"
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [
                        int(bbox_match.group(1)),
                        int(bbox_match.group(2)),
                        int(bbox_match.group(3)),
                        int(bbox_match.group(4)),
                    ]
                    if torchvision.ops.box_iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
    return rewards
