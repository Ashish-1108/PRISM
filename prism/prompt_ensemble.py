"""20 diverse prompt templates for LAION-CLAP text prototype ensembling."""

import torch
import torch.nn.functional as F

PROMPT_TEMPLATES = [
    "This is a sound of {}",
    "A sound of {}",
    "The sound of {}",
    "A recording of {}",
    "{}",
    "A {} sound",
    "Sound of {}",
    "A {} sound captured by a microphone",
    "An audio clip of {}",
    "This is an audio recording of {}",
    "The audio recording contains {}",
    "This audio contains the sound of {}",
    "A {} sound in an urban environment",
    "{} heard in the city",
    "A {} sound recorded outdoors",
    "A clear recording of {}",
    "The sound of {} in a quiet environment",
    "An isolated {} sound",
    "The sound of {} happening",
    "Audio of {} occurring",
]


def get_multi_prompt_text_features(label_map, model, templates=None):
    """
    Build text prototypes by averaging embeddings from multiple prompt templates.

    Args:
        label_map: dict {idx: class_name}
        model:     LAION-CLAP model instance
        templates: list of prompt strings (uses PROMPT_TEMPLATES if None)

    Returns:
        (C, 512) L2-normalized text prototypes
    """
    if templates is None:
        templates = PROMPT_TEMPLATES
    all_embeds = []
    for template in templates:
        prompts = [template.format(v) for _, v in label_map.items()]
        all_embeds.append(torch.tensor(model.get_text_embedding(prompts)))
    return F.normalize(torch.stack(all_embeds).mean(dim=0), p=2, dim=-1)
