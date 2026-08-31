import os

import pandas as pd

from ai_foundations import generation
from ai_foundations import visualizations
from ai_foundations.ngram import model as ngram_model


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

DATASET_URL = (
    "https://storage.googleapis.com/"
    "dm-educational/assets/ai_foundations/africa_galore.json"
)


def load_models():
    """Load the dataset, trigram model, and Gemma model."""

    africa_galore = pd.read_json(DATASET_URL)
    dataset = africa_galore["description"]

    print(f"Loaded Africa Galore dataset with {len(dataset)} paragraphs.")

    trigram = ngram_model.NGramModel(dataset, 3)
    print("Loaded trigram model.")

    print("Loading Gemma-1B model...")
    gemma = generation.load_gemma()
    print("Loaded Gemma-1B model.")

    return trigram, gemma


def generate_next_token(prompt, trigram, gemma):
    """Generate one token using Gemma and the trigram model."""

    gemma_text, _, _ = generation.prompt_transformer_model(
        prompt,
        max_new_tokens=1,
        loaded_model=gemma,
    )

    trigram_text = trigram.generate(1, prompt)

    print(f"\nPrompt:\n{prompt}")
    print(f"\nGemma-1B:\n{gemma_text}")
    print(f"\nTrigram model:\n{trigram_text}")


def compare_distributions(prompt, trigram, gemma):
    """Compare the next-token distributions of Gemma and the trigram model."""

    _, next_token_logits, tokenizer = generation.prompt_transformer_model(
        prompt,
        max_new_tokens=1,
        loaded_model=gemma,
    )

    print(f"\nNext-token distribution for:\n{prompt}")

    print("\nGemma-1B")
    visualizations.plot_next_token(
        next_token_logits,
        prompt=prompt,
        tokenizer=tokenizer,
    )

    print("\nTrigram model")

    context = tuple(prompt.split()[-2:])

    if context in trigram.probabilities:
        visualizations.plot_next_token(
            trigram.probabilities[context],
            prompt=prompt,
        )
    else:
        context_text = " ".join(context)
        print(
            "The trigram model cannot make a prediction because "
            f'the bigram "{context_text}" does not occur in the dataset.'
        )


def generate_sequence(
    prompt,
    num_tokens,
    trigram,
    gemma,
    sampling_mode=None,
):
    """Generate a sequence using both language models."""

    gemma_kwargs = {
        "max_new_tokens": num_tokens,
        "loaded_model": gemma,
    }

    if sampling_mode is not None:
        gemma_kwargs["sampling_mode"] = sampling_mode

    gemma_text, _, _ = generation.prompt_transformer_model(
        prompt,
        **gemma_kwargs,
    )

    if sampling_mode is None:
        trigram_text = trigram.generate(num_tokens, prompt)
    else:
        trigram_text = trigram.generate(
            num_tokens,
            prompt,
            sampling_mode=sampling_mode,
        )

    print(f"\nPrompt:\n{prompt}")
    print(f"\nGemma-1B:\n{gemma_text}")
    print(f"\nTrigram model:\n{trigram_text}")


def main():
    trigram, gemma = load_models()

    hungry_prompt = "Jide was hungry so she went looking for"
    thirsty_prompt = "Jide was thirsty so she went looking for"

    # Compare single-token predictions.
    generate_next_token(
        hungry_prompt,
        trigram,
        gemma,
    )

    # Compare next-token probability distributions.
    compare_distributions(
        hungry_prompt,
        trigram,
        gemma,
    )

    compare_distributions(
        thirsty_prompt,
        trigram,
        gemma,
    )

    # Compare longer deterministic generations.
    generate_sequence(
        hungry_prompt,
        num_tokens=50,
        trigram=trigram,
        gemma=gemma,
    )

    # Compare random generations.
    generate_sequence(
        thirsty_prompt,
        num_tokens=50,
        trigram=trigram,
        gemma=gemma,
        sampling_mode="random",
    )


if __name__ == "__main__":
    main()
