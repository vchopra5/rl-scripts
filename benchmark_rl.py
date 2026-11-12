# CODEX GENERATED
"""Compare base and saved Qwen models on the same addition problems."""

import argparse
import random
import re
import gc
import textwrap
from pathlib import Path

import torch

import run_qwen
from transformers import AutoProcessor, AutoModelForMultimodalLM


def verification(response, answer_tokens, expected_answer, eos_ids):
    match = re.match(r"\s*([+-]?\d+)", response)
    if not match or int(match.group(1)) != expected_answer:
        return -1.0
    clean = response[match.end():].strip() == ""
    stopped = answer_tokens[0, -1].item() in eos_ids
    if clean and stopped:
        return 1.0
    if not clean and not stopped:
        return 0.25
    return 0.5


def make_problems(count, max_num, seed):
    rng = random.Random(seed)
    problems = set()
    # The fixed examples are included in every prompt.
    excluded = {(2, 3), (1, 7)}
    possible = (2 * max_num + 1) * (2 * max_num + 2) // 2
    if count > possible - sum(-max_num <= a <= max_num and -max_num <= b <= max_num for a, b in excluded):
        raise ValueError("Not enough distinct problems for this number range")
    while len(problems) < count:
        pair = tuple(sorted((rng.randint(-max_num, max_num), rng.randint(-max_num, max_num))))
        if pair not in excluded:
            problems.add(pair)
    return sorted(problems)


def evaluate(problems, seed, samples):
    """Use the same prompts and random seed on both evaluations."""
    python_state = random.getstate()
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    correct = clean = total = 0
    reward_sum = 0.0
    example_outputs = {}
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            eos_ids = run_qwen.model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
            eos_ids = [token for token in eos_ids if token is not None]
            pad_id = run_qwen.processor.tokenizer.pad_token_id

            for problem_index, (n1, n2) in enumerate(problems, start=1):
                output, prompt_length, answers = run_qwen.run_model(n1, n2, preamble=False)
                tokens = output[:, prompt_length:]
                for index, answer in enumerate(answers):
                    row = tokens[index]
                    # Keep the first EOS, and discard any padding after it.
                    end = len(row)
                    for pos, token in enumerate(row.tolist()):
                        if token in eos_ids:
                            end = pos + 1
                            break
                        if pad_id is not None and pad_id not in eos_ids and token == pad_id:
                            end = pos
                            break
                    if end == 0:
                        reward = -1.0
                    else:
                        reward = verification(answer, row[:end].unsqueeze(0), n1 + n2, eos_ids)
                    total += 1
                    correct += reward > 0
                    clean += reward == 1.0
                    reward_sum += reward
                    if samples.get(problem_index - 1) == index:
                        example_outputs[problem_index - 1] = (answer, reward)
                if problem_index % 50 == 0 or problem_index == len(problems):
                    print(f"  Evaluated {problem_index}/{len(problems)} questions", flush=True)
    finally:
        random.setstate(python_state)
    return {"correct": correct, "clean": clean, "total": total,
            "mean_reward": reward_sum / total, "examples": example_outputs}


def show_examples(problems, samples, before, after):
    print(f"\n{'=' * 72}\nRandom examples: {len(samples)} questions\n{'=' * 72}")
    for number, index in enumerate(sorted(samples), start=1):
        n1, n2 = problems[index]
        print(f"\n[{number:02d}] {n1} + {n2} = {n1 + n2}")
        for label, result in (("Base", before), ("RL", after)):
            answer, reward = result["examples"][index]
            status = "correct" if reward > 0 else "wrong"
            rendered = textwrap.fill(repr(answer), width=96,
                                     initial_indent="    ", subsequent_indent="    ",
                                     break_long_words=False, break_on_hyphens=False)
            print(f"  {label} ({status}, reward {reward:g}):\n{rendered}")


def show(label, result):
    total = result["total"]
    print(f"{label}: accuracy {result['correct']}/{total} ({result['correct'] / total:.1%}); "
          f"correct and stopped {result['clean']}/{total} ({result['clean'] / total:.1%}); "
          f"mean reward {result['mean_reward']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=1500, help="Number of evaluation questions")
    parser.add_argument("--examples", type=int, default=50, help="Number of random question examples to display")
    parser.add_argument("--max-num", type=int, default=1000, help="Evaluation operand range")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trained-model", default="./models/Qwen3.5-0.8B-RL", help="Directory containing the saved trained model")
    args = parser.parse_args()
    if args.questions < 1 or args.max_num < 1 or args.examples < 0:
        parser.error("questions and max-num must be positive; examples must be nonnegative")
    if not Path(args.trained_model).is_dir():
        parser.error(f"Trained model directory not found: {args.trained_model}")

    problems = make_problems(args.questions, args.max_num, args.seed + 1)
    sample_rng = random.Random(args.seed + 3)
    indices = sample_rng.sample(range(len(problems)), min(args.examples, len(problems)))
    samples = {index: sample_rng.randrange(8) for index in indices}
    print(f"Device: {run_qwen.device}; {args.questions} evaluation questions, "
          "8 responses each", flush=True)
    print("Evaluating base model...", flush=True)
    before = evaluate(problems, args.seed + 2, samples)
    show("Before", before)

    print(f"Loading saved model: {args.trained_model}", flush=True)
    run_qwen.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_qwen.processor = AutoProcessor.from_pretrained(args.trained_model)
    run_qwen.model = AutoModelForMultimodalLM.from_pretrained(args.trained_model).to(run_qwen.device)
    print("Evaluating saved model...", flush=True)
    after = evaluate(problems, args.seed + 2, samples)
    show("After ", after)
    change = (after["correct"] / after["total"] - before["correct"] / before["total"]) * 100
    print(f"Accuracy change: {change:+.1f} percentage points", flush=True)
    # show_examples(problems, samples, before, after)
    # print("Evaluation only; no model weights were changed or saved.", flush=True)


if __name__ == "__main__":
    main()

