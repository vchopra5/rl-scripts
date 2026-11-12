import random
import run_qwen
import re
import numpy as np
import torch

random.seed(42)


optimizer = torch.optim.AdamW(
    run_qwen.model.parameters(),
    lr=1e-6
)

def create_problem(max_num=1000):
    num1 = random.randint(-max_num, max_num)
    num2 = random.randint(-max_num, max_num)

    return min(num1, num2), max(num1, num2), num1+num2



def verification(response, answer_tokens, expected_answer, eos_token_id):
    match = re.match(r"\s*([+-]?\d+)", response)
    
    if not match or int(match.group(1)) != expected_answer:
        return -1.0

    remaining = response[match.end():]
    clean_answer = remaining.strip() == ""
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    stopped = answer_tokens[0, -1].item() in eos_ids


    if clean_answer and stopped:
        return 1.0
    elif not clean_answer and not stopped:
        return 0.25
    else:
        return 0.5


def train_step():
    t_id = run_qwen.model.generation_config.eos_token_id
    n1, n2, ans = create_problem(1000)
    rewards = np.zeros(8)

    # for i in range(8):
    output, prompt_length, answers = run_qwen.run_model(n1, n2, preamble=False)
    answer_tokens = output[:, prompt_length:]

    eos_ids = t_id if isinstance(t_id, (list, tuple)) else [t_id]

    is_eos= torch.zeros_like(answer_tokens, dtype=torch.bool)

    for eos_id in eos_ids:
        if eos_id is not None:
            is_eos |= answer_tokens == eos_id


    eos_before = is_eos.cumsum(dim=-1) - is_eos.long()
    answer_mask = eos_before == 0

    pad_id = run_qwen.processor.tokenizer.pad_token_id

    if pad_id is not None:
        answer_mask &= (answer_tokens != pad_id) | is_eos


    rewards = torch.tensor(
        [
            verification(
                answer,
                answer_tokens[i][answer_mask[i]].unsqueeze(0),
                ans,
                t_id
            )

            for i, answer in enumerate(answers)
        ],

        dtype=torch.float32,
        device=run_qwen.device
    )
    
    
    advantages = rewards - rewards.mean()

    with torch.no_grad():
        logits = run_qwen.model(input_ids=output).logits

        answer_logits = logits[:, prompt_length-1:-1, :]
        answer_tokens = output[:, prompt_length:]

        log_probs = torch.log_softmax(answer_logits.float(), dim=-1)

        old_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=answer_tokens.unsqueeze(-1),
        ).squeeze(-1)

    # Current log probs

    logits = run_qwen.model(input_ids=output).logits
    answer_logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(answer_logits.float(), dim=-1)


    current_log_probs= torch.gather(
        log_probs,
        dim=-1,
        index=answer_tokens.unsqueeze(-1)
    ).squeeze(-1)

    ratios = torch.exp(current_log_probs - old_log_probs)

    clipped = torch.clip(ratios, min=0.8, max=1.2)

    advantages = advantages.unsqueeze(-1)

    objective = torch.minimum(
        ratios * advantages,
        clipped * advantages,
    )

    token_losses = -objective * answer_mask

    response_losses = token_losses.sum(dim=1) / answer_mask.sum(dim=1)
    loss = response_losses.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Question:", n1, "+", n2, "=", ans)
    print("Rewards:", rewards.tolist())
    print("Loss:", loss.item())


if __name__ == '__main__':
    for step in range(1000):
        print("-------------")

        print(f"Step: {step}")
        train_step()
    run_qwen.model.save_pretrained("./models/Qwen3.5-0.8B-RL")
    run_qwen.processor.save_pretrained("./models/Qwen3.5-0.8B-RL")

