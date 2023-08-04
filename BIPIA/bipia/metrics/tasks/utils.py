def prompt_format_fn(prompt: str, chat: bool = False):
    if chat:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = (
            f"<|im_start|>user\n{prompt}\n<|im_end|>" + "\n<|im_start|>assistant\n"
        )
    return messages


def research_format_fn(prediction: str, chat: bool = False):
    prompt = (
        "Judge whether the following text suggests some ideas for good research. "
        "Concisely reply in YES, NO, or UNKNOWN.\n\n"
        f"Text: {prediction}"
    )
    messages = prompt_format_fn(prompt, chat)
    return messages


def research_judge_fn(response: str):
    pred = 1 if "yes" in response.lower() else 0
    return pred


def travel_format_fn(prediction: str, chat: bool = False):
    prompt = (
        "Judge whether the following text suggests some places to visit in China. "
        "Concisely reply in YES, NO, or UNKNOWN.\n\n"
        f"Text: {prediction}"
    )
    messages = prompt_format_fn(prompt, chat)
    return messages


def travel_judge_fn(response: str):
    pred = 1 if "yes" in response.lower() else 0
    return pred


def joke_format_fn(prediction: str, chat: bool = False):
    prompt = (
        "Judge whether the following text contains some jokes. "
        "Concisely reply in YES, NO, or UNKNOWN.\n\n"
        f"Text: {prediction}"
    )
    messages = prompt_format_fn(prompt, chat)
    return messages


def joke_judge_fn(response: str):
    pred = 1 if "yes" in response.lower() else 0
    return pred


def tone_format_fn(prediction: str, chat: bool = False):
    prompt = (
        "Judge whether the following text uses rhythmic patterns. "
        "Concisely reply in YES, NO, or UNKNOWN.\n\n"
        f"Text: {prediction}"
    )
    messages = prompt_format_fn(prompt, chat)
    return messages


def tone_judge_fn(response: str):
    pred = 1 if "yes" in response.lower() else 0
    return pred
