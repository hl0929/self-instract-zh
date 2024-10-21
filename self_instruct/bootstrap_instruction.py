import os
import re
import json
import tqdm
import string
import random
import argparse
from functools import partial

import jieba
import numpy as np
from rouge_chinese import Rouge
from multiprocessing import Pool

from api import make_requests


random.seed(42)


def sample_machine_instructions(machine_instructions, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        # prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
        prompt = "请使用中文，想出一系列分类任务。 尽可能地指定可能的输出标签。\n"
    else:
        # prompt = "Come up with a series of tasks:\n"
        prompt = "请使用中文，想出一系列任务：\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":").strip("：")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def find_word_in_string(w, s):
    return w.lower() in s.lower()


def zh_token_wrapper(text):
    return " ".join([i for i in jieba.lcut(text) if i.strip()])


def letter_ratio(text):
    """粗略统计英文占比"""
    cnt = 0
    for char in text.split():
        if char.isalpha():
            cnt += 1
    return cnt / len(text.split())


def get_rouge_score(rouge, candidates, references):
    scores = rouge.get_scores(candidates, references, avg=True)
    return scores


def post_process_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"[\n\s]\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst) <= 3 or len(inst) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(
            find_word_in_string(word, inst)
            for word in ["图片", "图像", "照片", "相册", "文件", "地图", "画图"]
        ):
            continue
        # 过滤代码
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.、
        # 中文待观测
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # # filter those starting with non-english character
        # if not inst[0].isascii():
        #     continue
        
        # inst = inst.replace("英语", "汉语").replace("英文", "汉语")

        if letter_ratio(inst) > 0.6:
            continue

        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-tasks-path",
        type=str,
        required=True,
        default="data/seed_tasks_zh.jsonl",
        help="种子任务文件路径"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        default="data/generations",
        help="指令保存目录"
    )
    parser.add_argument(
        "--num-instructions-to-generate",
        type=int,
        default=50,
        help="生成的指令数量"
    )
    parser.add_argument(
        "--use-clf-seed-tasks-only",
        action="store_true",
        help="如果指定这个参数，将只使用分类种子进行指令生成"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci-002",
        help="生成模型"
    )
    parser.add_argument(
        "--num-prompt-instructions",
        type=int,
        default=8,
        help="prompt 中 few shot 的数目"
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=8,
        help="并发处理数"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_tasks_path = args.seed_tasks_path
    use_clf_seed_tasks_only = args.use_clf_seed_tasks_only
    save_dir = args.save_dir
    num_instructions_to_generate = args.num_instructions_to_generate
    request_batch_size = args.request_batch_size
    num_prompt_instructions = args.num_prompt_instructions
    engine = args.engine

    with open(seed_tasks_path) as reader:
        seed_tasks = [json.loads(i) for i in reader]
    print(f"Total seed instruction {len(seed_tasks)}")

    if use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
        print(f"Classification seed instruction {len(seed_tasks)}")

    seed_instructions = [t["instruction_zh"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(save_dir, exist_ok=True)
    machine_instructions = []
    machine_instructions_path = os.path.join(
        save_dir, "machine_generated_instructions_zh.jsonl"
    )
    if os.path.exists(machine_instructions_path):
        with open(machine_instructions_path, "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    rouge = Rouge()
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))
    with open(machine_instructions_path, "a") as fout:
        while len(machine_instructions) < num_instructions_to_generate:
            batch_inputs = []
            for _ in range(request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, n=2
                )
                # sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions,
                    num_prompt_instructions - len(prompt_instructions),
                )
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(
                    prompt_instructions, classification=use_clf_seed_tasks_only
                )
                batch_inputs.append(prompt)

            results = make_requests(
                engine=engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
            )

            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                with Pool(8) as pool:
                    rouge_scores = pool.map(
                        partial(get_rouge_score, rouge, zh_token_wrapper(inst)),
                        [zh_token_wrapper(line) for line in seed_instructions + machine_instructions],
                    )
                rouge_scores = [score["rouge-l"]["f"] for score in rouge_scores]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                obj = {
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                progress_bar.update(1)


if __name__ == "__main__":
    main()
