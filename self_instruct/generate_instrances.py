import os
import json
import tqdm
import random
import argparse
from collections import OrderedDict

from api import make_requests
from templates.prompts import (
    output_first_template_for_clf,
    input_first_template_for_gen,
)


random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/generations",
        help="指令保存目录"
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        help="如果指定则只生成有限的个数"
    )
    parser.add_argument(
        "--clf-result-file",
        type=str,
        default="is_clf_or_not_davinci-002_template_cls.jsonl",
        help="分类结果文件"
    )
    parser.add_argument(
        "--classification-tasks-only",
        action="store_true",
        help="如果指定则只处理分类任务"
    )
    parser.add_argument(
        "--generation-tasks-only",
        action="store_true",
        help="如果指定则只处理生成任务"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances_zh.jsonl",
        help="结果保存文件"
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=5,
        help="批处理数量"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci-002",
        help="调用的模型"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir
    num_instructions = args.num_instructions
    clf_result_file = args.clf_result_file
    classification_tasks_only = args.classification_tasks_only
    generation_tasks_only = args.generation_tasks_only
    output_file = args.output_file
    request_batch_size = args.request_batch_size
    engine = args.engine

    machine_instructions_path = os.path.join(
        save_dir, "machine_generated_instructions_zh.jsonl"
    )
    with open(machine_instructions_path) as fin:
        lines = fin.readlines()
        if num_instructions is not None:
            lines = lines[:num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {}
    clf_instruction_path = os.path.join(save_dir, clf_result_file)
    with open(clf_instruction_path) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = (
                data["is_classification"].strip().lower() == "yes"
            )

    if classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]

    if generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(save_dir, output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), request_batch_size):
            batch = tasks[batch_idx : batch_idx + request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k])
                        for k in [
                            "instruction",
                            "raw_instances",
                            "instance_metadata",
                            "instruction_metadata",
                            "most_similar",
                            "avg_similarity_score",
                        ]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = (
                            output_first_template_for_clf
                            + " "
                            + task["instruction"].strip()
                            + "\n"
                        )
                        prompts.append(prompt)
                    else:
                        prompt = (
                            input_first_template_for_gen
                            + " "
                            + task["instruction"].strip()
                            + "\n"
                        )
                        prompts.append(prompt)

                results = make_requests(
                    engine=engine,
                    prompts=prompts,
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
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0][
                            "text"
                        ]
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k])
                        for k in [
                            "instruction",
                            "raw_instances",
                            "instance_metadata",
                            "instruction_metadata",
                            "most_similar",
                            "avg_similarity_score",
                        ]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))


if __name__ == "__main__":
    main()
