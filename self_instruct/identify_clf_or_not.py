import os
import tqdm
import json
import argparse
from collections import OrderedDict

from api import make_requests
from templates.prompts import classification_prompt


templates = {
    "template_cls": classification_prompt
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        default="data/generations",
        help="指令保存目录"
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        help="指定需要处理的指令个数，如果不指定则处理所有"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci-002",
        help="调用的模型"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="template_cls",
        help="使用的 prompt"
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=5,
        help="批量处理的数目"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir
    num_instructions = args.num_instructions
    engine = args.engine
    template = args.template
    request_batch_size = args.request_batch_size
    
    machine_instructions_path = os.path.join(
        save_dir, "machine_generated_instructions_zh.jsonl"
    )
    with open(machine_instructions_path) as fin:
        lines = [json.loads(i) for i in fin.readlines()]
        if num_instructions is not None:
            lines = lines[:num_instructions]
            
    output_path = os.path.join(save_dir, f"is_clf_or_not_{engine}_{template}.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except Exception as e:
                    print(e)
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), request_batch_size):
            batch = [line for line in lines[batch_idx: batch_idx + request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict((k, data[k].strip()) for k in ["instruction", "is_classification"])
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prefix = templates[template]
                prompts = [prefix + d["instruction"].strip() + "\n" + "Is it classification? " for d in batch]
                results = make_requests(
                    engine=engine,
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1)
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][0]["text"].strip()
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict((k, data[k]) for k in ["instruction", "is_classification"])
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))


if __name__ == "__main__":
    main()