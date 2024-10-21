import argparse
from tqdm import tqdm

from api import get_gen
from utils import read_jsonline, save_jsonline
from templates.prompts import translation_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="需要进行翻译的原始文件，文件格式为jsonline"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        default="data/seed_tasks_zh.jsonl",
        help="翻译后的文件，文件格式为jsonline"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    en_seed_tasks = read_jsonline(args.raw_path)
    print("任务数:", len(en_seed_tasks))
    
    translated_data = []
    for _, task in enumerate(tqdm(en_seed_tasks)):
        name_en = task["name"]
        instruction_en = task["instruction"]
        input_en = task["instances"][0]["input"]
        output_en = task["instances"][0]["output"]
        
        name_zh = get_gen(text=name_en, system=translation_prompt)
        instruction_zh = get_gen(text=instruction_en, system=translation_prompt) if instruction_en else ""
        input_zh = get_gen(text=input_en, system=translation_prompt) if input_en else ""
        output_zh = get_gen(text=output_en, system=translation_prompt)
        
        task["name_zh"] = name_zh
        task["instruction_zh"] = instruction_zh
        task["instances_zh"] = [
            {
                "input_zh": input_zh,
                "output_zh": output_zh
            }
        ]
        
        translated_data.append(task)
        # if index > 2:
        #     break
    
    save_jsonline(args.save_path, translated_data)
    print(f"结果保存到 {args.save_path}")
    


if __name__ == "__main__":
    main()
