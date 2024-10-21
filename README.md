
<div align="center">
    <h1>Self-Instruct: 使用自生成指令对齐语言模型中文版</h1>
</div>

此代码库的公开代码是基于 davinci-002 completion 进行中文实验的。更多中文实验是基于 qwen 系列进行探索的（待开源），这部分包括指令生成、SFT 训练等。

# self-instruct-zh 介绍

self-instruct-zh copy from [self-instruct](https://github.com/yizhongw/self-instruct)

由于要处理的是中文，因此专门针对中文文本处理做了些优化。


# self-instruct-zh 运行

```shell
# 把种子任务从英文翻译为中文
sh scripts/translation.sh

# 从种子任务出发生成新的指令数据
sh scripts/generate_instructions.sh

# 识别指令是否是分类任务
sh scripts/is_clf_or_not.sh

# 为每条指令生成实例
sh scripts/generate_instances.sh 

# 过滤、去重、格式化为微调格式
sh scripts/prepare_for_finetuning.sh 
```

