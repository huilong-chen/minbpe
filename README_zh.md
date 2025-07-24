# minbpe

用于（字节级）字节对编码（BPE）算法的最小、干净的代码，该算法通常用于LLM分词。BPE算法是"字节级"的，因为它在UTF-8编码的字符串上运行。

这个算法由[GPT-2论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)和OpenAI发布的GPT-2[代码](https://github.com/openai/gpt-2)推广用于LLM。[Sennrich et al. 2015](https://arxiv.org/abs/1508.07909)被引用为在NLP应用中使用BPE的原始参考文献。如今，所有现代LLM（如GPT、Llama、Mistral）都使用该算法来训练其分词器。

这个仓库中有两个分词器，都可以执行分词器的3个主要功能：1）在给定文本上训练分词器词汇表和合并规则，2）从文本编码为标记，3）从标记解码为文本。仓库的文件如下：

1. [minbpe/base.py](minbpe/base.py)：实现`Tokenizer`类，这是基类。它包含`train`、`encode`和`decode`存根、保存/加载功能，还有一些常见的实用函数。这个类不是直接使用的，而是被继承的。
2. [minbpe/basic.py](minbpe/basic.py)：实现`BasicTokenizer`，这是BPE算法的最简单实现，直接在文本上运行。
3. [minbpe/regex.py](minbpe/regex.py)：实现`RegexTokenizer`，它通过正则表达式模式进一步分割输入文本，这是一个预处理阶段，在分词之前按类别（例如：字母、数字、标点符号）分割输入文本。这确保了不会在类别边界上发生合并。这在GPT-2论文中引入，并且在GPT-4中继续使用。如果有的话，这个类还处理特殊标记。
4. [minbpe/gpt4.py](minbpe/gpt4.py)：实现`GPT4Tokenizer`。这个类是`RegexTokenizer`（上面的2）的一个轻量级包装器，它完全重现了[tiktoken](https://github.com/openai/tiktoken)库中GPT-4的分词。包装处理了一些细节，如恢复分词器中的确切合并和处理一些不幸的（可能是历史性的？）1字节标记排列。

最后，脚本[train.py](train.py)在输入文本[tests/taylorswift.txt](tests/taylorswift.txt)（这是她的维基百科条目）上训练两个主要的分词器，并将词汇表保存到磁盘以供可视化。这个脚本在我的（M1）MacBook上运行约25秒。

以上所有文件都非常短小且有详细注释，并且在文件底部包含使用示例。

## 快速开始

作为最简单的示例，我们可以重现[Wikipedia上的BPE文章](https://en.wikipedia.org/wiki/Byte_pair_encoding)如下：

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256是字节标记，然后进行3次合并
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# 写入两个文件：toy.model（用于加载）和toy.vocab（用于查看）
```

根据Wikipedia，在输入字符串："aaabdaaabac"上运行bpe进行3次合并的结果是字符串："XdXac"，其中X=ZY，Y=ab，Z=aa。需要注意的是，minbpe总是将256个单独的字节分配为标记，然后根据需要从那里合并字节。所以对我们来说a=97，b=98，c=99，d=100（它们的[ASCII](https://www.asciitable.com)值）。然后当(a,a)合并为Z时，Z将变成256。同样Y将变成257，X为258。所以我们从256个字节开始，进行3次合并得到上述结果，预期输出为[258, 100, 258, 97, 99]。

## 推理：与GPT-4的比较

我们可以通过以下方式验证`RegexTokenizer`与[tiktoken](https://github.com/openai/tiktoken)中的GPT-4分词器具有功能对等性：

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

（你需要`pip install tiktoken`才能运行）。在底层，`GPT4Tokenizer`只是`RegexTokenizer`的一个轻量级包装器，传入了GPT-4的合并规则和特殊标记。我们还可以确保特殊标记被正确处理：

```python
text = ""

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# [100257, 15339, 1917]
```

注意，就像tiktoken一样，我们必须在encode调用中明确声明我们使用和解析特殊标记的意图。否则这可能成为一个重大隐患，无意中用特殊标记对攻击者控制的数据（如用户提示）进行分词。`allowed_special`参数可以设置为"all"、"none"或允许的特殊标记列表。

## 训练

与tiktoken不同，这段代码允许你训练自己的分词器。原则上据我所知，如果你在大型数据集上训练`RegexTokenizer`，词汇量为100K，你将重现GPT-4分词器。

你可以遵循两条路径。首先，你可以决定不需要使用正则表达式模式分割和预处理文本的复杂性，也不关心特殊标记。在这种情况下，选择`BasicTokenizer`。你可以训练它，然后编码和解码，例如：

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("mymodel") # 写入mymodel.model和mymodel.vocab
tokenizer.load("mymodel.model") # 从磁盘加载模型，词汇表仅用于查看
```

如果你想要按照OpenAI的方法使用正则表达式模式按类别分割文本，那么采用他们的方法是个好主意。GPT-4模式是`RegexTokenizer`的默认值，所以你可以简单地这样做：

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("tok32k") # 写入tok32k.model和tok32k.vocab
tokenizer.load("tok32k.model") # 从磁盘加载模型
```

当然，你需要根据数据集的大小更改词汇量。

**特殊标记**。最后，你可能希望向分词器添加特殊标记。使用`register_special_tokens`函数注册这些标记。例如，如果你用32768的词汇量训练，那么前256个标记是原始字节标记，接下来的32768-256是合并标记，之后你可以添加特殊标记。最后一个"真实"的合并标记的ID将是32767（词汇量-1），所以你的第一个特殊标记应该紧跟在其后，ID正好是32768。因此：

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"": 32768})
tokenizer.encode("", allowed_special="all")
```

当然，你也可以在之后添加更多标记。最后，我想强调的是，我努力保持代码本身的清洁、可读和可修改。你不应该害怕阅读代码并理解它的工作原理。测试也是查看更多使用示例的好地方。这提醒我：

## 测试

我们使用pytest库进行测试。所有测试都位于`tests/`目录中。如果你还没有安装，请先`pip install pytest`，然后：

```bash
$ pytest -v .
```

运行测试。（-v是详细模式，稍微漂亮一点）。

## 社区扩展

* [gnp/minbpe-rs](https://github.com/gnp/minbpe-rs)：`minbpe`的Rust实现，与Python版本提供（近乎）一对一的对应关系

## 练习

对于尝试学习BPE的人，这里是如何逐步构建自己的minbpe的建议练习进度。参见[exercise.md](exercise.md)。

## 讲座

我在[YouTube视频](https://www.youtube.com/watch?v=zduSFxRajkE)中构建了这个仓库中的代码。你也可以在[lecture.md](lecture.md)中找到文本形式的讲座。

## 待办事项

- 编写一个更优化的Python版本，可以在大文件和大词汇表上运行
- 编写一个更加优化的C或Rust版本（仔细思考）
- 将GPT4Tokenizer重命名为GPTTokenizer并支持GPT-2/GPT-3/GPT-3.5？
- 编写类似GPT4Tokenizer的LlamaTokenizer（即尝试sentencepiece等效）

## 许可证

MIT