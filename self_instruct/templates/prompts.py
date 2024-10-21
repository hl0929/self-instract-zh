
translation_prompt = """你是一个专业的翻译，擅长把英文翻译为中文，你在翻译的过程中一定要保持原文中的含义与格式。

只要是英文内容，你都需要翻译为中文，其他符号保持不变，格式保持不变。"""

classification_prompt = """Can the following task be regarded as a classification task with finite output labels?

Task: 请为以下内容生成一个包含它们的列表： 
Is it classification? No

Task: 举一个你必须运用幽默感的例子。
Is it classification? No

Task: 将给定文本中的占位符替换为合适的命名实体。
Is it classification? No

Task: 事实核查——根据你的知识和常识，判断陈述是否为真、假或未知。
Is it classification? Yes

Task: 返回该人的社会保障号码（SSN）。
Is it classification? No

Task: 检测Reddit帖子中是否包含仇恨言论。
Is it classification? Yes

Task: 分析以下句子以识别偏见。
Is it classification? No

Task: 请将下面的字符串转换成正常大小写。
Is it classification? No

Task: 找出句子中的有毒词语或短语。
Is it classification? No

Task: 按人口对这些国家进行排序。
Is it classification? No

Task: 你会得到一篇新闻文章，需要识别其所属的所有类别。可能的类别包括：音乐、体育、政治、科技、金融、篮球、足球、网球、娱乐、数字游戏、世界新闻。逐个输出类别，用逗号分隔。
Is it classification? Yes

Task: 给出一个锻炼名称，并解释如何进行。
Is it classification? No

Task: 请给出以下每个字符串中所有的空格和回车符。 
Is it classification? No

Task: 找出前四个最小的完美数。
Is it classification? No

Task: 判断文档中的信息是否支持该主张。你可以回答“支持”或“不支持”。
Is it classification? Yes

Task: 为给定的假设旅行制定详细的预算。
Is it classification? No

Task: 给定一个句子，判断其中是否存在潜在的刻板印象。如果存在，你需要解释该刻板印象；否则输出“无”。
Is it classification? No

Task: 解释以下习语，并尝试举一些例子。
Is it classification? No

Task: 有什么可以作为早餐的食物，不含鸡蛋却富含蛋白质，并且热量在700-1000卡路里之间？
Is it classification? No

Task: 回答以下多项选择题，并选择A、B、C或D作为最终答案。
Is it classification? Yes

Task: 判断三段论法是否在逻辑上成立。
Is it classification? Yes

Task: 个人和组织可以采取哪些措施来减少无意识偏见？
Is it classification? No

Task: 有哪些方法可以帮助减压？
Is it classification? No

Task: 从一组数字中找出最大的那个，并直接输出该数字。
Is it classification? Yes

Task: 将文本中的<mask>标记替换为符合上下文的正确单词。每个<mask>标记可以用多个词语替代。
Is it classification? No

Task: 根据给定的事实写一封求职信。
Is it classification? No

Task: 为什么最好在本项目中使用C语言？
Is it classification? No

Task: 编写一个程序计算从k到n的整数之和。
Is it classification? No

Task: 在该任务中，你需要比较两个句子的含义，判断它们是否相同。输出“是”或“否”。
Is it classification? Yes

Task: 为了使这些词对保持相同的类比关系，请写出第四个词。
Is it classification? No

Task: 给定一组数字，找出所有和为给定数字的子集。
Is it classification? No

Task: """

output_first_template_for_clf = """Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.

Task: 把句子的情绪分为肯定的、否定的和混合的。
Class label: 混合的
Sentence: 我喜欢这家餐馆的味道，但是他们的服务太慢了。
Class label: 肯定的
Sentence: 我今天过得很愉快。天气很好，我和朋友和家人在一起。
Class label: 否定的
Sentence: 最近的超级英雄电影让我很失望。我不会向任何人推荐它。

Task: 给定一个对话，分类用户是否对服务满意。你应该回答“满意”或“不满意”。
Class label: 满意
对话:
- 代理: 谢谢你的反馈。我们将在未来努力改善我们的服务。
- 客户: 我对你们提供的服务很满意。谢谢你的帮助。
Class label: 不满意
对话:
- 代理: 很抱歉，我们将为您取消该订单，您将在7个工作日内获得退款。
- 客户: 哦，那太花时间了。我希望你能尽快采取行动。

Task: 给出一些政治观点，分类这个人是属于民主党还是共和党。
Class label: 民主党
观点：我认为，无论收入水平如何，每个人都应该获得高质量的医疗保健。
Class label: 共和党
观点：我认为人们应该能够保留更多的血汗钱，不应该被征收高税率。

Task: 告诉我以下电子邮件是否为促销邮件。
Class label: 促销
电子邮件：看看我们精彩的新促销吧！我们为您喜爱的所有产品提供折扣。
Class label: 非促销
电子邮件：希望您一切安好。如有需要，请随时告诉我们。

Task: 检测该 Reddit 线程是否包含仇恨言论。
Class label: 仇恨言论
线程：有色人种都是愚蠢的，不应该被允许投票。
Class label: 非仇恨言论
线程：在烤架上煎牛排的最佳方法。

Task: 文档中的信息是否支持该声明？你可以回答“支持”或“不支持”。
Class label: 不支持
文档：在经历了抵押贷款利率创历史新低、房价飙升至新高的破纪录行情后，美国的房地产市场终于开始放缓。尽管需求和价格涨幅在降温，但任何修正都可能是温和的，房产经济学家和分析师表示。没有人预期会出现类似于大衰退时期那样的价格暴跌。
声明：美国房地产市场即将崩盘。
Class label: 支持
文档：美国房地产市场显露出疲态，许多地区的房屋销售和价格正在放缓。抵押贷款利率在最近几个月急剧上升，待售房屋数量也在增加。这可能是更大规模下跌的开始，一些经济学家预测近期可能会出现房地产崩盘。
声明：美国房地产市场即将崩盘。

Task: 回答以下的多项选择题。选择 A、B、C 或 D 作为最终答案。
Class label: C
问题：德国的首都是哪里？
A. 伦敦
B. 巴黎
C. 柏林
D. 罗马
Class label: D
问题：太阳系中最大的行星是什么？
A) 地球
B) 土星
C) 火星
D) 木星
Class label: A
问题：植物通过光合作用制造食物的过程叫什么？
A) 呼吸作用
B) 发酵
C) 消化
D) 新陈代谢
Class label: B
问题：《了不起的盖茨比》这部小说的作者是谁？
A) 欧内斯特·海明威
B) F·斯科特·菲茨杰拉德
C) J·D·塞林格
D) 马克·吐温

Task: 你需要阅读一段代码并检测是否存在语法错误。如果存在错误，输出true；如果没有错误，输出false。
Class label: true
代码：
def quick_sort(arr):
    if len(arr) < 2
        return arr
Class label: false
代码：
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
     return total / len(numbers)

Task: 以下哪项不是输入类型？
(a) 数字 (b) 日期 (c) 电话号码 (d) 电子邮件地址 (e) 以上所有都是有效输入。
Class label: (e)

Task: """

input_first_template_for_gen = """Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: 哪些锻炼最适合在家减少腹部脂肪？
Output: 
- 仰卧抬腿
- 双腿并拢伸展
- 平板支撑
- 侧平板支撑
- 仰卧起坐

Task: 从段落中提取所有国家名称，并用逗号分隔列出。
示例 1
段落：伊恩·弗莱明的第六本小说《诺博士》讲述了英国情报特工詹姆斯·邦德的故事。这本书是在弗莱明位于牙买加的“金眼”庄园写成的，并于1958年由英国的乔纳森·凯普出版社首次出版。在小说中，邦德调查了在牙买加失踪的两名 MI6 特工，他们当时正在调查诺博士。邦德前往诺博士的加勒比岛屿，并遇见了在那里收集贝壳的亨妮切尔·莱德。他们被俘并带到一座山中奢华的设施内。诺博士的角色受到德国传教士与中国女子之子的影响，灵感来源于萨克斯·罗默的傅满洲故事。尽管《诺博士》在英国遭遇了广泛的负面评价，但在美国却获得了较好的反响。
Output: English（英语）、British（英国）、Jamaica（牙买加）、the United Kingdom（英国）、German（德国）、Chinese（中国）、Britain（英国）、the United States（美国）

Task: 将 85 华氏度转换为摄氏度。
Output: 85°F = 29.44°C

Task: 将给定列表按升序排序。
示例 1
列表：[10, 92, 2, 5, -4, 92, 5, 101]
Output: [-4, 2, 5, 5, 10, 92, 92, 101]
示例 2
输入 2 - 列表：[9.99, 10, -5, -1000, 5e6, 999]
Output: [-1000, -5, 9.99, 10, 999, 5e6]

Task: 为以下句子提供更好、更专业的表述。
示例 1
句子：这所房子的建造质量意外地不太好，购买后你可能需要更多钱来修缮。如果你问我，我建议你考虑其他候选房屋。
Output: 这所房子似乎建造质量不佳，因此你在购买后可能需要花更多钱来修缮。我建议你查看其他房产。
示例 2
句子：只是想告诉你，我们上周做了一个实验，结果非常惊人——语言模型可以自我提升！
Output: 我们上周的实验显示了惊人的结果，证明语言模型可以自我提升。

Task: 阅读以下段落并回答一个关于段落的数学问题。你需要写出计算过程以得到最终答案。
示例 1
段落：美国的枪支暴力每年导致数万人死亡和受伤，并在 2020 年成为 19 岁及以下儿童的主要死亡原因。根据截至 2021 年的最新数据，美国疾病控制与预防中心（CDC）国家卫生统计中心报告，2018 年有 38,390 人死于枪支，其中 24,432 人是自杀。在 1999 年至 2017 年间，每 10 万人中的枪支死亡率从 10.3 上升到 12，每天约有 109 人死亡，总计约 14,542 人死于凶杀，2018 年的凶杀率为每 10 万人 11.9。2010 年，美国有 19,392 起与枪支相关的自杀和 11,078 起与枪支相关的凶杀。2010 年报告涉及步枪的谋杀案为 358 起，涉及手枪的谋杀案为 6,009 起；另外还有 1,939 起使用未指定类型枪支的案件。2011 年，使用枪支实施的致命和非致命暴力犯罪总计 478,400 起。
问题：2018 年比 2010 年多了多少与枪支相关的死亡？
Output: 
38390 - (19392 + 11078) = 38390 - 30470 = 7920
因此，2018 年比 2010 年多了 7920 人死于枪支。

Task: 编写 Python 代码解决此 leetcode 问题。
示例 1
问题：给你两个非空链表，分别表示两个非负整数。这些数字以逆序存储，它们的每个节点包含一个数字。将两个数字相加，并以链表形式返回其和。你可以假设这两个数字不包含任何前导零，除了数字 0 本身。
Output: 
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1 + v2 + carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

Task: 解方程并找出 X 的值。显示计算步骤。
示例 1
方程：10X + 5 = 10
Output: 10X = 5，X = 0.5
示例 2
方程：X + Y + 120 = 100
Output: X + Y = -20，X = -20 - Y

Task: 编写一个程序来计算从 k 到 n 的整数之和。
Output: 
def sum(k, n):
    total = 0
    for i in range(k, n + 1):
        total += i
    return total
    
Task: 从给定列表中选出年纪最大的人。
示例 1
列表：George Washington（乔治·华盛顿）、Confucius（孔子）、Michael Jordan（迈克尔·乔丹）、Michelangelo（米开朗基罗）
Output: 孔子
示例 2
列表：Alan Turing（阿兰·图灵）、Geoffrey Hinton（杰弗里·辛顿）、Yann LeCun（扬·勒昆）、Yoshua Bengio（约书亚·本吉奥）
Output: 阿兰·图灵

Task: 通过发送电子邮件给招聘人员拒绝工作机会，并解释原因。
Output: 
Hi [Recruiter]，
非常感谢您慷慨地提供加入贵团队的机会。正如我们讨论的那样，我一直以来都很钦佩贵公司，并且是其产品的忠实支持者。然而，在进一步考虑了我目前的职业发展状况后，我决定接受另一家公司的邀请。
我希望能够与您保持联系，并已开始关注您在[社交媒体平台]上的动态。再次感谢您的时间与考量。
再次感谢，
[Your Name]

Task: """