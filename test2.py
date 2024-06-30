import re

item = 'huar4'
erhua_pinyin = item[:-2] + item[-1]
print(erhua_pinyin)
exit()
def is_mandarin_for_spss(text_string):
  """判断一个传入字符是否是汉字，传入可能是字符串"""
  if len(text_string) > 1:
    return False
  code_point = ord(text_string)
  if code_point >= 0x4e00 and code_point <= 0x9fff:
    return True
  else:
    return False
  
if is_mandarin_for_spss("您"):
    print("是汉字")
exit()
sent = "1111 他们担心你去上学校，hello world，Mr. Wang. We are good. 你好呀，hello world"

sid = re.search(r'^(\d+_)*\d+', sent)
if sid is None:
    raise ValueError(f"Check the sentence whether exists sid.\n{sent}")
sid = sid.group()
sent_cont = re.sub(r'^(\d+_)*\d+\s+', '', sent).strip()
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()

### preprocess the text
sent_cont = re.sub(r'#4', r'', sent_cont)
# Mrs.#1 dr.#1
sent_cont = re.sub(r'[,;:.?!，。？；：、！](#\d)', r'\1', sent_cont)
# #3:#3，   #3；#3；
sent_cont = re.sub(r'(#\d)\s?(#\d)+', r'\1', sent_cont)
sent_cont = re.sub(r'[,;:.?!，。？；：、！]+([,;:.?!，。？；：、！])', r'\1', sent_cont)
sent_cont = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'', sent_cont)
sent_cont = re.sub(r'[,;:.?!，。？；：、！]$', r'', sent_cont)
print(sent_cont)

### generate the question list
sent_cont = re.sub(r'(#\d)', r' \1 ', sent_cont)   
sent_cont = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', sent_cont) 
# merge the Mr. Mrs. etc.
sent_cont = re.sub(r'Mr .', r'Mr.', sent_cont)
sent_cont = re.sub(r'Mrs .', r'Mrs.', sent_cont)

# print(text_list)
# add the blanks for words (whether is chinese characters)
sent_cont = re.sub(r'([\u400e-\u9fa5])', r' \1 ', sent_cont)
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
sent_cont = re.sub(r'(#)\s(\d)', r'\1\2', sent_cont)   # combine the split rhythm.

sent_list = sent_cont.split()
sent_list = [text for text in sent_list if text != '']
print(sent_list)
exit()
aa = "%$&"
if '%' in aa:
    print("hahah")
exit()
import re
sent_cont = "腾#3；#3；#3；驷Mr.#1 hello伟#3；#4。"
sent_cont = re.sub(r'[,;:.?!，。？；：、！](#\d)', r'\1', sent_cont)
# #3:#3，   #3；#3；
sent_cont = re.sub(r'(#\d)\s?(#\d)+', r'\1', sent_cont)
print(sent_cont)
sent_cont = re.sub(r'[,;:.?!，。？；：、！]+([,;:.?!，。？；：、！])', r'\1', sent_cont)
text_seq = re.sub(r'#\d[,;:.?!，。？；：、！]$', r'', sent_cont)
print(text_seq)
exit()
prosody = "%[1]"
prosody = re.sub(r'([/%$&])', r' \1 ', prosody)
prosody_list = prosody.split()
prosody_list = [text for text in prosody_list if text != '']
print(prosody)
exit()
question = "包[1|2]括[1|2] Pontifex,Pontifex二之后更[1|2]名为[1|2|3] Bridge Construction Set,以及Bridge It"

sent_cont = re.sub(r'\[[1-9|]+[1-9]\]', r' ', question)
print(sent_cont)
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
sent_cont = re.sub(r'[,;:.?!，。？；：、！]$', r'', sent_cont)
sent_cont = re.sub(r'(#\d)', r' \1 ', sent_cont)   
sent_cont = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', sent_cont) 
sent_cont = re.sub(r'\'', r'', sent_cont)
# add the blanks for words (whether is chinese characters)
sent_cont = re.sub(r'([\u400e-\u9fa5])', r' \1 ', sent_cont)
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
sent_cont = re.sub(r'(#)\s(\d)', r'\1\2', sent_cont)   # combine the split rhythm.
sent_list = sent_cont.split()
sent_list = [text for text in sent_list if text != '']
print(len(sent_list))
print(sent_list)

answer = "/[2]%[1]&/%&/%/[2]/%[2]%%&//%%/"
# split first
answer = re.sub(r'([/%$&])', r' \1 ', answer)
print(answer)
# merge
answer = re.sub(r'([/%$&])\s(\[[1-9]\])', r'\1\2', answer)
print(answer)
answer_list = answer.split()
answer_list = [text for text in answer_list if text != '']
print(len(answer_list))
print(answer_list)

exit()
import re
sent_cont = "你好呀，hello world"
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()

# strip prosody, for english has space 
# sent_cont = re.sub(r'#\d', r' ', sent_cont)
# sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
# sent_cont = re.sub(r'([^a-zA-Z0-2])\s+', r'\1', sent_cont) 
# sent_cont = re.sub(r'\s+([^a-zA-Z0-2])', r'\1', sent_cont)
sent_cont = re.sub(r'[,;:.?!，。？；：、！]$', r'', sent_cont)

sent_cont = re.sub(r'(#\d)', r' \1 ', sent_cont)   
# text_list = re.sub(r'([a-zAZ]+)(\')(s)', r'\1\3', text_list)  # remove the \' from \'s.
# Characters that are not in the range of the set can be matched by negation, ^ represent negation.
# \u400e-\u9fa5 : Chinese characters range
# text_list = re.sub(r'[^a-zA-Z\u400e-\u9fa5 ]', ' ', text_list)  
# for not chinese and english characters
# print(text_list)
sent_cont = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', sent_cont) 
sent_cont = re.sub(r'\'', r'', sent_cont)
# print(text_list)
# add the blanks for words (whether is chinese characters)
sent_cont = re.sub(r'([\u400e-\u9fa5])', r' \1 ', sent_cont)
sent_cont = re.sub(r'\s+', ' ', sent_cont).strip()
sent_cont = re.sub(r'(#)\s(\d)', r'\1\2', sent_cont)   # combine the split rhythm.

sent_cont = sent_cont.split()
sent_cont = [text for text in sent_cont if text != '']
print(sent_cont)

exit()
a = [1,2,3,4,5]
b = [10,2,3,4,5]
print(a[0:2], a[2:4])

import re
question = "在狱中[zhong1|zhong4]，张明宝悔恨交加，写了[le5|liao3|liao4]一份忏悔书"
question = re.sub(r'\[([a-z]+\d\|?)+\]', r'', question)
print(question)

exit()
import re

# Unicode code point
code_point = 0x20BBE

# Convert to corresponding character
character = chr(code_point)

print(character)


exit()
sent_cont = "他们#1担[dan1]心#3，野象#1“搬家”后#3可能#1重[chong2]返#4"
sent_cont = re.sub(r'\[\w+\d\]', '', sent_cont).strip()
print(sent_cont)
exit()
def extract_boundaries(text):
    """提取韵律边界的索引位置及其类型"""
    boundaries = []
    current_index = 0
    i = 0
    while i < len(text):
        if text[i] == '#':
            # 提取数字作为边界类型
            boundary_type = ''
            i += 1
            while i < len(text) and text[i].isdigit():
                boundary_type += text[i]
                i += 1
            boundaries.append((current_index, int(boundary_type)))
        else:
            if text[i] != ' ':
                current_index += 1
            i += 1
    return boundaries

def calculate_metrics(true_text, pred_text):
    """计算各级别韵律边界的Precision, Recall和F1分数，以及块准确率"""
    true_boundaries = extract_boundaries(true_text)
    pred_boundaries = extract_boundaries(pred_text)
    
    true_dict = {}
    pred_dict = {}
    
    # 将边界按类型分类
    for index, btype in true_boundaries:
        if btype not in true_dict:
            true_dict[btype] = set()
        true_dict[btype].add(index)
    
    for index, btype in pred_boundaries:
        if btype not in pred_dict:
            pred_dict[btype] = set()
        pred_dict[btype].add(index)
    
    results = {}
    
    # 计算每个类型的Precision, Recall, F1
    all_types = set(true_dict.keys()).union(set(pred_dict.keys()))
    for btype in all_types:
        TP = len(true_dict.get(btype, set()) & pred_dict.get(btype, set()))
        FP = len(pred_dict.get(btype, set()) - true_dict.get(btype, set()))
        FN = len(true_dict.get(btype, set()) - pred_dict.get(btype, set()))
        
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        results[btype] = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }
    
    # 计算块准确率
    total_blocks = len(true_boundaries)
    correct_blocks = sum(1 for t, p in zip(true_boundaries, pred_boundaries) if t == p)
    block_accuracy = correct_blocks / total_blocks if total_blocks > 0 else 0
    
    results['Block Accuracy'] = block_accuracy
    
    return results

# 示例数据
# true_text = "我#1爱#2中国#3。"
# pred_text = "我爱#1中国#2。"

true_text = "我#1爱#1中国#3，你好呢#3。"
pred_text = "我爱#1中国#1，你好呢#3。"
# 计算指标
results = calculate_metrics(true_text, pred_text)

# 打印结果
for key, value in results.items():
    if isinstance(value, dict):
        print(f"Type #{key}:")
        for metric, score in value.items():
            print(f"  {metric}: {score:.2f}")
    else:
        print(f"Block Accuracy: {value:.2f}")
