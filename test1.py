import re

def is_rhotic_accent(pinyin):
  erhua_list = ['ar', 'air', 'anr', 'angr', 
  'aor', 'bar', 'bair', 'banr', 'bangr', 'baor', 'beir', 'benr', 'bengr',
  'bir', 'bianr', 'biaor', 'bier', 'binr', 'bingr', 'bor', 'bur', 'car', 
  'cair', 'canr', 'cangr', 'caor', 'cer', 'cenr', 'cengr', 'char', 'chair',
  'chanr', 'changr', 'chaor', 'cher', 'chenr', 'chengr', 'chir', 'chongr',
  'chour', 'chur', 'chuar', 'chuair', 'chuanr', 'chuangr', 'chuir', 'chunr',
  'chuor', 'cir', 'congr', 'cour', 'cur', 'cuanr', 'cuir', 'cunr', 'cuor', 
  'dar', 'dair', 'danr', 'dangr', 'daor', 'der', 'deir', 'denr', 'dengr',
  'dir', 'diar', 'dianr', 'diaor', 'dier', 'dingr', 'diur', 'dongr', 'dour',
  'dur', 'duanr', 'duir', 'dunr', 'duor', 'eir', 'enr', 'engr', 'err', 'far',
  'fair', 'fanr', 'fangr', 'feir', 'fenr', 'fengr', 'fiaor', 'for', 'four', 
  'fur', 'gar', 'gair', 'ganr', 'gangr', 'gaor', 'ger', 'geir', 'genr', 
  'gengr', 'gongr', 'gour', 'gur', 'guar', 'guair', 'guanr', 'guangr',
  'guir', 'gunr', 'guor', 'har', 'hair', 'hanr', 'hangr', 'haor', 'her', 
  'heir', 'henr', 'hengr', 'hongr', 'hour', 'hur', 'huar', 'huair', 'huanr',
  'huangr', 'huir', 'hunr', 'huor', 'jir', 'jiar', 'jianr', 'jiangr', 'jiaor',
  'jier', 'jinr', 'jingr', 'jiongr', 'jiur', 'jur', 'juanr', 'juer', 'junr', 
  'kar', 'kair', 'kanr', 'kangr', 'kaor', 'ker', 'keir', 'kenr', 'kengr',
  'kiur', 'kongr', 'kour', 'kur', 'kuar', 'kuair', 'kuanr', 'kuangr', 'kuir',
  'kunr', 'kuor', 'lar', 'lair', 'lanr', 'langr', 'laor', 'ler', 'leir', 
  'lengr', 'lir', 'liar', 'lianr', 'liangr', 'liaor', 'lier', 'linr', 
  'lingr', 'liur', 'lor', 'longr', 'lour', 'lur', 'lvr', 'luanr', 
  'lver', 'lunr', 'luor', 'mar', 'mair', 'manr', 'mangr', 'maor', 
  'mer', 'meir', 'menr', 'mengr', 'mir', 'mianr', 'miaor', 'mier', 
  'minr', 'mingr', 'miur', 'mor', 'mour', 'mur', 'nar', 'nair', 
  'nanr', 'nangr', 'naor', 'ner', 'neir', 'nenr', 'nengr', 'nir', 
  'niar', 'nianr', 'niangr', 'niaor', 'nier', 'ninr', 'ningr', 
  'niur', 'nongr', 'nour', 'nur', 'nvr', 'nuanr', 'nver', 'nuor', 
  'or', 'our', 'par', 'pair', 'panr', 'pangr', 'paor', 'peir', 
  'penr', 'pengr', 'pir', 'pianr', 'piaor', 'pier', 'pinr', 
  'pingr', 'por', 'pour', 'pur', 'qir', 'qiar', 'qianr', 
  'qiangr', 'qiaor', 'qier', 'qinr', 'qingr', 'qiongr', 
  'qiur', 'qur', 'quanr', 'quer', 'qunr', 'ranr', 'rangr', 
  'raor', 'rer', 'renr', 'rengr', 'rir', 'rongr', 'rour', 'rur', 
  'ruar', 'ruanr', 'ruir', 'runr', 'ruor', 'sar', 'sair', 'sanr',
  'sangr', 'saor', 'ser', 'seir', 'senr', 'sengr', 'shar', 'shair', 
  'shanr', 'shangr', 'shaor', 'sher', 'sheir', 'shenr', 'shengr', 'shir',
  'shour', 'shur', 'shuar', 'shuair', 'shuanr', 'shuangr', 'shuir', 
  'shunr', 'shuor', 'sir', 'songr', 'sour', 'sur', 'suanr', 'suir',
  'sunr', 'suor', 'tar', 'tair', 'tanr', 'tangr', 'taor', 'ter', 'tengr',
  'tir', 'tianr', 'tiaor', 'tier', 'tingr', 'tongr', 'tour', 'tur', 'tuanr',
  'tuir', 'tunr', 'tuor', 'war', 'wair', 'wanr', 'wangr', 'weir', 'wenr', 
  'wengr', 'wor', 'wur', 'xir', 'xiar', 'xianr', 'xiangr', 'xiaor', 'xier',
  'xinr', 'xingr', 'xiongr', 'xiur', 'xur', 'xuanr', 'xuer', 'xunr', 'yar',
  'yanr', 'yangr', 'yaor', 'yer', 'yir', 'yinr', 'yingr', 'yor', 'yongr', 
  'your', 'yur', 'yuanr', 'yuer', 'yunr', 'zar', 'zair', 'zanr', 'zangr', 
  'zaor', 'zer', 'zeir', 'zenr', 'zengr', 'zhar', 'zhair', 'zhanr', 'zhangr',
  'zhaor', 'zher', 'zheir', 'zhenr', 'zhengr', 'zhir', 'zhongr', 'zhour', 
  'zhur', 'zhuar', 'zhuair', 'zhuanr', 'zhuangr', 'zhuir', 'zhunr', 'zhuor', 
  'zir', 'zongr', 'zour', 'zur', 'zuanr', 'zuir', 'zunr', 'zuor']

  if pinyin[0:-1] in erhua_list:
    return True
  else:
    return False
  
def align_text_phoneme(text_seq, phoneme_seq):
    # for text pinyin
    text_list = re.sub(r'(#\d)', r' \1 ', text_seq)   
    # text_list = re.sub(r'([a-zAZ]+)(\')(s)', r'\1\3', text_list)  # remove the \' from \'s.
    # Characters that are not in the range of the set can be matched by negation, ^ represent negation.
    # \u400e-\u9fa5 : Chinese characters range
    # text_list = re.sub(r'[^a-zA-Z\u400e-\u9fa5 ]', ' ', text_list)  
    # for not chinese and english characters
    # print(text_list)
    text_list = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', text_list) 
    text_list = re.sub(r'\'', r'', text_list)
    # print(text_list)
    # add the blanks for words (whether is chinese characters)
    text_list = re.sub(r'([\u400e-\u9fa5])', r' \1 ', text_list)
    text_list = re.sub(r'\s+', ' ', text_list).strip()
    text_list = re.sub(r'(#)\s(\d)', r'\1\2', text_list)   # combine the split rhythm.
    text_list = text_list.split()
    text_list = [text for text in text_list if text != '']
    # for phoneme
    phoneme_list = re.sub(r'([a-z]+[1-5])', r'/ \1 /', phoneme_seq)
    phoneme_list = re.sub(r'([-])', r'/ \1 /', phoneme_list)
    phoneme_list = re.sub(r'^/ ', '', phoneme_list).strip()
    phoneme_list = re.sub(r' /$', '', phoneme_list).strip()
    phoneme_list = re.sub(r'/ /', '/', phoneme_list)
    phoneme_list = [phoneme.strip() for phoneme in phoneme_list.split('/')]
    phoneme_list = [phoneme for phoneme in phoneme_list if phoneme != '']
    # phoneme_list = phoneme_list.split('/')
    return text_list, phoneme_list




prosody_level = ['#0', '#1', '#2', '#3', '#4']

def is_mandarin(uchar):
  """判断一个unicode是否是汉字"""
  code_point = ord(uchar)
  if code_point >= 0x4e00 and code_point <= 0x9fa5:
    return True
  else:
    return False

def is_mandarin_for_spss(text_string):
  """判断一个传入字符是否是汉字，传入可能是字符串"""
  if len(text_string) > 1:
    return False
  code_point = ord(text_string)
  if code_point >= 0x4e00 and code_point <= 0x9fa5:
    return True
  else:
    return False

def is_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[a-zA-Z]', text_string):
    return True
  else:
	  return False

def is_cmu_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[A-Z]', text_string):
    return True
  else:
	  return False

def is_pinyin_for_spss(text_string):
  """判断一个传入字符是否是拼音"""
  if re.search(r'^[a-z]', text_string):
    return True
  else:
	  return False

def is_punct_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	puncts = [',', '，', ':', '：', '。', '.', '!', '！', '?', '？', ';','；','、']
	if len(text_string) > 1:
		return False
	if text_string in puncts:
		return True
	else:
		return False

def is_prosody_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	prosody_level = ['#1', '#2', '#3', '#4']
	if text_string in prosody_level:
		return True
	else:
		return False

def num_mandarin(text):
  """compute number of mandarin chars in text"""
  i = 0
  for t in text:
    if is_mandarin(t):
      i += 1
  return i


def check_trans_align(text_seq, phoneme_seq):
    """check the wore and pinyin align and generate the pinyin_prosody_dicts"""
    text_list, phoneme_list = align_text_phoneme(text_seq, phoneme_seq)
    
    sequence_pinyin_prosody = []  ## for keep the current sequence's word_pinyin_prosody
    # try:
    zh_en_count = 0   # the count pinyin of cheinese and english
    len_text_list = len(text_list)
    len_phoneme_list = len(phoneme_list)
    s_index =  0
    while s_index < len_text_list:
      word_pinyin_prosody = []  # the word_pinyin_prosody
      ### the text is chinese or english
      c_index = s_index
      ### for the word is chinese or english  
      if zh_en_count < len_phoneme_list:
        c_phoneme = phoneme_list[zh_en_count]
      # the text still has word, but the pinyin no more
      elif is_mandarin_for_spss(text_list[s_index]) or is_english_for_spss(text_list[s_index]):
        raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
      # print(text_list[s_index])
      # print(c_phoneme)
      if is_mandarin_for_spss(text_list[s_index]):
        ### for 儿化音
        if zh_en_count < len_phoneme_list  and s_index+1 < len_text_list:
          if is_rhotic_accent(c_phoneme):              
            s_index += 1 # skip the next "儿"
        # print(text_list[c_index:s_index+1], c_phoneme, c_phoneme[:-1])

        # add tone
        # tone = c_phoneme[-1]
        # tmp_pinyin = []
        # for item in self.dict_pinyin[c_phoneme[:-1]]:
        #   tmp_pinyin.append(item+tone)
        word_pinyin_prosody.append(text_list[c_index:s_index+1])
        # origin 
        # word_pinyin_prosody.append(self.dict_pinyin[c_phoneme[:-1]])
        # add tone 
        word_pinyin_prosody.append(c_phoneme)
        zh_en_count += 1
        # print(word_pinyin_prosody)
      elif is_english_for_spss(text_list[s_index]):
        # print(text_list[c_index:s_index+1], c_phoneme)
        c_phoneme = ' ' + c_phoneme + ' '
        word_pinyin_prosody.append(text_list[c_index:s_index+1])
        # 重音变成0
        # c_phoneme = re.sub(r'(\w)\d', r'\1', c_phoneme)
        # c_phoneme = re.sub(r'(\w)\s', r'\1 0 ', c_phoneme)
        # c_phoneme = re.sub(r'(\w)\s(\d)', r'\1\2', c_phoneme)
        c_phoneme = re.sub(r'\.', r'', c_phoneme)

        word_pinyin_prosody.append(c_phoneme.split())
        zh_en_count += 1
      else:
        word_pinyin_prosody.append(text_list[c_index:s_index+1])
      ## sequence_pinyin_prosody add word_pinyin_prosody
      sequence_pinyin_prosody.append(word_pinyin_prosody)
      s_index += 1
    # the text no more, but the pinyin still has text
    if zh_en_count != len_phoneme_list:
      raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
    # except Exception as e:
    #   print(e)
    #   raise ValueError(f"text_list and phoneme_list not align!\n{text_list}\n{phoneme_list}")
    
    return sequence_pinyin_prosody

text_seq = "遛弯儿#2，都得#2躲远点#4。"
phoneme_seq = "liu4 wanr1 dou1 dei3 duo2 yuan2 dian3"
# text_list, phoneme_list = align_text_phoneme(text_seq, phoneme_seq)
results = check_trans_align(text_seq, phoneme_seq)
print(results)

for item in results:
  tmp_text = ''.join(item[0])
  print(tmp_text)

# print(text_list, phoneme_list)

exit()

def is_rhotic_accent(pinyin):
  erhua_list = ['ar', 'air', 'anr', 'angr', 
  'aor', 'bar', 'bair', 'banr', 'bangr', 'baor', 'beir', 'benr', 'bengr',
  'bir', 'bianr', 'biaor', 'bier', 'binr', 'bingr', 'bor', 'bur', 'car', 
  'cair', 'canr', 'cangr', 'caor', 'cer', 'cenr', 'cengr', 'char', 'chair',
  'chanr', 'changr', 'chaor', 'cher', 'chenr', 'chengr', 'chir', 'chongr',
  'chour', 'chur', 'chuar', 'chuair', 'chuanr', 'chuangr', 'chuir', 'chunr',
  'chuor', 'cir', 'congr', 'cour', 'cur', 'cuanr', 'cuir', 'cunr', 'cuor', 
  'dar', 'dair', 'danr', 'dangr', 'daor', 'der', 'deir', 'denr', 'dengr',
  'dir', 'diar', 'dianr', 'diaor', 'dier', 'dingr', 'diur', 'dongr', 'dour',
  'dur', 'duanr', 'duir', 'dunr', 'duor', 'eir', 'enr', 'engr', 'err', 'far',
  'fair', 'fanr', 'fangr', 'feir', 'fenr', 'fengr', 'fiaor', 'for', 'four', 
  'fur', 'gar', 'gair', 'ganr', 'gangr', 'gaor', 'ger', 'geir', 'genr', 
  'gengr', 'gongr', 'gour', 'gur', 'guar', 'guair', 'guanr', 'guangr',
  'guir', 'gunr', 'guor', 'har', 'hair', 'hanr', 'hangr', 'haor', 'her', 
  'heir', 'henr', 'hengr', 'hongr', 'hour', 'hur', 'huar', 'huair', 'huanr',
  'huangr', 'huir', 'hunr', 'huor', 'jir', 'jiar', 'jianr', 'jiangr', 'jiaor',
  'jier', 'jinr', 'jingr', 'jiongr', 'jiur', 'jur', 'juanr', 'juer', 'junr', 
  'kar', 'kair', 'kanr', 'kangr', 'kaor', 'ker', 'keir', 'kenr', 'kengr',
  'kiur', 'kongr', 'kour', 'kur', 'kuar', 'kuair', 'kuanr', 'kuangr', 'kuir',
  'kunr', 'kuor', 'lar', 'lair', 'lanr', 'langr', 'laor', 'ler', 'leir', 
  'lengr', 'lir', 'liar', 'lianr', 'liangr', 'liaor', 'lier', 'linr', 
  'lingr', 'liur', 'lor', 'longr', 'lour', 'lur', 'lvr', 'luanr', 
  'lver', 'lunr', 'luor', 'mar', 'mair', 'manr', 'mangr', 'maor', 
  'mer', 'meir', 'menr', 'mengr', 'mir', 'mianr', 'miaor', 'mier', 
  'minr', 'mingr', 'miur', 'mor', 'mour', 'mur', 'nar', 'nair', 
  'nanr', 'nangr', 'naor', 'ner', 'neir', 'nenr', 'nengr', 'nir', 
  'niar', 'nianr', 'niangr', 'niaor', 'nier', 'ninr', 'ningr', 
  'niur', 'nongr', 'nour', 'nur', 'nvr', 'nuanr', 'nver', 'nuor', 
  'or', 'our', 'par', 'pair', 'panr', 'pangr', 'paor', 'peir', 
  'penr', 'pengr', 'pir', 'pianr', 'piaor', 'pier', 'pinr', 
  'pingr', 'por', 'pour', 'pur', 'qir', 'qiar', 'qianr', 
  'qiangr', 'qiaor', 'qier', 'qinr', 'qingr', 'qiongr', 
  'qiur', 'qur', 'quanr', 'quer', 'qunr', 'ranr', 'rangr', 
  'raor', 'rer', 'renr', 'rengr', 'rir', 'rongr', 'rour', 'rur', 
  'ruar', 'ruanr', 'ruir', 'runr', 'ruor', 'sar', 'sair', 'sanr',
  'sangr', 'saor', 'ser', 'seir', 'senr', 'sengr', 'shar', 'shair', 
  'shanr', 'shangr', 'shaor', 'sher', 'sheir', 'shenr', 'shengr', 'shir',
  'shour', 'shur', 'shuar', 'shuair', 'shuanr', 'shuangr', 'shuir', 
  'shunr', 'shuor', 'sir', 'songr', 'sour', 'sur', 'suanr', 'suir',
  'sunr', 'suor', 'tar', 'tair', 'tanr', 'tangr', 'taor', 'ter', 'tengr',
  'tir', 'tianr', 'tiaor', 'tier', 'tingr', 'tongr', 'tour', 'tur', 'tuanr',
  'tuir', 'tunr', 'tuor', 'war', 'wair', 'wanr', 'wangr', 'weir', 'wenr', 
  'wengr', 'wor', 'wur', 'xir', 'xiar', 'xianr', 'xiangr', 'xiaor', 'xier',
  'xinr', 'xingr', 'xiongr', 'xiur', 'xur', 'xuanr', 'xuer', 'xunr', 'yar',
  'yanr', 'yangr', 'yaor', 'yer', 'yir', 'yinr', 'yingr', 'yor', 'yongr', 
  'your', 'yur', 'yuanr', 'yuer', 'yunr', 'zar', 'zair', 'zanr', 'zangr', 
  'zaor', 'zer', 'zeir', 'zenr', 'zengr', 'zhar', 'zhair', 'zhanr', 'zhangr',
  'zhaor', 'zher', 'zheir', 'zhenr', 'zhengr', 'zhir', 'zhongr', 'zhour', 
  'zhur', 'zhuar', 'zhuair', 'zhuanr', 'zhuangr', 'zhuir', 'zhunr', 'zhuor', 
  'zir', 'zongr', 'zour', 'zur', 'zuanr', 'zuir', 'zunr', 'zuor']

  if pinyin[0:-1] in erhua_list:
    return True
  else:
    return False

def process_text_with_pinyin(text, pinyin_line):
    text = re.sub(r'(#\d)', ' \1 ', text).strip()
    characters = list(text)
    pinyins = pinyin_line.split()
    
    result = []
    pinyin_index = 0
    
    for char in characters:
        if char in "，。？！；：":  # Add other punctuation as needed
            result.append([char, ""])
        elif char.startswith('#'):
            # Handle prosodic markers
            result.append([char, ""])
        else:
            # Handle normal characters and erhua
            if pinyin_index < len(pinyins):
                current_pinyin = pinyins[pinyin_index]
                # Check if the character potentially starts an erhua sequence
                if pinyin_index + 1 < len(pinyins) and is_rhotic_accent(current_pinyin + pinyins[pinyin_index + 1]):
                    # Combine the character with the next one if it's an erhua
                    char += characters[pinyin_index + 1]
                    current_pinyin += pinyins[pinyin_index + 1]
                    pinyin_index += 1  # Skip the next character as it's part of the current one
                result.append([char, current_pinyin])
                pinyin_index += 1
            else:
                # No more pinyin available, just add the character
                result.append([char, ""])
    
    return result

# Example usage
text = "遛弯儿#2，都得#2躲远点#4。"
pinyin_line = "liu4 wanr1 dou1 dei3 duo2 yuan2 dian3"
output = process_text_with_pinyin(text, pinyin_line)
print(output)


exit()
import re

puncts = [",", ".", ";", "?", "，", "。", "；", "？", "："]
def result2label(pinyin_prosody, pure_text):
  """convert results to labeled text.
    Args:
      pinyin_prosody: llm results which contain pinyin and prosody, egs: "zai4 ci4 #1 jiao1 yi4 shi2 #3 jie4 shao4 ren2 #1 jian4 mou3 #2 he2 ta1 de5 #1 peng2 you5 #1 yang2 mou3 #3 da2 qi3 le5 #1 qi2 ta1 #1 suan4 pan5 #4".
      text: the original text, egs: '再次交易时，介绍人建某和她的朋友杨某打起了其他算盘'.
    Returns: the labeled text, egs: "再次#1交易时#3，介绍人#1建某#2和她的#1朋友#1杨某#3打起了#1其他算盘#4".
  """
  pinyin_prosody = re.sub(r'\s+', ' ', pinyin_prosody).strip()
  pure_text = pure_text.strip()
  pinyin_prosody = pinyin_prosody.split()
  text = [item for item in pure_text]
     
  print(text)
  new_results = []
  len_pinyin_prosody = len(pinyin_prosody)
  j = 0 # index of pinyin_prosody
  for i in range(len(text)):
    # pinyin_prosody is out of range
    if j >= len_pinyin_prosody:
      break

    c_text = text[i]
    new_results.append(c_text)
    # punct has no corresponding pinyin and prosody
    if c_text in puncts:
       continue
    
    c_pinyin_prosody = pinyin_prosody[j]
    if j+1 < len_pinyin_prosody:
      next_pinyin_prosody = pinyin_prosody[j+1]
    # print(c_text, c_pinyin_prosody, next_pinyin_prosody)
    # if current text has prosody, add it
    if next_pinyin_prosody in ["#1", "#2", "#3", "#4"]:
      new_results.append(next_pinyin_prosody)
      j += 1
    j += 1
  return ''.join(new_results)

pinyin_prosody = "zai4 ci4 #1 jiao1 yi4 shi2 #3 jie4 shao4 ren2 #1 jian4 mou3 #2 he2 ta1 de5 #1 peng2 you5 #1 yang2 mou3 #3 da2 qi3 le5 #1 qi2 ta1 #1 suan4 pan5 #4"  
text = "再次交易时，介绍人建某和她的朋友杨某打起了其他算盘"
a = result2label(pinyin_prosody, text)
print(a)

exit()
for i in range(10):
    print(f"i1: {i}")
    if i==3:
        i += 10
    print(f"i2: {i}")
exit()
import json
data = [{
    "name": "John",
    "age": 30,
    "city": "New York"
}]

# json_string = json.dumps(data, indent=4)
# print(json_string)
a = [0,1,2,3,4,5,6,7]
print(a[:4])
print(a[4:])

exit()
dict_tmp = []

for i in range(10):
    tmp_dict = {}
    tmp_dict['conversations'] = []
    
    sub_tmp_dict1 = {"from": "system", 
                     "value": "asasda1"}
    sub_tmp_dict2 = {"from": "system", 
                     "value": "Question:" + " asasda2"}
    sub_tmp_dict3 = {"from": "system", 
                     "value": "Answer:" + " asasda3"}
    tmp_dict['conversations'].append((sub_tmp_dict1, sub_tmp_dict2, sub_tmp_dict3))
    dict_tmp.append(tmp_dict)

with open('data.json', 'w') as f:
    json.dump(dict_tmp, f, indent=4)
