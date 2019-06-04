import  jieba
import re
sentence="为什么说连美国，也不敢在中国近海挑衅？看完大快人心_!_美国,中国近海,国际形势"
text = ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence))
print(text)
words=jieba.cut(text)
final_word=" ".join(words)
print(final_word)
# for word in words:
#  full=''.join(re.findall(u'[\u4e00-\u9fff]+', word))
#  words = jieba.cut(sentence)
# print("res",full)