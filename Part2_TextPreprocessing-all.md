```python
import pandas as pd
import numpy as np
import sklearn
```

## Tokenization


```python
#word Tokenization
```


```python
# the way NLTK dealing with apostrophe
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))  
```

    ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    


```python
from nltk.tokenize import WordPunctTokenizer  
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```

    ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    


```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```

    ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
    


```python
#Penn Treebank Tokenization
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
```


```python
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
```


```python
print(tokenizer.tokenize(text))
```

    ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
    


```python
#Sentence Tokenization
```


```python
from nltk.tokenize import sent_tokenize
```


```python
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))
```

    ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
    


```python
# result when there are some . in the middle of sentence.
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
```

    ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
    


```python
import kss
```


```python
text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))
```

    ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']
    


```python
#exercise
from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))
```

    ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
    


```python
from nltk.tag import pos_tag
```


```python
x=word_tokenize(text)
pos_tag(x)
```




    [('I', 'PRP'),
     ('am', 'VBP'),
     ('actively', 'RB'),
     ('looking', 'VBG'),
     ('for', 'IN'),
     ('Ph.D.', 'NNP'),
     ('students', 'NNS'),
     ('.', '.'),
     ('and', 'CC'),
     ('you', 'PRP'),
     ('are', 'VBP'),
     ('a', 'DT'),
     ('Ph.D.', 'NNP'),
     ('student', 'NN'),
     ('.', '.')]




```python
from konlpy.tag import Okt  
okt=Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

    ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
    


```python
# 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제
import re
text = "I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))
```

     was wondering anyone out there could enlighten this car.
    


```python
from nltk.stem import WordNetLemmatizer
```


```python
n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
```


```python
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
print([n.lemmatize(w) for w in words])
```

    ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
    


```python
n.lemmatize('dies','v')
```




    'die'




```python
#어간 추출
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
```


```python
words=word_tokenize(text)
print(words)
```

    ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
    


```python
print([s.stem(w) for w in words])
```

    ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
    


```python
words=['formalize', 'allowance', 'electricical']
print([s.stem(w) for w in words])
```

    ['formal', 'allow', 'electric']
    


```python
s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])
```

    ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
    


```python
from nltk.stem import LancasterStemmer
l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])
```

    ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
    


```python
#stopwords
from nltk.corpus import stopwords
stopwords.words('english')[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]




```python
stopwords.words('english')
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     "she's",
     'her',
     'hers',
     'herself',
     'it',
     "it's",
     'its',
     'itself',
     'they',
     'them',
     'their',
     'theirs',
     'themselves',
     'what',
     'which',
     'who',
     'whom',
     'this',
     'that',
     "that'll",
     'these',
     'those',
     'am',
     'is',
     'are',
     'was',
     'were',
     'be',
     'been',
     'being',
     'have',
     'has',
     'had',
     'having',
     'do',
     'does',
     'did',
     'doing',
     'a',
     'an',
     'the',
     'and',
     'but',
     'if',
     'or',
     'because',
     'as',
     'until',
     'while',
     'of',
     'at',
     'by',
     'for',
     'with',
     'about',
     'against',
     'between',
     'into',
     'through',
     'during',
     'before',
     'after',
     'above',
     'below',
     'to',
     'from',
     'up',
     'down',
     'in',
     'out',
     'on',
     'off',
     'over',
     'under',
     'again',
     'further',
     'then',
     'once',
     'here',
     'there',
     'when',
     'where',
     'why',
     'how',
     'all',
     'any',
     'both',
     'each',
     'few',
     'more',
     'most',
     'other',
     'some',
     'such',
     'no',
     'nor',
     'not',
     'only',
     'own',
     'same',
     'so',
     'than',
     'too',
     'very',
     's',
     't',
     'can',
     'will',
     'just',
     'don',
     "don't",
     'should',
     "should've",
     'now',
     'd',
     'll',
     'm',
     'o',
     're',
     've',
     'y',
     'ain',
     'aren',
     "aren't",
     'couldn',
     "couldn't",
     'didn',
     "didn't",
     'doesn',
     "doesn't",
     'hadn',
     "hadn't",
     'hasn',
     "hasn't",
     'haven',
     "haven't",
     'isn',
     "isn't",
     'ma',
     'mightn',
     "mightn't",
     'mustn',
     "mustn't",
     'needn',
     "needn't",
     'shan',
     "shan't",
     'shouldn',
     "shouldn't",
     'wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't"]




```python
example = "Family is not an important thing. It's everything."
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
print(word_tokens)
print(result)
```

    ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
    ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
    


```python
#removing stopwords in korean
example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
stop_words=stop_words.split(' ')
word_tokens=word_tokenize(example)
result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
print(word_tokens)
print(result)
```

    ['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
    ['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
    


```python
#.
import re
r=re.compile("a.c")
r.search("kkk")
```


```python
r.search("abc")
```




    <re.Match object; span=(0, 3), match='abc'>




```python
#?
r=re.compile("ab?c")
r.search("abbc")
```


```python
r.search("abc")
```




    <re.Match object; span=(0, 3), match='abc'>




```python
r.search("ac")
```




    <re.Match object; span=(0, 2), match='ac'>




```python
#*
r=re.compile("ab*c")
r.search("a")
```


```python
r.search("abbbc")
```




    <re.Match object; span=(0, 5), match='abbbc'>




```python
#+ 
r=re.compile("ab+c")
r.search("abbbbbbbc")
```




    <re.Match object; span=(0, 9), match='abbbbbbbc'>




```python
r.search("ac")
```


```python
#^
r=re.compile("^a")
r.search("bbc")
```


```python
r.search("ab")
```




    <re.Match object; span=(0, 1), match='a'>




```python
r=re.compile("ab{2,8}c")
r.search("abc")
```


```python
r.search("abbbbbc")
```




    <re.Match object; span=(0, 7), match='abbbbbc'>




```python
r=re.compile("a{2,}bc")
r.search("aaabc")
```




    <re.Match object; span=(0, 5), match='aaabc'>




```python
#[]
r=re.compile("[abc]")
r.search("z")
```


```python
r.search("b")
```




    <re.Match object; span=(0, 1), match='b'>




```python
r=re.compile("[a-z]")
r.search("A")
```


```python
r.search("aBC")
```




    <re.Match object; span=(0, 1), match='a'>




```python
r=re.compile("[^abc]")
r.search("111")
```




    <re.Match object; span=(0, 1), match='1'>




```python
r.match("1ab1")
```




    <re.Match object; span=(0, 1), match='1'>




```python
#re.split()
text="사과 딸기 수박 메론 바나나"
re.split(" ",text)
```




    ['사과', '딸기', '수박', '메론', '바나나']




```python
text="""사과
딸기
수박
메론
바나나"""
re.split("\n",text)
```




    ['사과', '딸기', '수박', '메론', '바나나']




```python
text="사과+딸기+수박+메론+바나나"
re.split("\+",text)
['사과', '딸기', '수박', '메론', '바나나']  
```




    ['사과', '딸기', '수박', '메론', '바나나']




```python
text="사과+딸기+수박+메론+바나나"
re.split("\+",text)
```




    ['사과', '딸기', '수박', '메론', '바나나']




```python
#re.findall()
#정규 표현식과 매치되는 모든 문자열들을 리스트로 리턴
text="""이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""  
re.findall("\d+",text)
```




    ['010', '1234', '1234', '30']




```python
text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
```


```python
re.sub('[^a-zA-Z]',' ',text)
```




    'Regular expression   A regular expression  regex or regexp     sometimes called a rational expression        is  in theoretical computer science and formal language theory  a sequence of characters that define a search pattern '




```python
text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""  
```


```python
re.split('\s+',text)
```




    ['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']




```python
re.findall('\d+',text)
```




    ['100', '101', '102']




```python
re.findall('[A-Z]{4}',text)
```




    ['PROF', 'STUD', 'STUD']




```python
#정규 표현식을 이용한 토큰화
from nltk.tokenize import RegexpTokenizer
```


```python
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```

    ['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
    


```python
tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```

    ["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
    


```python
#gaps=True : 해당 정규 표현식을 토큰으로 나누기 위한 기준으로 사용
```


```python
#Integer encoding
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
```


```python
#tokenization
text=sent_tokenize(text)
print(text)
```

    ['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']
    


```python
# word tokenization
voca={}
sentences=[]
stop_words=set(stopwords.words('english'))

for i in text:
    sentence= word_tokenize(i)
    result=[]
    
    for word in sentence:
        word = word.lower()
        
        if word not in stop_words:
            if len(word)>2:
                result.append(word)
                if word not in voca:
                    voca[word]=0
                voca[word]+=1
    sentences.append(result)
print(sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    


```python
voca
```




    {'barber': 8,
     'person': 3,
     'good': 1,
     'huge': 5,
     'knew': 1,
     'secret': 6,
     'kept': 4,
     'word': 2,
     'keeping': 2,
     'driving': 1,
     'crazy': 1,
     'went': 1,
     'mountain': 1}




```python
voca_sorted=sorted(voca.items(),key=lambda x:x[1],reverse=True)
print(voca_sorted)
```

    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
    


```python
word_to_index={}
i=0
for(word,frequency) in voca_sorted:
    if(frequency>1):
        i+=1
        word_to_index[word]=i
print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
    


```python
voca_size=5
words_frequency=[w for w,c in word_to_index.items() if c>=voca_size+1]
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    


```python
word_to_index['OOV']=len(word_to_index)+1
```


```python
encoded=[]
for s in sentences:
    temp=[]
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)
```

    [[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
    


```python
from collections import Counter
```


```python
print(sentences)
```

    [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
    


```python
words=sum(sentences,[])
```


```python
words
```




    ['barber',
     'person',
     'barber',
     'good',
     'person',
     'barber',
     'huge',
     'person',
     'knew',
     'secret',
     'secret',
     'kept',
     'huge',
     'secret',
     'huge',
     'secret',
     'barber',
     'kept',
     'word',
     'barber',
     'kept',
     'word',
     'barber',
     'kept',
     'secret',
     'keeping',
     'keeping',
     'huge',
     'secret',
     'driving',
     'barber',
     'crazy',
     'barber',
     'went',
     'huge',
     'mountain']




```python
vocab=Counter(words)
```


```python
vocab
```




    Counter({'barber': 8,
             'person': 3,
             'good': 1,
             'huge': 5,
             'knew': 1,
             'secret': 6,
             'kept': 4,
             'word': 2,
             'keeping': 2,
             'driving': 1,
             'crazy': 1,
             'went': 1,
             'mountain': 1})




```python
vocab=vocab.most_common(voca_size)
vocab
```




    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]




```python
word_to_index={}
i=0
for(word,frequency) in vocab:
    i+=1
    word_to_index[word]=i
word_to_index
```




    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}




```python
from nltk import FreqDist
```


```python
voca_b = FreqDist(np.hstack(sentences))
voca_b
```




    FreqDist({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, ...})




```python
voca_b=voca_b.most_common(voca_size)
voca_b
```




    [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]




```python
word_to_index={word[0]: index+1 for index,word in enumerate(voca_b)}
```


```python
word_to_index
```




    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}




```python
from tensorflow.keras.preprocessing.text import Tokenizer
```


```python
tokenizer=Tokenizer(num_words=voca_size+1)
tokenizer.fit_on_texts(sentences)
```


```python
print(tokenizer.word_index)
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
    


```python
print(tokenizer.word_counts)
```

    OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
    


```python
print(tokenizer.texts_to_sequences(sentences))
```

    [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
    


```python
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)
words_frequency=[w for w,c in tokenizer.word_index.items() if c>= voca_size+1]
for w in words_frequency:
    del tokenizer.word_index[w]
    del tokenizer.word_counts[w]
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))
```

    {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
    OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
    [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
    


```python
#단어 집합에 없는 단어 OOV : oov_token
```


```python
sentences
```




    [['barber', 'person'],
     ['barber', 'good', 'person'],
     ['barber', 'huge', 'person'],
     ['knew', 'secret'],
     ['secret', 'kept', 'huge', 'secret'],
     ['huge', 'secret'],
     ['barber', 'kept', 'word'],
     ['barber', 'kept', 'word'],
     ['barber', 'kept', 'secret'],
     ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
     ['barber', 'went', 'huge', 'mountain']]




```python
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)
encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)
```

    [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
    


```python
max_len=max(len(item) for item in encoded)
print(max_len)
```

    7
    


```python
#padding
for item in encoded:
    while len(item) <max_len:
        item.append(0)
padded_np=np.array(encoded)
padded_np
```




    array([[ 1,  5,  0,  0,  0,  0,  0],
           [ 1,  8,  5,  0,  0,  0,  0],
           [ 1,  3,  5,  0,  0,  0,  0],
           [ 9,  2,  0,  0,  0,  0,  0],
           [ 2,  4,  3,  2,  0,  0,  0],
           [ 3,  2,  0,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  2,  0,  0,  0,  0],
           [ 7,  7,  3,  2, 10,  1, 11],
           [ 1, 12,  3, 13,  0,  0,  0]])




```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
encoded=tokenizer.texts_to_sequences(sentences)
print(encoded)
```

    [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
    


```python
padded=pad_sequences(encoded)
padded
```




    array([[ 0,  0,  0,  0,  0,  1,  5],
           [ 0,  0,  0,  0,  1,  8,  5],
           [ 0,  0,  0,  0,  1,  3,  5],
           [ 0,  0,  0,  0,  0,  9,  2],
           [ 0,  0,  0,  2,  4,  3,  2],
           [ 0,  0,  0,  0,  0,  3,  2],
           [ 0,  0,  0,  0,  1,  4,  6],
           [ 0,  0,  0,  0,  1,  4,  6],
           [ 0,  0,  0,  0,  1,  4,  2],
           [ 7,  7,  3,  2, 10,  1, 11],
           [ 0,  0,  0,  1, 12,  3, 13]])




```python
padded=pad_sequences(encoded,padding='post')
padded
```




    array([[ 1,  5,  0,  0,  0,  0,  0],
           [ 1,  8,  5,  0,  0,  0,  0],
           [ 1,  3,  5,  0,  0,  0,  0],
           [ 9,  2,  0,  0,  0,  0,  0],
           [ 2,  4,  3,  2,  0,  0,  0],
           [ 3,  2,  0,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  6,  0,  0,  0,  0],
           [ 1,  4,  2,  0,  0,  0,  0],
           [ 7,  7,  3,  2, 10,  1, 11],
           [ 1, 12,  3, 13,  0,  0,  0]])




```python
#limits maximum length
padded = pad_sequences(encoded, padding = 'post', maxlen = 5)
padded
```




    array([[ 1,  5,  0,  0,  0],
           [ 1,  8,  5,  0,  0],
           [ 1,  3,  5,  0,  0],
           [ 9,  2,  0,  0,  0],
           [ 2,  4,  3,  2,  0],
           [ 3,  2,  0,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  6,  0,  0],
           [ 1,  4,  2,  0,  0],
           [ 3,  2, 10,  1, 11],
           [ 1, 12,  3, 13,  0]])




```python
#One-Hot Encoding
from konlpy.tag import Okt  
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다")  
print(token)
```

    ['나', '는', '자연어', '처리', '를', '배운다']
    


```python
word2index={}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
print(word2index)
```

    {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}
    


```python
def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
       return one_hot_vector
```


```python
one_hot_encoding("자연어",word2index)
```




    [0, 0, 1, 0, 0, 0]




```python
text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
```


```python
#keras to_categorical()
from tensorflow.keras.utils import to_categorical
t=Tokenizer()
t.fit_on_texts([text])
print(t.word_index)
```

    {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
    


```python
sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded=t.texts_to_sequences([sub_text])
print(encoded)
```

    [[2, 5, 1, 6, 3, 7]]
    


```python
one_hot=to_categorical(encoded)
print(one_hot)
```

    [[[0. 0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 1.]]]
    


```python
sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
X,y = zip(*sequences) # *를 추가
print(X)
print(y)
```

    ('a', 'b', 'c')
    (1, 2, 3)
    


```python
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>메일 본문</th>
      <th>스팸 메일 유무</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>당신에게 드리는 마지막 혜택!</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내일 뵐 수 있을지 확인 부탁드...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>도연씨. 잘 지내시죠? 오랜만입...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(광고) AI로 주가를 예측할 수 있다!</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=df['메일 본문']
y=df['스팸 메일 유무']
```


```python
print(X)
```

    0          당신에게 드리는 마지막 혜택!
    1      내일 뵐 수 있을지 확인 부탁드...
    2      도연씨. 잘 지내시죠? 오랜만입...
    3    (광고) AI로 주가를 예측할 수 있다!
    Name: 메일 본문, dtype: object
    


```python
print(y)
```

    0    1
    1    0
    2    0
    3    1
    Name: 스팸 메일 유무, dtype: int64
    


```python
ar = np.arange(0,16).reshape((4,4))
X=ar[:, :3]
y=ar[:,3]
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
```


```python
!pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```

    Collecting git+https://github.com/haven-jeon/PyKoSpacing.git
      Cloning https://github.com/haven-jeon/PyKoSpacing.git to c:\users\user\appdata\local\temp\pip-req-build-dhat1nsm
    Requirement already satisfied: tensorflow==2.5.0 in c:\users\user\anaconda3\lib\site-packages (from pykospacing==0.5) (2.5.0)
    Requirement already satisfied: h5py==3.1.0 in c:\users\user\anaconda3\lib\site-packages (from pykospacing==0.5) (3.1.0)
    Collecting argparse>=1.4.0
      Using cached argparse-1.4.0-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: protobuf>=3.9.2 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (3.15.8)
    Requirement already satisfied: keras-preprocessing~=1.1.2 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.1.2)
    Requirement already satisfied: gast==0.4.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (0.4.0)
    Requirement already satisfied: flatbuffers~=1.12.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.12)
    Requirement already satisfied: grpcio~=1.34.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.34.1)
    Requirement already satisfied: termcolor~=1.1.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.1.0)
    Requirement already satisfied: google-pasta~=0.2 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (0.2.0)
    Requirement already satisfied: six~=1.15.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.15.0)
    Requirement already satisfied: wrapt~=1.12.1 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.12.1)
    Requirement already satisfied: typing-extensions~=3.7.4 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (3.7.4.3)
    Requirement already satisfied: keras-nightly~=2.5.0.dev in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (2.5.0.dev2021032900)
    Requirement already satisfied: numpy~=1.19.2 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.19.4+vanilla)
    Requirement already satisfied: astunparse~=1.6.3 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (1.6.3)
    Requirement already satisfied: tensorflow-estimator<2.6.0,>=2.5.0rc0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (2.5.0)
    Requirement already satisfied: wheel~=0.35 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (0.35.1)
    Requirement already satisfied: opt-einsum~=3.3.0 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (3.3.0)
    Requirement already satisfied: absl-py~=0.10 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (0.12.0)
    Requirement already satisfied: tensorboard~=2.5 in c:\users\user\anaconda3\lib\site-packages (from tensorflow==2.5.0->pykospacing==0.5) (2.5.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (0.4.4)
    Requirement already satisfied: setuptools>=41.0.0 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (50.3.1.post20201107)
    Requirement already satisfied: google-auth<2,>=1.6.3 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (1.30.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (0.6.0)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (2.24.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (1.8.0)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (1.0.1)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\user\anaconda3\lib\site-packages (from tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (3.3.4)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\user\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (1.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3.6" in c:\users\user\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\users\user\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (4.2.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\user\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (0.2.8)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (2020.6.20)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\user\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (1.25.11)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\user\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\user\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (3.0.4)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\user\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (3.1.0)
    Requirement already satisfied: pyasn1>=0.1.3 in c:\users\user\anaconda3\lib\site-packages (from rsa<5,>=3.1.4; python_version >= "3.6"->google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow==2.5.0->pykospacing==0.5) (0.4.8)
    Building wheels for collected packages: pykospacing
      Building wheel for pykospacing (setup.py): started
      Building wheel for pykospacing (setup.py): finished with status 'done'
      Created wheel for pykospacing: filename=pykospacing-0.5-py3-none-any.whl size=2255674 sha256=1b8e82f31ca228143d14fda038b9572ee2b7f107e5b9370460fe0c7412ed0df3
      Stored in directory: C:\Users\user\AppData\Local\Temp\pip-ephem-wheel-cache-8iqah4e6\wheels\79\a0\33\16f2cd03d21f76a663f5d69a0b96f0351335385349136fbd03
    Successfully built pykospacing
    Installing collected packages: argparse, pykospacing
    Successfully installed argparse-1.4.0 pykospacing-0.5
    


```python
!pip install git+https://github.com/ssut/py-hanspell.git
```

    Collecting git+https://github.com/ssut/py-hanspell.git
      Cloning https://github.com/ssut/py-hanspell.git to c:\users\user\appdata\local\temp\pip-req-build-9mm6_gvl
    Requirement already satisfied: requests in c:\users\user\anaconda3\lib\site-packages (from py-hanspell==1.1) (2.24.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\anaconda3\lib\site-packages (from requests->py-hanspell==1.1) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\user\anaconda3\lib\site-packages (from requests->py-hanspell==1.1) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\user\anaconda3\lib\site-packages (from requests->py-hanspell==1.1) (1.25.11)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\user\anaconda3\lib\site-packages (from requests->py-hanspell==1.1) (3.0.4)
    Building wheels for collected packages: py-hanspell
      Building wheel for py-hanspell (setup.py): started
      Building wheel for py-hanspell (setup.py): finished with status 'done'
      Created wheel for py-hanspell: filename=py_hanspell-1.1-py3-none-any.whl size=4899 sha256=1baba8800135d850e049595a6c41b10544233b497bfc6ac6a9118efcb8699f3a
      Stored in directory: C:\Users\user\AppData\Local\Temp\pip-ephem-wheel-cache-rh4srbb7\wheels\3f\a5\73\e4d2806ae141d274fdddaabf8c0ed79be9357d36bfdc99e4b4
    Successfully built py-hanspell
    Installing collected packages: py-hanspell
    Successfully installed py-hanspell-1.1
    


```python
from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)
```

    맞춤법 틀리면 왜 안돼? 쓰고 싶은 대로 쓰면 되지
    


```python
!pip install soynlp
```

    Collecting soynlp
      Downloading soynlp-0.0.493-py3-none-any.whl (416 kB)
    Requirement already satisfied: scipy>=1.1.0 in c:\users\user\anaconda3\lib\site-packages (from soynlp) (1.5.2)
    Requirement already satisfied: numpy>=1.12.1 in c:\users\user\anaconda3\lib\site-packages (from soynlp) (1.19.4+vanilla)
    Requirement already satisfied: psutil>=5.0.1 in c:\users\user\anaconda3\lib\site-packages (from soynlp) (5.7.2)
    Requirement already satisfied: scikit-learn>=0.20.0 in c:\users\user\anaconda3\lib\site-packages (from soynlp) (0.23.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn>=0.20.0->soynlp) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\user\anaconda3\lib\site-packages (from scikit-learn>=0.20.0->soynlp) (0.17.0)
    Installing collected packages: soynlp
    Successfully installed soynlp-0.0.493
    


```python
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
```


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
```




    ('2016-10-20.txt', <http.client.HTTPMessage at 0x21fe6d49820>)




```python
corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)
```




    30091




```python
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
```

    training was done. used memory 1.272 Gb
    all cohesion probabilities was computed. # words = 223348
    all branching entropies was computed # words = 361598
    all accessor variety was computed # words = 361598
    


```python
word_score_table["반포한강"].cohesion_forward
```




    0.19841268168224552




```python
word_score_table["반포한강공원"].cohesion_forward
```




    0.37891487632839754




```python
word_score_table["디스"].right_branching_entropy
```




    1.6371694761537934




```python
word_score_table["디스플레"].right_branching_entropy
```




    -0.0




```python
word_score_table["디스플레이"].right_branching_entropy
```




    3.1400392861792916




```python
pip install customized_konlpy
```

    Collecting customized_konlpyNote: you may need to restart the kernel to use updated packages.
      Downloading customized_konlpy-0.0.64-py3-none-any.whl (881 kB)
    Requirement already satisfied: konlpy>=0.4.4 in c:\users\user\anaconda3\lib\site-packages (from customized_konlpy) (0.5.2)
    Requirement already satisfied: Jpype1>=0.6.1 in c:\users\user\anaconda3\lib\site-packages (from customized_konlpy) (1.1.2)
    Requirement already satisfied: numpy>=1.6 in c:\users\user\anaconda3\lib\site-packages (from konlpy>=0.4.4->customized_konlpy) (1.19.4+vanilla)
    Requirement already satisfied: beautifulsoup4==4.6.0 in c:\users\user\anaconda3\lib\site-packages (from konlpy>=0.4.4->customized_konlpy) (4.6.0)
    Requirement already satisfied: tweepy>=3.7.0 in c:\users\user\anaconda3\lib\site-packages (from konlpy>=0.4.4->customized_konlpy) (3.10.0)
    Requirement already satisfied: lxml>=4.1.0 in c:\users\user\anaconda3\lib\site-packages (from konlpy>=0.4.4->customized_konlpy) (4.6.1)
    Requirement already satisfied: colorama in c:\users\user\anaconda3\lib\site-packages (from konlpy>=0.4.4->customized_konlpy) (0.4.4)
    Requirement already satisfied: requests[socks]>=2.11.1 in c:\users\user\anaconda3\lib\site-packages (from tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (2.24.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\user\anaconda3\lib\site-packages (from tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (1.3.0)
    Requirement already satisfied: six>=1.10.0 in c:\users\user\anaconda3\lib\site-packages (from tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (1.15.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\anaconda3\lib\site-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\user\anaconda3\lib\site-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\user\anaconda3\lib\site-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\user\anaconda3\lib\site-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (1.25.11)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6; extra == "socks" in c:\users\user\anaconda3\lib\site-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (1.7.1)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\user\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy>=0.4.4->customized_konlpy) (3.1.0)
    Installing collected packages: customized-konlpy
    Successfully installed customized-konlpy-0.0.64
    
    


```python

```
