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
    [nltk_data]   Unzipping corpora\wordnet.zip.
    




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

```
