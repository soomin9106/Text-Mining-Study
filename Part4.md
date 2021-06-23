```python
from konlpy.tag import Okt
import re  
okt=Okt()

token=re.sub("(\.)","",
             "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.") 
```


```python
token=okt.morphs(token)
```


```python
word2index={}  
bow=[]  
for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca]=len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index)-1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index=word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index]=bow[index]+1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)  
```

    {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
    


```python
bow
```




    [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]




```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

    [[1 1 2 1 2 1]]
    {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
    


```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```

    [[1 1 1 1 1]]
    {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
    


```python
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```

    [[1 1 1]]
    {'family': 0, 'important': 1, 'thing': 2}
    


```python
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```

    [[1 1 1 1]]
    {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
    


```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

    [[0 1 0 1 0 1 0 1 1]
     [0 0 1 0 0 0 0 1 0]
     [1 0 0 0 1 0 1 0 0]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
    


```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
```

    [[0.         0.46735098 0.         0.46735098 0.         0.46735098
      0.         0.35543247 0.46735098]
     [0.         0.         0.79596054 0.         0.         0.
      0.         0.60534851 0.        ]
     [0.57735027 0.         0.         0.         0.57735027 0.
      0.57735027 0.         0.        ]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
    


```python

```
