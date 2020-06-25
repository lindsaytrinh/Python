
# 1. Data Understanding


```python
import pandas as pd
import numpy as np
import scipy as sp
```


```python
# load the data
df = pd.read_csv("twcs.csv")
df.head()
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>inbound</th>
      <th>created_at</th>
      <th>text</th>
      <th>response_tweet_id</th>
      <th>in_response_to_tweet_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 22:10:47 +0000 2017</td>
      <td>@115712 I understand. I would like to assist y...</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>115712</td>
      <td>True</td>
      <td>Tue Oct 31 22:11:45 +0000 2017</td>
      <td>@sprintcare and how do you propose we do that</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>115712</td>
      <td>True</td>
      <td>Tue Oct 31 22:08:27 +0000 2017</td>
      <td>@sprintcare I have sent several private messag...</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 21:54:49 +0000 2017</td>
      <td>@115712 Please send us a Private Message so th...</td>
      <td>3</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>115712</td>
      <td>True</td>
      <td>Tue Oct 31 21:49:35 +0000 2017</td>
      <td>@sprintcare I did.</td>
      <td>4</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# take a look at the column text
df['text']
```




    0          @115712 I understand. I would like to assist y...
    1              @sprintcare and how do you propose we do that
    2          @sprintcare I have sent several private messag...
    3          @115712 Please send us a Private Message so th...
    4                                         @sprintcare I did.
    5          @115712 Can you please send us a private messa...
    6                  @sprintcare is the worst customer service
    7          @115713 This is saddening to hear. Please shoo...
    8          @sprintcare You gonna magically change your co...
    9          @115713 We understand your concerns and we'd l...
    10         @sprintcare Since I signed up with you....Sinc...
    11         @115713 H there! We'd definitely like to work ...
    12         @115714 y’all lie about your “great” connectio...
    13         @115715 Please send me a private message so th...
    14         @115714 whenever I contact customer support, t...
    15                @115716 What information is incorrect? ^JK
    16         @Ask_Spectrum Would you like me to email you a...
    17         @115716 Our department is part of the corporat...
    18         @Ask_Spectrum I received this from your corpor...
    19                                 @115716 No thank you. ^JK
    20         @Ask_Spectrum The correct way to do it is via ...
    21         @Ask_Spectrum That is INCORRECT information I ...
    22         @115716 The information pertaining to the acco...
    23         actually that's a broken link you sent me and ...
    24         @115717 Hello, My apologies for any frustratio...
    25         Yo @Ask_Spectrum, your customer service reps a...
    26         @115718 I apologize for the inconvenience. I w...
    27         My picture on @Ask_Spectrum pretty much every ...
    28         @115719 Help has arrived! We are sorry to see ...
    29         @VerizonSupport I finally got someone that hel...
                                     ...                        
    2811744    @823860 Hi there Hannah - Please can you send ...
    2811745    @ArgosHelpers ordered items for same day deliv...
    2811746    @823861 Hi Gillian, I am sorry to read this.  ...
    2811747    @116245 when you reserve an item online for st...
    2811748            @385866 Sorry about that Stacey :( Helen.
    2811749    Ahh spent £120 yesterday and now Argos have th...
    2811750    @823862 Hi, wir haben auf deine DN geantwortet...
    2811751    @AskPayPal Könnt ihr mal bitte auf meine DM An...
    2811752    @823863 Sorry to see this. If anything is dama...
    2811753         @115817 seriously ?? https://t.co/i7JhZaQuGg
    2811754    @823864 If you need assistance, please use the...
    2811755    Second day (night) in a row my package is "on ...
    2811756    @823865 If you need assistance, please use the...
    2811757    @115817 @UPSHelp Why does the tracking record ...
    2811758    @823866 当サイトからそのようなメールをお送りすることはございません。当サイトの名をか...
    2811759    いきなり来たんだけど\nなんですかこれ！！？\n\n@120465 https://t.co...
    2811760    @783956 Hi, apologies for the inconvenience. T...
    2811761    @Safaricom_Care It's almost clocking 24hrs sin...
    2811762       @823867 we have replied you via DM.Thanks-Emir
    2811763    Hai @AirAsiaSupport #asking how many days need...
    2811764    @823868 Sorry but kindly try to clear browser,...
    2811765    @AirAsiaSupport \n\nI am unable to do web chec...
    2811766    @134664 Can you Dm us your order number and we...
    2811767    @524544 That's a Peak service. The 09:56 is th...
    2811768    @VirginTrains Hope you are well? Does the 9.30...
    2811769    @823869 Hey, we'd be happy to look into this f...
    2811770    @115714 wtf!? I’ve been having really shitty s...
    2811771    @143549 @sprintcare You have to go to https://...
    2811772    @823870 Sounds delicious, Sarah! 😋 https://t.c...
    2811773    @AldiUK  warm sloe gin mince pies with ice cre...
    Name: text, Length: 2811774, dtype: object




```python
# take a look at the distinct value of author_id
df.author_id.unique()
```




    array(['sprintcare', '115712', '115713', ..., '823868', '823869',
           '823870'], dtype=object)




```python
# summarize all the column
df.describe(include='all')
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>inbound</th>
      <th>created_at</th>
      <th>text</th>
      <th>response_tweet_id</th>
      <th>in_response_to_tweet_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.811774e+06</td>
      <td>2811774</td>
      <td>2811774</td>
      <td>2811774</td>
      <td>2811774</td>
      <td>1771145</td>
      <td>2.017439e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>702777</td>
      <td>2</td>
      <td>2061666</td>
      <td>2782618</td>
      <td>1771145</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>AmazonHelp</td>
      <td>True</td>
      <td>Wed Oct 18 10:15:42 +0000 2017</td>
      <td>@AirAsiaSupport</td>
      <td>96456</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>169840</td>
      <td>1537843</td>
      <td>18</td>
      <td>287</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.504565e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.463141e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.616450e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.665730e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.601652e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.155105e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.507772e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.439805e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.253296e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.220646e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.987950e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.987950e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# count total number of NaN values in each column
df.isnull().sum(axis = 0)
```




    tweet_id                         0
    author_id                        0
    inbound                          0
    created_at                       0
    text                             0
    response_tweet_id          1040629
    in_response_to_tweet_id     794335
    dtype: int64



The data set basically does not have null values in all columns except response_tweet_id and in_response_to_tweet_id. It simply means that the tweet does not have responses. We can leave it like that or replace the null values with 0.

# 2. Clean the data

 With text data, there are some common data cleaning techniques, which are also known as text pre-processing techniques. I will not it down here for furture use. **Those techniques include:**

Make text all lower case

Remove punctuation

Remove numerical values

Remove common non-sesical text

Tokenize text

Remove stop words

**More data cleaning steps after tokenizations:**

Stemming/ lemmatization

Parts of speech tagging

Create bi-grams or tri-grams

Deal with typos...

It depends on the data set to apply those techniques. I will perform the 6 steps that apply to all text and then will consider if I need to do more steps to clean the data.

# # 2.1 Clean the data round 1


```python
# pickle imports
import pickle
```


```python
# Apply the first round of text cleaning techniques
import re
import string

def clean_text_round1(text):
  '''make text lowercase, remove punctuations, remove words that attach with numbers'''
  text = text.lower()
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\w*\d\w*','', text)
  return text

round1 = lambda x: clean_text_round1(x)
```


```python
# select Sprint
df = df[df["author_id"] == "sprintcare"]
df.head()
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>inbound</th>
      <th>created_at</th>
      <th>text</th>
      <th>response_tweet_id</th>
      <th>in_response_to_tweet_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 22:10:47 +0000 2017</td>
      <td>@115712 I understand. I would like to assist y...</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 21:54:49 +0000 2017</td>
      <td>@115712 Please send us a Private Message so th...</td>
      <td>3</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 21:46:24 +0000 2017</td>
      <td>@115712 Can you please send us a private messa...</td>
      <td>5,7</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 22:10:35 +0000 2017</td>
      <td>@115713 This is saddening to hear. Please shoo...</td>
      <td>NaN</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>sprintcare</td>
      <td>False</td>
      <td>Tue Oct 31 20:03:31 +0000 2017</td>
      <td>@115713 We understand your concerns and we'd l...</td>
      <td>12</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's take a look at the updated text
df1 = pd.DataFrame(df.text.apply(round1))
df1.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i understand i would like to assist you we wo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>please send us a private message so that we c...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>can you please send us a private message so t...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>this is saddening to hear please shoot us a d...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>we understand your concerns and wed like for ...</td>
    </tr>
  </tbody>
</table>
</div>



After removing the punctuations, lower the text and remove number, I notice that there are tweets using emojis and I want to utilize them instead of discarding them from the data. I will convert emojis and emoticons (if any) back to text.

# # 2.2 Clean the Emojis and Emoticons


```python
# Install emot library
!pip install emot
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
```

    Requirement already satisfied: emot in c:\programdata\anaconda3\lib\site-packages (2.1)
    


```python
# Create a function to convert emojis into text
def convert_emojis(text):
  for emot in UNICODE_EMO:
    text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return text

# Clean the emojis
df2 = convert_emojis(df1.text)
df2.head()
```




    0     i understand i would like to assist you we wo...
    3     please send us a private message so that we c...
    5     can you please send us a private message so t...
    7     this is saddening to hear please shoot us a d...
    9     we understand your concerns and wed like for ...
    Name: text, dtype: object



Noticing that the data still has emojis and some non-english tweet, I will get rid of them and focus on 1 language (English) only.


```python
# Clean all the symbols that not in alphabet
round2 = lambda x: re.sub(r'[^a-zA-Z ]+', '', x)
df3 = pd.DataFrame(df2.apply(round2))
df3.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i understand i would like to assist you we wo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>please send us a private message so that we c...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>can you please send us a private message so t...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>this is saddening to hear please shoot us a d...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>we understand your concerns and wed like for ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove non-English words
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
```

    [nltk_data] Downloading package words to
    [nltk_data]     C:\Users\Lindsay\AppData\Roaming\nltk_data...
    [nltk_data]   Package words is already up-to-date!
    


```python
round3 = lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) \
         if w.lower() in words or not w.isalpha())
df4 = pd.DataFrame(df3.text.apply(round3))
df4.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i understand i would like to assist you we wou...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>please send us a private message so that we ca...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>can you please send us a private message so th...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>this is saddening to hear please shoot us a so...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>we understand your and wed like for you to ple...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's pickle for later use
df4.to_pickle("corpus1.pkl")
```

## 2.3 Remove stop words, frequent words, and rare words


```python
# Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Lindsay\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
```


```python
df5 = df4["text"].apply(lambda text: remove_stopwords(text))
df5.head()
```




    0    understand would like assist would need get pr...
    3    please send us private message assist click me...
    5          please send us private message gain account
    7                  saddening hear please shoot us look
    9    understand wed like please send us direct mess...
    Name: text, dtype: object




```python
# Top frequent words
from collections import Counter
cnt = Counter()
for text in df5.values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(20)

```




    [('us', 14789),
     ('please', 10448),
     ('send', 7830),
     ('assist', 7280),
     ('message', 4384),
     ('direct', 3852),
     ('hey', 3008),
     ('like', 2915),
     ('help', 2894),
     ('issue', 2558),
     ('hi', 2241),
     ('look', 2230),
     ('would', 2217),
     ('sprint', 1917),
     ('thank', 1616),
     ('know', 1548),
     ('want', 1483),
     ('let', 1370),
     ('hello', 1359),
     ('service', 1274)]




```python
# Remove top 10 frequent words
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

df6 = df5.apply(lambda text: remove_freqwords(text))
df6.head()
```




    0    understand would would need get private link
    3                       private click top profile
    5                            private gain account
    7                       saddening hear shoot look
    9                               understand wed aa
    Name: text, dtype: object




```python
# Remove rare words
n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

df7 = df6.apply(lambda text: remove_rarewords(text))
df7.head()
```




    0    understand would would need get private link
    3                       private click top profile
    5                            private gain account
    7                       saddening hear shoot look
    9                               understand wed aa
    Name: text, dtype: object




```python
df7.describe()
```




    count     22381
    unique    16264
    top            
    freq        725
    Name: text, dtype: object




```python
# Let's pickle this file before we reorganize the data to document-term matrix
df7.to_pickle("corpus2.pkl")
```

## 2.4 Organize the data

### Corpus


```python
df7 = pd.read_pickle("corpus2.pkl")
df7.head()
```




    0    understand would would need get private link
    3                       private click top profile
    5                            private gain account
    7                       saddening hear shoot look
    9                               understand wed aa
    Name: text, dtype: object




```python
# drop old column text in the old dataframe
data = df.drop(columns = ["text", "inbound", "response_tweet_id", "in_response_to_tweet_id"])
data.head()
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:47 +0000 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:54:49 +0000 2017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:46:24 +0000 2017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:35 +0000 2017</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 20:03:31 +0000 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge the data set with the column text from df7
data = pd.concat([data, df7], axis=1)
data.head()
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>created_at</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:47 +0000 2017</td>
      <td>understand would would need get private link</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:54:49 +0000 2017</td>
      <td>private click top profile</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:46:24 +0000 2017</td>
      <td>private gain account</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:35 +0000 2017</td>
      <td>saddening hear shoot look</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 20:03:31 +0000 2017</td>
      <td>understand wed aa</td>
    </tr>
  </tbody>
</table>
</div>



### Document-term Matrix


```python
# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = "english")
data_cv = cv.fit_transform(df7)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = df7.index
data_dtm.head()
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
      <th>aa</th>
      <th>abbreviation</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>abreast</th>
      <th>absolute</th>
      <th>absolutely</th>
      <th>absurd</th>
      <th>acceder</th>
      <th>...</th>
      <th>yes</th>
      <th>yester</th>
      <th>yesterday</th>
      <th>ym</th>
      <th>youd</th>
      <th>youve</th>
      <th>yr</th>
      <th>zero</th>
      <th>zip</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2626 columns</p>
</div>




```python
# Pickle the document-term matrix
data_dtm.to_pickle("dtm.pkl")
pickle.dump(cv, open("cv.pkl", "wb"))
```

# 3. Sentiment analysis


```python
# Create quick lambda functions to find the polarity and subjectivity of each tweet
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob
!pip install textblob
from textblob import TextBlob
```

    Requirement already satisfied: textblob in c:\programdata\anaconda3\lib\site-packages (0.15.3)
    Requirement already satisfied: nltk>=3.1 in c:\programdata\anaconda3\lib\site-packages (from textblob) (3.4)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from nltk>=3.1->textblob) (1.12.0)
    Requirement already satisfied: singledispatch in c:\programdata\anaconda3\lib\site-packages (from nltk>=3.1->textblob) (3.4.0.3)
    


```python
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['text'].apply(pol)
data['subjectivity'] = data['text'].apply(sub)
data.head()
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
      <th>tweet_id</th>
      <th>author_id</th>
      <th>created_at</th>
      <th>text</th>
      <th>polarity</th>
      <th>subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:47 +0000 2017</td>
      <td>understand would would need get private link</td>
      <td>0.00</td>
      <td>0.3750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:54:49 +0000 2017</td>
      <td>private click top profile</td>
      <td>0.25</td>
      <td>0.4375</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 21:46:24 +0000 2017</td>
      <td>private gain account</td>
      <td>0.00</td>
      <td>0.3750</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 22:10:35 +0000 2017</td>
      <td>saddening hear shoot look</td>
      <td>0.00</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>sprintcare</td>
      <td>Tue Oct 31 20:03:31 +0000 2017</td>
      <td>understand wed aa</td>
      <td>0.00</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's pickle df8 for future use
data.to_pickle("data.pkl")
```


```python
# Let's plot the results
import matplotlib.pyplot as plt

plt.scatter(data["polarity"],data["subjectivity"])
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()
```


![png](output_45_0.png)


Looking at the sentiment analysis, I realize that first of all, the customers' tweets are highly objective instead of subjective, meaning that what they wrote was based on their opinions rather than facts. Secondly, in terms of polarity, customers'sentiment seems to split equally between negative and positive. One of the reasons could be Sprint products do not meet a large proportion of their customers' expectation or their customer service is poorly conducted. We need to dive deeper to find out the reason.


```python
# Sentiment of customer over time
plt.plot(data["created_at"], data["polarity"])
```




    [<matplotlib.lines.Line2D at 0x2bea4895710>]



    Error in callback <function install_repl_displayhook.<locals>.post_execute at 0x000002BEAAB17950> (for post_execute):
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\pyplot.py in post_execute()
        107             def post_execute():
        108                 if matplotlib.is_interactive():
    --> 109                     draw_all()
        110 
        111             # IPython >= 2
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\_pylab_helpers.py in draw_all(cls, force)
        130         for f_mgr in cls.get_all_fig_managers():
        131             if force or f_mgr.canvas.figure.stale:
    --> 132                 f_mgr.canvas.draw_idle()
        133 
        134 atexit.register(Gcf.destroy_all)
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\backend_bases.py in draw_idle(self, *args, **kwargs)
       1897         if not self._is_idle_drawing:
       1898             with self._idle_draw_cntx():
    -> 1899                 self.draw(*args, **kwargs)
       1900 
       1901     def draw_cursor(self, event):
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py in draw(self)
        400         toolbar = self.toolbar
        401         try:
    --> 402             self.figure.draw(self.renderer)
        403             # A GUI class may be need to update a window using this draw, so
        404             # don't forget to call the superclass.
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\figure.py in draw(self, renderer)
       1647 
       1648             mimage._draw_list_compositing_images(
    -> 1649                 renderer, self, artists, self.suppressComposite)
       1650 
       1651             renderer.close_group('figure')
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        136     if not_composite or not has_images:
        137         for a in artists:
    --> 138             a.draw(renderer)
        139     else:
        140         # Composite any adjacent images together
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axes\_base.py in draw(self, renderer, inframe)
       2626             renderer.stop_rasterizing()
       2627 
    -> 2628         mimage._draw_list_compositing_images(renderer, self, artists)
       2629 
       2630         renderer.close_group('axes')
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        136     if not_composite or not has_images:
        137         for a in artists:
    --> 138             a.draw(renderer)
        139     else:
        140         # Composite any adjacent images together
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in draw(self, renderer, *args, **kwargs)
       1183         renderer.open_group(__name__)
       1184 
    -> 1185         ticks_to_draw = self._update_ticks(renderer)
       1186         ticklabelBoxes, ticklabelBoxes2 = self._get_tick_bboxes(ticks_to_draw,
       1187                                                                 renderer)
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in _update_ticks(self, renderer)
       1021 
       1022         interval = self.get_view_interval()
    -> 1023         tick_tups = list(self.iter_ticks())  # iter_ticks calls the locator
       1024         if self._smart_bounds and tick_tups:
       1025             # handle inverted limits
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in iter_ticks(self)
        969         self.major.formatter.set_locs(majorLocs)
        970         majorLabels = [self.major.formatter(val, i)
    --> 971                        for i, val in enumerate(majorLocs)]
        972 
        973         minorLocs = self.minor.locator()
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in <listcomp>(.0)
        969         self.major.formatter.set_locs(majorLocs)
        970         majorLabels = [self.major.formatter(val, i)
    --> 971                        for i, val in enumerate(majorLocs)]
        972 
        973         minorLocs = self.minor.locator()
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\category.py in __call__(self, x, pos)
        140             return ""
        141         r_mapping = {v: StrCategoryFormatter._text(k)
    --> 142                      for k, v in self._units.items()}
        143         return r_mapping.get(int(np.round(x)), '')
        144 
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\category.py in <dictcomp>(.0)
        140             return ""
        141         r_mapping = {v: StrCategoryFormatter._text(k)
    --> 142                      for k, v in self._units.items()}
        143         return r_mapping.get(int(np.round(x)), '')
        144 
    

    KeyboardInterrupt: 


    Error in callback <function flush_figures at 0x000002BEAAB842F0> (for post_execute):
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\pylab\backend_inline.py in flush_figures()
        115         # ignore the tracking, just draw and close all figures
        116         try:
    --> 117             return show(True)
        118         except Exception as e:
        119             # safely show traceback if in IPython, else raise
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\pylab\backend_inline.py in show(close, block)
         37             display(
         38                 figure_manager.canvas.figure,
    ---> 39                 metadata=_fetch_figure_metadata(figure_manager.canvas.figure)
         40             )
         41     finally:
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\display.py in display(include, exclude, metadata, transient, display_id, *objs, **kwargs)
        302             publish_display_data(data=obj, metadata=metadata, **kwargs)
        303         else:
    --> 304             format_dict, md_dict = format(obj, include=include, exclude=exclude)
        305             if not format_dict:
        306                 # nothing to display (e.g. _ipython_display_ took over)
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\formatters.py in format(self, obj, include, exclude)
        178             md = None
        179             try:
    --> 180                 data = formatter(obj)
        181             except:
        182                 # FIXME: log the exception
    

    <C:\ProgramData\Anaconda3\lib\site-packages\decorator.py:decorator-gen-9> in __call__(self, obj)
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\formatters.py in catch_format_error(method, self, *args, **kwargs)
        222     """show traceback on failed format call"""
        223     try:
    --> 224         r = method(self, *args, **kwargs)
        225     except NotImplementedError:
        226         # don't warn on NotImplementedErrors
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\formatters.py in __call__(self, obj)
        339                 pass
        340             else:
    --> 341                 return printer(obj)
        342             # Finally look for special method names
        343             method = get_real_method(obj, self.print_method)
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\pylabtools.py in <lambda>(fig)
        242 
        243     if 'png' in formats:
    --> 244         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        245     if 'retina' in formats or 'png2x' in formats:
        246         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))
    

    C:\ProgramData\Anaconda3\lib\site-packages\IPython\core\pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        126 
        127     bytes_io = BytesIO()
    --> 128     fig.canvas.print_figure(bytes_io, **kw)
        129     data = bytes_io.getvalue()
        130     if fmt == 'svg':
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, **kwargs)
       2047                         orientation=orientation,
       2048                         dryrun=True,
    -> 2049                         **kwargs)
       2050                     renderer = self.figure._cachedRenderer
       2051                     bbox_artists = kwargs.pop("bbox_extra_artists", None)
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py in print_png(self, filename_or_obj, *args, **kwargs)
        508 
        509         """
    --> 510         FigureCanvasAgg.draw(self)
        511         renderer = self.get_renderer()
        512 
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py in draw(self)
        400         toolbar = self.toolbar
        401         try:
    --> 402             self.figure.draw(self.renderer)
        403             # A GUI class may be need to update a window using this draw, so
        404             # don't forget to call the superclass.
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\figure.py in draw(self, renderer)
       1647 
       1648             mimage._draw_list_compositing_images(
    -> 1649                 renderer, self, artists, self.suppressComposite)
       1650 
       1651             renderer.close_group('figure')
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        136     if not_composite or not has_images:
        137         for a in artists:
    --> 138             a.draw(renderer)
        139     else:
        140         # Composite any adjacent images together
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axes\_base.py in draw(self, renderer, inframe)
       2626             renderer.stop_rasterizing()
       2627 
    -> 2628         mimage._draw_list_compositing_images(renderer, self, artists)
       2629 
       2630         renderer.close_group('axes')
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        136     if not_composite or not has_images:
        137         for a in artists:
    --> 138             a.draw(renderer)
        139     else:
        140         # Composite any adjacent images together
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         48                 renderer.start_filter()
         49 
    ---> 50             return draw(artist, renderer, *args, **kwargs)
         51         finally:
         52             if artist.get_agg_filter() is not None:
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in draw(self, renderer, *args, **kwargs)
       1183         renderer.open_group(__name__)
       1184 
    -> 1185         ticks_to_draw = self._update_ticks(renderer)
       1186         ticklabelBoxes, ticklabelBoxes2 = self._get_tick_bboxes(ticks_to_draw,
       1187                                                                 renderer)
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in _update_ticks(self, renderer)
       1021 
       1022         interval = self.get_view_interval()
    -> 1023         tick_tups = list(self.iter_ticks())  # iter_ticks calls the locator
       1024         if self._smart_bounds and tick_tups:
       1025             # handle inverted limits
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in iter_ticks(self)
        969         self.major.formatter.set_locs(majorLocs)
        970         majorLabels = [self.major.formatter(val, i)
    --> 971                        for i, val in enumerate(majorLocs)]
        972 
        973         minorLocs = self.minor.locator()
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\axis.py in <listcomp>(.0)
        969         self.major.formatter.set_locs(majorLocs)
        970         majorLabels = [self.major.formatter(val, i)
    --> 971                        for i, val in enumerate(majorLocs)]
        972 
        973         minorLocs = self.minor.locator()
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\category.py in __call__(self, x, pos)
        140             return ""
        141         r_mapping = {v: StrCategoryFormatter._text(k)
    --> 142                      for k, v in self._units.items()}
        143         return r_mapping.get(int(np.round(x)), '')
        144 
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\category.py in <dictcomp>(.0)
        140             return ""
        141         r_mapping = {v: StrCategoryFormatter._text(k)
    --> 142                      for k, v in self._units.items()}
        143         return r_mapping.get(int(np.round(x)), '')
        144 
    

    C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\category.py in _text(value)
        149         if isinstance(value, bytes):
        150             value = value.decode(encoding='utf-8')
    --> 151         elif not isinstance(value, str):
        152             value = str(value)
        153         return value
    

    KeyboardInterrupt: 


# 4. Topic Modeling

## 4.1 Topic Modeling Attempt #1 (All Text)


```python
!pip install gensim
```

    Requirement already satisfied: gensim in c:\users\lindsay\appdata\roaming\python\python37\site-packages (3.8.3)
    Requirement already satisfied: scipy>=0.18.1 in c:\programdata\anaconda3\lib\site-packages (from gensim) (1.2.1)
    Requirement already satisfied: Cython==0.29.14 in c:\users\lindsay\appdata\roaming\python\python37\site-packages (from gensim) (0.29.14)
    Requirement already satisfied: smart-open>=1.8.1 in c:\programdata\anaconda3\lib\site-packages (from gensim) (2.0.0)
    Requirement already satisfied: numpy>=1.11.3 in c:\programdata\anaconda3\lib\site-packages (from gensim) (1.16.2)
    Requirement already satisfied: six>=1.5.0 in c:\programdata\anaconda3\lib\site-packages (from gensim) (1.12.0)
    Requirement already satisfied: boto3 in c:\programdata\anaconda3\lib\site-packages (from smart-open>=1.8.1->gensim) (1.14.7)
    Requirement already satisfied: boto in c:\programdata\anaconda3\lib\site-packages (from smart-open>=1.8.1->gensim) (2.49.0)
    Requirement already satisfied: requests in c:\programdata\anaconda3\lib\site-packages (from smart-open>=1.8.1->gensim) (2.21.0)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in c:\programdata\anaconda3\lib\site-packages (from boto3->smart-open>=1.8.1->gensim) (0.3.3)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in c:\programdata\anaconda3\lib\site-packages (from boto3->smart-open>=1.8.1->gensim) (0.10.0)
    Requirement already satisfied: botocore<1.18.0,>=1.17.7 in c:\programdata\anaconda3\lib\site-packages (from boto3->smart-open>=1.8.1->gensim) (1.17.7)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests->smart-open>=1.8.1->gensim) (1.24.1)
    Requirement already satisfied: idna<2.9,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests->smart-open>=1.8.1->gensim) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in c:\programdata\anaconda3\lib\site-packages (from requests->smart-open>=1.8.1->gensim) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests->smart-open>=1.8.1->gensim) (2019.3.9)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in c:\programdata\anaconda3\lib\site-packages (from botocore<1.18.0,>=1.17.7->boto3->smart-open>=1.8.1->gensim) (2.8.0)
    Requirement already satisfied: docutils<0.16,>=0.10 in c:\programdata\anaconda3\lib\site-packages (from botocore<1.18.0,>=1.17.7->boto3->smart-open>=1.8.1->gensim) (0.14)
    


```python
# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
from gensim import matutils, models
import scipy.sparse

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```


```python
data = pd.read_pickle("dtm.pkl")
data.head()
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
      <th>aa</th>
      <th>abbreviation</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>abreast</th>
      <th>absolute</th>
      <th>absolutely</th>
      <th>absurd</th>
      <th>acceder</th>
      <th>...</th>
      <th>yes</th>
      <th>yester</th>
      <th>yesterday</th>
      <th>ym</th>
      <th>youd</th>
      <th>youve</th>
      <th>yr</th>
      <th>zero</th>
      <th>zip</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2626 columns</p>
</div>




```python
# One of the required inputs is a term-document matrix
tdm = data.transpose()
tdm.head()
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
      <th>0</th>
      <th>3</th>
      <th>5</th>
      <th>7</th>
      <th>9</th>
      <th>11</th>
      <th>13</th>
      <th>743</th>
      <th>746</th>
      <th>749</th>
      <th>...</th>
      <th>2811047</th>
      <th>2811049</th>
      <th>2811122</th>
      <th>2811124</th>
      <th>2811183</th>
      <th>2811185</th>
      <th>2811282</th>
      <th>2811487</th>
      <th>2811689</th>
      <th>2811769</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aa</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>abbreviation</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ability</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>able</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>aboard</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22381 columns</p>
</div>




```python
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
```


```python
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("cv.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
```


```python
# LDA for num_topics = 4
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)
lda.print_topics()
```




    [(0,
      '0.049*"whats" + 0.048*"going" + 0.045*"thank" + 0.036*"thanks" + 0.035*"reaching" + 0.035*"today" + 0.026*"service" + 0.025*"happening" + 0.021*"hi" + 0.020*"long"'),
     (1,
      '0.070*"look" + 0.048*"know" + 0.042*"let" + 0.039*"hi" + 0.038*"wed" + 0.035*"assistance" + 0.033*"hear" + 0.026*"follow" + 0.025*"order" + 0.025*"hello"'),
     (2,
      '0.035*"sprint" + 0.032*"team" + 0.027*"account" + 0.025*"information" + 0.023*"contact" + 0.019*"thank" + 0.019*"hi" + 0.018*"time" + 0.018*"device" + 0.017*"resolution"'),
     (3,
      '0.052*"feel" + 0.043*"way" + 0.039*"sprint" + 0.032*"want" + 0.028*"understand" + 0.028*"link" + 0.027*"hate" + 0.024*"apologize" + 0.023*"customer" + 0.022*"dont"')]



Now that I divide all the text into 4 topics but it does not really make sense to me.

## 4.2 Topic Modeling - Attempt #2 (Nouns Only)


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Lindsay\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
import nltk
nltk.download('averaged_perceptron_tagger')
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\Lindsay\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping taggers\averaged_perceptron_tagger.zip.
    




    True




```python
# Let's create a function to pull out nouns from a string of text
from nltk import word_tokenize, pos_tag

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)
```


```python
# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('corpus2.pkl')
data_clean.head()
```




    0    understand would would need get private link
    3                       private click top profile
    5                            private gain account
    7                       saddening hear shoot look
    9                               understand wed aa
    Name: text, dtype: object




```python
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_clean.apply(nouns))
data_nouns.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>understand link</td>
    </tr>
    <tr>
      <th>3</th>
      <td>click top profile</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gain account</td>
    </tr>
    <tr>
      <th>7</th>
      <td>look</td>
    </tr>
    <tr>
      <th>9</th>
      <td>understand aa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said','hi','hello','thank', 'let', 'today',
                 'look', 'whats', 'check', 'thanks']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.text)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
data_dtmn.head()
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
      <th>aa</th>
      <th>abbreviation</th>
      <th>ability</th>
      <th>aboard</th>
      <th>absolute</th>
      <th>acceder</th>
      <th>accelerate</th>
      <th>acceleration</th>
      <th>accept</th>
      <th>access</th>
      <th>...</th>
      <th>year</th>
      <th>yellow</th>
      <th>yes</th>
      <th>yesterday</th>
      <th>ym</th>
      <th>youd</th>
      <th>youve</th>
      <th>yr</th>
      <th>zip</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1662 columns</p>
</div>




```python
# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
```


```python
# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)
ldan.print_topics()
```




    [(0,
      '0.100*"order" + 0.076*"assistance" + 0.060*"account" + 0.043*"resolution" + 0.040*"link" + 0.037*"data" + 0.035*"ce" + 0.034*"review" + 0.030*"coverage" + 0.028*"area"'),
     (1,
      '0.080*"sprint" + 0.073*"team" + 0.058*"information" + 0.048*"thanks" + 0.043*"contact" + 0.036*"visit" + 0.029*"network" + 0.027*"reach" + 0.025*"inconvenience" + 0.025*"zip"'),
     (2,
      '0.103*"service" + 0.083*"way" + 0.057*"customer" + 0.041*"sprint" + 0.040*"shoot" + 0.033*"hate" + 0.029*"experience" + 0.028*"feel" + 0.027*"change" + 0.023*"opportunity"'),
     (3,
      '0.077*"device" + 0.055*"phone" + 0.054*"feedback" + 0.045*"situation" + 0.031*"hear" + 0.031*"concern" + 0.023*"streets" + 0.023*"type" + 0.021*"attention" + 0.020*"kind"')]



## Topic Modeling - Attempt #3 (Nouns and Adjectives)


```python
# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
```


```python
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(df7.apply(nouns_adj))
data_nouns_adj.head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>understand private link</td>
    </tr>
    <tr>
      <th>3</th>
      <td>private click top profile</td>
    </tr>
    <tr>
      <th>5</th>
      <td>private gain account</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hear shoot look</td>
    </tr>
    <tr>
      <th>9</th>
      <td>understand aa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.text)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna.head()
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
      <th>aa</th>
      <th>abbreviation</th>
      <th>ability</th>
      <th>able</th>
      <th>aboard</th>
      <th>absolute</th>
      <th>absurd</th>
      <th>acceder</th>
      <th>accelerate</th>
      <th>acceleration</th>
      <th>...</th>
      <th>year</th>
      <th>yellow</th>
      <th>yes</th>
      <th>yesterday</th>
      <th>ym</th>
      <th>youd</th>
      <th>youve</th>
      <th>yr</th>
      <th>zip</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2119 columns</p>
</div>




```python
# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())
```


```python
# Let's try 4 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=10)
ldana.print_topics()
```




    [(0,
      '0.041*"thanks" + 0.034*"assistance" + 0.032*"good" + 0.030*"data" + 0.028*"ce" + 0.027*"code" + 0.026*"nearest" + 0.024*"cross" + 0.023*"coverage" + 0.022*"area"'),
     (1,
      '0.060*"team" + 0.056*"way" + 0.042*"contact" + 0.032*"resolution" + 0.028*"information" + 0.028*"number" + 0.028*"account" + 0.025*"feel" + 0.024*"assistance" + 0.024*"hate"'),
     (2,
      '0.063*"service" + 0.060*"device" + 0.047*"understand" + 0.045*"customer" + 0.042*"feedback" + 0.037*"shoot" + 0.036*"sprint" + 0.035*"hear" + 0.025*"phone" + 0.024*"store"'),
     (3,
      '0.058*"sprint" + 0.045*"order" + 0.032*"link" + 0.031*"visit" + 0.031*"happy" + 0.030*"sorry" + 0.030*"situation" + 0.025*"available" + 0.022*"private" + 0.021*"follow"')]



It makes more sense to use nouns and adjectives to run the topic modeling

Topic 0: data service issue

Topic 1: customer service complaint

Topic 2: store feedback

Topic 3: privacy issue
