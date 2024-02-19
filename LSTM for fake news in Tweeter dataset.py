import numpy as np
import pandas as pd
import sklearn
import string
import re
from textblob import TextBlob
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
from nltk.probability import FreqDist
from sklearn.decomposition import LatentDirichletAllocation,NMF
#import spacy
#from wordcloud import WordCloud,STOPWORDS
from collections import Counter,defaultdict
#from PIL import Image
#from transformers import BertTokenizer,BertForSequenceClassification,AdamW,BertConfig,get_linear_schedule_with_warmup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torchtext import data
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns
import datetime
from sklearn.metrics import accuracy_score,f1_score
import contractions
import time
from nltk.corpus import stopwords
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df=pd.read_csv(r'/home/pegah/Desktop/train_NN.csv')
print(train_df.info())
print(train_df.head(7))
test=pd.read_csv(r'/home/pegah/Desktop/test_NN.csv')
print(test.info())
#train_df.drop(columns=['id','keyword','location'],inplace=True)
#print(train_df[~train_df["location"].isnull()].head(7))
#print(train_df[train_df["target"]==0]["text"].values[1])
#print(train_df[train_df["target"]==1]["text"].values[1])
#text cleaning
train_df["text_clean"]=train_df["text"].apply(lambda x:x.lower())
print(train_df.head(7))
train_df["text_clean"]=train_df["text_clean"].apply(lambda x:contractions.fix(x))
print(train_df["text"][67])
print(train_df["text_clean"][67])
def remove_URL(text):
    return re.sub(r"http?://\S+|www\.\S+","",text)
print(train_df["text"][31])
print(train_df["text_clean"][31])
print(train_df["text"][37])
print(train_df["text_clean"][37])
def remove_html(text):
    html=re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html,"",text)
train_df["text_clean"]=train_df["text_clean"].apply(lambda x:remove_html(x))
print(train_df["text"][62])
print(train_df["text_clean"][62])
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r'',text)
train_df["text_clean"]=train_df["text_clean"].apply(lambda x:remove_non_ascii(x))
print(train_df["text"][7586])
print(train_df["text_clean"][7586])
def remove_punct(text):
    return text.translate(str.maketrans('','',string.punctuation))
train_df["text_clean"]=train_df["text_clean"].apply(lambda x:remove_punct(x))
print(train_df["text"][5])
print(train_df["text_clean"][5])
def other_clean(text):  
    sample_typos_slang = { "w/e": "whatever", "usagov": "usa government", "recentlu": "recently", "ph0tos": "photos", "amirite": "am i right", "exp0sed": "exposed", "<3": "love", "luv": "love", "amageddon": "armageddon", "trfc": "traffic", "16yr": "16 year" }
    sample_acronyms = { "mh370": "malaysia airlines flight 370", "okwx": "oklahoma city weather", "arwx": "arkansas weather", "gawx": "georgia weather", "scwx": "south carolina weather", "cawx": "california weather", "tnwx": "tennessee weather", "azwx": "arizona weather", "alwx": "alabama weather", "usnwsgov": "united states national weather service", "2mw": "tomorrow" } 
    sample_abbr = { "$" : " dollar ", "€" : " euro ", "4ao" : "for adults only", "a.m" : "before midday", "a3" : "anytime anywhere anyplace", "aamof" : "as a matter of fact", "acct" : "account", "adih" : "another day in hell", "afaic" : "as far as i am concerned", "afaict" : "as far as i can tell", "afaik" : "as far as i know", "afair" : "as far as i remember", "afk" : "away from keyboard", "app" : "application", "approx" : "approximately", "apps" : "applications", "asap" : "as soon as possible", "asl" : "age, sex, location", "atk" : "at the keyboard", "ave." : "avenue", "aymm" : "are you my mother", "ayor" : "at your own risk", "b&b" : "bed and breakfast", "b+b" : "bed and breakfast", "b.c" : "before christ", "b2b" : "business to business", "b2c" : "business to customer", "b4" : "before", "b4n" : "bye for now", "b@u" : "back at you", "bae" : "before anyone else", "bak" : "back at keyboard", "bbbg" : "bye bye be good", "bbc" : "british broadcasting corporation", "bbias" : "be back in a second", "bbl" : "be back later", "bbs" : "be back soon", "be4" : "before", "bfn" : "bye for now", "blvd" : "boulevard", "bout" : "about", "brb" : "be right back", "bros" : "brothers", "brt" : "be right there", "bsaaw" : "big smile and a wink", "btw" : "by the way", "bwl" : "bursting with laughter", "c/o" : "care of", "cet" : "central european time", "cf" : "compare", "cia" : "central intelligence agency", "csl" : "can not stop laughing", "cu" : "see you", "cul8r" : "see you later", "cv" : "curriculum vitae", "cwot" : "complete waste of time", "cya" : "see you", "cyt" : "see you tomorrow", "dae" : "does anyone else", "dbmib" : "do not bother me i am busy", "diy" : "do it yourself", "dm" : "direct message", "dwh" : "during work hours", "e123" : "easy as one two three", "eet" : "eastern european time", "eg" : "example", "embm" : "early morning business meeting", "encl" : "enclosed", "encl." : "enclosed", "etc" : "and so on", "faq" : "frequently asked questions", "fawc" : "for anyone who cares", "fb" : "facebook", "fc" : "fingers crossed", "fig" : "figure", "fimh" : "forever in my heart", "ft." : "feet", "ft" : "featuring", "ftl" : "for the loss", "ftw" : "for the win", "fwiw" : "for what it is worth", "fyi" : "for your information", "g9" : "genius", "gahoy" : "get a hold of yourself", "gal" : "get a life", "gcse" : "general certificate of secondary education", "gfn" : "gone for now", "gg" : "good game", "gl" : "good luck", "glhf" : "good luck have fun", "gmt" : "greenwich mean time", "gmta" : "great minds think alike", "gn" : "good night", "g.o.a.t" : "greatest of all time", "goat" : "greatest of all time", "goi" : "get over it", "gps" : "global positioning system", "gr8" : "great", "gratz" : "congratulations", "gyal" : "girl", "h&c" : "hot and cold", "hp" : "horsepower", "hr" : "hour", "hrh" : "his royal highness", "ht" : "height", "ibrb" : "i will be right back", "ic" : "i see", "icq" : "i seek you", "icymi" : "in case you missed it", "idc" : "i do not care", "idgadf" : "i do not give a damn fuck", "idgaf" : "i do not give a fuck", "idk" : "i do not know", "ie" : "that is", "i.e" : "that is", "ifyp" : "i feel your pain", "IG" : "instagram", "iirc" : "if i remember correctly", "ilu" : "i love you", "ily" : "i love you", "imho" : "in my humble opinion", "imo" : "in my opinion", "imu" : "i miss you", "iow" : "in other words", "irl" : "in real life", "j4f" : "just for fun", "jic" : "just in case", "jk" : "just kidding", "jsyk" : "just so you know", "l8r" : "later", "lb" : "pound", "lbs" : "pounds", "ldr" : "long distance relationship", "lmao" : "laugh my ass off", "lmfao" : "laugh my fucking ass off", "lol" : "laughing out loud", "ltd" : "limited", "ltns" : "long time no see", "m8" : "mate", "mf" : "motherfucker", "mfs" : "motherfuckers", "mfw" : "my face when", "mofo" : "motherfucker", "mph" : "miles per hour", "mr" : "mister", "mrw" : "my reaction when", "ms" : "miss", "mte" : "my thoughts exactly", "nagi" : "not a good idea", "nbc" : "national broadcasting company", "nbd" : "not big deal", "nfs" : "not for sale", "ngl" : "not going to lie", "nhs" : "national health service", "nrn" : "no reply necessary", "nsfl" : "not safe for life", "nsfw" : "not safe for work", "nth" : "nice to have", "nvr" : "never", "nyc" : "new york city", "oc" : "original content", "og" : "original", "ohp" : "overhead projector", "oic" : "oh i see", "omdb" : "over my dead body", "omg" : "oh my god", "omw" : "on my way", "p.a" : "per annum", "p.m" : "after midday", "pm" : "prime minister", "poc" : "people of color", "pov" : "point of view", "pp" : "pages", "ppl" : "people", "prw" : "parents are watching", "ps" : "postscript", "pt" : "point", "ptb" : "please text back", "pto" : "please turn over", "qpsa" : "what happens", "ratchet" : "rude", "rbtl" : "read between the lines", "rlrt" : "real life retweet", "rofl" : "rolling on the floor laughing", "roflol" : "rolling on the floor laughing out loud", "rotflmao" : "rolling on the floor laughing my ass off", "rt" : "retweet", "ruok" : "are you ok", "sfw" : "safe for work", "sk8" : "skate", "smh" : "shake my head", "sq" : "square", "srsly" : "seriously", "ssdd" : "same stuff different day", "tbh" : "to be honest", "tbs" : "tablespooful", "tbsp" : "tablespooful", "tfw" : "that feeling when", "thks" : "thank you", "tho" : "though", "thx" : "thank you", "tia" : "thanks in advance", "til" : "today i learned", "tl;dr" : "too long i did not read", "tldr" : "too long i did not read","tmb" : "tweet me back", "tntl" : "trying not to laugh", "ttyl" : "talk to you later", "u" : "you", "u2" : "you too", "u4e" : "yours for ever", "utc" : "coordinated universal time", "w/" : "with", "w/o" : "without", "w8" : "wait", "wassup" : "what is up", "wb" : "welcome back", "wtf" : "what the fuck", "wtg" : "way to go", "wtpa" : "where the party at", "wuf" : "where are you from", "wuzup" : "what is up", "wywh" : "wish you were here", "yd" : "yard", "ygtr" : "you got that right", "ynk" : "you never know", "zzz" : "sleeping bored and tired" } 
    sample_typos_slang_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')
    sample_acronyms_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)') 
    sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)') 
    text = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], text) 
    text = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], text) 
    text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text) 
    return text
train_df["text_clean"]=train_df["text_clean"].apply(lambda x:other_clean(x))
print(train_df["text"][4409])
print(train_df["text_clean"][4409])
print("text:",TextBlob("sleapy and there is no plaxe Im gioong to.").correct())
train_df['tokenized']=train_df['text_clean'].apply(word_tokenize)
print(train_df.head(7))
stop=set(stopwords.words('english'))
train_df['stopwords_removed']=train_df['tokenized'].apply(lambda x:[word for word in x if word not in stop])
print(train_df.head(7))
def porter_stemmer(text):
    stemmer=nltk.PorterStemmer()
    stems=[stemmer.stem(i) for i in text]
    return stems
train_df['porter_stemmer']=train_df['stopwords_removed'].apply(lambda x:porter_stemmer(x))
print(train_df.head(7))
def snowball_stemmer(text):
    stemmer=nltk.SnowballStemmer("english")
    stems=[stemmer.stem(i) for i in text]
    return stems
train_df['snowball_stemmer']=train_df['stopwords_removed'].apply(lambda x:snowball_stemmer(x))
print(train_df.head(7))
def lancaster_stemmer(text):
    stemmer=nltk.LancasterStemmer()
    stems=[stemmer.stem(i) for i in text]
    return stems
train_df['lancaster_stemmer']=train_df['stopwords_removed'].apply(lambda x:lancaster_stemmer(x))
print(train_df.head(7))
wordnet_map={"N":wordnet.NOUN,"V":wordnet.VERB,"J":wordnet.ADJ,"R":wordnet.ADV}
train_sents=brown.tagged_sents(categories='news')
t0=nltk.DefaultTagger('NN')
t1=nltk.UnigramTagger(train_sents,backoff=t0)
t2=nltk.BigramTagger(train_sents,backoff=t1)
def pos_tag_wordnet(text,pos_tag_type="pos_tag"):
    pos_tagged_text=t2.tag(text)
    pos_tagged_text=[(word,wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys() else (word,wordnet.NOUN) for (word,pos_tag) in pos_tagged_text]
    return pos_tagged_text
pos_tag_wordnet(train_df['stopwords_removed'][2])
train_df['combined_postag_wnet']=train_df['stopwords_removed'].apply(lambda x:pos_tag_wordnet(x))
print(train_df.head(7))
def lemmatize_word(text):
    lemmatizer=WordNetLemmatizer()
    lemma=[lemmatizer.lemmatize(word,tag) for word,tag in text]
    return lemma
#lemmatization without pos tagging
lemmatizer=WordNetLemmatizer()
train_df['lemmatize_word_wo_pos']=train_df['stopwords_removed'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])
train_df['lemmatize_word_wo_pos']=train_df['lemmatize_word_wo_pos'].apply(lambda x:[word for word in x if word not in stop])
print(train_df.head(7))
print(train_df["combined_postag_wnet"][8])
print(train_df["lemmatize_word_wo_pos"][8])
#lemmatization with pos tag
lemmatizer=WordNetLemmatizer()
train_df['lemmatize_word_wo_pos']=train_df['combined_postag_wnet'].apply(lambda x:lemmatize_word(x))
train_df['lemmatize_word_wo_pos']=train_df['lemmatize_word_wo_pos'].apply(lambda x:[word for word in x if word not in stop])
train_df['lemmatize_text']=[' '.join(map(str,l)) for l in train_df['lemmatize_word_wo_pos']]
print(train_df.head(7))
print(train_df["text"][8])
print(train_df["combined_postag_wnet"][8])
print(train_df["lemmatize_word_wo_pos"][8])
#print(train_df["lemmatize_word_w_pos"][8])
print(train_df["text"][0],train_df["lemmatize_text"][0])
def cv(data,ngram=1,MAX_NB_WORDS=75000):
    count_vectorizer=CountVectorizer(ngram_range=(ngram,ngram),max_features=MAX_NB_WORDS)
    emb=count_vectorizer.fit_transform(data).toarray()
    print("count vectorize with",str(np.array(emb).shape[1]),"features")
    return emb,count_vectorizer
def print_out(emb,feat,ngram,compared_sentence=0):
    print(ngram,"bag_of_word:")
    #print(feat.get_features_names(),"\n")
    print(ngram,"bag_of_feature:")
    print(test_cv_1gram.vocabulary_,"\n")
    print("bow matrix:")
    print(pd.DataFrame(emb.transpose(),index=feat.get_feature_names()).head(7),"\n")
    print(ngram,"vector example:")
    print(train_df["lemmatize_text"][compared_sentence])
    print(emb[compared_sentence],"\n")
test_corpus=train_df["lemmatize_text"][:5].tolist()
print("the test corpus:",test_corpus,"\n")
test_cv_em_1gram,test_cv_1gram=cv(test_corpus,ngram=1)
print_out(test_cv_em_1gram,test_cv_1gram,ngram="Uni-gram")
test_cv_em_2gram,test_cv_2gram=cv(test_corpus,ngram=2)
print_out(test_cv_em_2gram,test_cv_2gram,ngram="Bi-gram")
test_cv_em_3gram,test_cv_3gram=cv(test_corpus,ngram=3)
print_out(test_cv_em_3gram,test_cv_3gram,ngram="Tri-gram")
train_df_corpus=train_df["lemmatize_text"].tolist()
train_df_em_1gram,vc_1gram=cv(train_df_corpus,1)
train_df_em_2gram,vc_2gram=cv(train_df_corpus,2)
train_df_em_3gram,vc_3gram=cv(train_df_corpus,3)
print(len(train_df_corpus))
print(train_df_em_1gram.shape)
print(train_df_em_2gram.shape)
print(train_df_em_3gram.shape)
del train_df_em_1gram,train_df_em_2gram,train_df_em_3gram
def TFIDF(data,ngram=1,MAX_NB_WORDS=75000):
    tfidf_x=TfidfVectorizer(ngram_range=(ngram,ngram),max_features=MAX_NB_WORDS)
    emb=tfidf_x.fit_transform(data).toarray()
    print("tf-idf with",str(np.array(emb).shape[1]),"features")
    return emb,tfidf_x
test_corpus=train_df["lemmatize_text"][:5].tolist()
print("the test corpus:",test_corpus,"\n")
test_tfidf_em_1gram,test_tfidf_1gram=TFIDF(test_corpus,ngram=1)
print_out(test_tfidf_em_1gram,test_tfidf_1gram,ngram="Uni-gram")
test_tfidf_em_2gram,test_tfidf_2gram=TFIDF(test_corpus,ngram=2)
print_out(test_tfidf_em_2gram,test_tfidf_2gram,ngram="Bi-gram")
test_tfidf_em_3gram,test_tfidf_3gram=TFIDF(test_corpus,ngram=3)
print_out(test_tfidf_em_3gram,test_tfidf_3gram,ngram="Tri-gram")
train_df_corpus=train_df["lemmatize_text"].tolist()
train_df_tfidf_1gram,tfidf_1gram=TFIDF(train_df_corpus,1)
train_df_tfidf_2gram,tfidf_2gram=TFIDF(train_df_corpus,2)
train_df_tfidf_3gram,tfidf_3gram=TFIDF(train_df_corpus,3)
print(len(train_df_corpus))
print(train_df_tfidf_1gram.shape)
print(train_df_tfidf_2gram.shape)
print(train_df_tfidf_3gram.shape)
del train_df_tfidf_1gram,train_df_tfidf_2gram,train_df_tfidf_3gram

x_train,valid_df=train_test_split(train_df)
print(x_train.head(7))
print(valid_df.head(7))
seed=42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
TEXT=data.Field(tokenize='spacy',include_length=True)
LABEL=data.LabelField(dtype=torch.float)
class DataFrameDataset(data.Dataset):
    def _init(self, df, fields, is_test=False, **kwargs): 
        examples = []
        for i, row in df.iterrows(): 
            label = row.target if not is_test else None 
            text = row.lemmatize_text
            examples.append(data.Example.fromlist([text, label], fields)) 
        super().init_(examples, fields, **kwargs) 
    def sort_key(ex): 
        return len(ex.lemmatize_text) 
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None) 
        data_field = fields 
        if x_train is not None: 
            train_data = cls(x_train.copy(), data_field, **kwargs) 
        if val_df is not None: 
            val_data = cls(val_df.copy(), data_field, **kwargs) 
        if test_df is not None: 
            test_data = cls(test_df.copy(), data_field, True, **kwargs) 
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
fields=[('text',TEXT),('label',LABEL)]
train_ds,val_ds=DataFrameDataset.splits(fields,x_train=x_train,val_df=valid_df)
print(vars(train_ds[15]))
print(type(train_ds[15]))
MAX_VOCAB_SIZE=25000
TEXT.build_vocab(train_ds,max_size=MAX_VOCAB_SIZE,vectors='glove.6B.200d',unk_init=torch.Tensor.zero_)
LABEL.build_vocab(train_ds)
BATCH_SIZE=128
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator,valid_iterator=data.BucketIterator.splits((train_ds,val_ds),batch_size=BATCH_SIZE,sort_within_batch=True,device=device)
num_epochs=25
learning_rate=0.001
INPUT_DIM=len(TEXT.vocab)
EMBEDDING_DIM=200
HIDDEN_DIM=256
OUTPUT_DIM=1
N_LAYERS=2
BIDIRECTIONAL=True
DROPOUT=0.2
PAD_IDX=TEXT.vocab.stoi[TEXT.pad_token]
class LSTM_net(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout,pad_idx):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.rnn=nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,bidirectional=bidirectional,dropout=dropout)
        self.fc1=nn.Linear(hidden_dim,1)
        self.dropout=nn.Dropout(dropout)
    def forward(self,text,text_length):
        embedded=self.embedding(text)
        packed_embedded=nn.utils.rnn.pack_padded_sequence(embedded,text_length)
        packed_output,(hidden,cell)=self.rnn(packed_embedded)
        hidden=self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
        output=self.fc1(hidden)
        output=self.dropout(self.fc1(output))
        return output
model=LSTM_net(INPUT_DIM,EMBEDDING_DIM,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT,PAD_IDX)
pretrained_embeddings=TEXT.vocab.vectors 
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX]=torch.zeros(EMBEDDING_DIM)
print(model.embedding.weight.data)
model.to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
def binary_accuracy(preds,y):
    rounded_preds=torch.round(torch.sigmoid(preds))
    correct=(rounded_preds==y).float()
    acc=correct.sum()/len(correct)
    return acc
def train(model,iterator):
    epoch_loss=0
    epoch_acc=0
    model.train()
    for batch in iterator:
        text,text_lengths=batch.lemmatize_text
        optimizer.zero_grad()
        predictions=model(text,text_lengths).squeeze(1)
        loss=criterion(predictions,batch.label)
        acc=binary_accuracy(predictions,batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        epoch_acc+=acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)
def evaluate(model,iterator):
    epoch_acc=0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths=batch.lemmatize_text
            predictions=model(text,text_lengths).squeeze(1)
            acc=binary_accuracy(predictions,batch.label)
            epoch_acc+=acc.item()
    return epoch_acc/len(iterator)
t=time.time()
loss=[]
acc=[]
val_acc=[]
for epoch in range(num_epochs):
    train_loss,train_acc=train(model,train_iterator)
    valid_acc=evaluate(model,valid_iterator)
    print('train loss:{train_loss:.3f}|train acc:{train_acc*100:.2f}%')
    print('val acc:{valid_acc*100:.2f}%')
    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)
print('time:{time.time()-t:.3f}')
plt.xlabel("runs")
plt.ylabel("normalized measure of loss/accuracy")
x_len=list(range(len(acc)))
plt.axis([0,max(x_len),0,1])
plt.title('result of LSTM')
loss=np.asarray(loss)/max(loss)
plt.plot(x_len,loss,'r.',label="loss")
plt.plot(x_len,acc,'b.',label="accuracy")
plt.plot(x_len,val_acc,'g.',label="val_accuracy")
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.2)
plt.show()
#bert transformer model
labels=train_df['target'].values
idx=len(labels)
combined=pd.concat([train_df,test])
combined=combined.text.values
tokenizer=BertTokenizer.from_pretrained('bert-large-uncased',do_lower_case=True)
max_len=0
for text in combined:
    input_ids=tokenizer.encode(text,add_special_tokens=True)
    max_len=max(max_len,len(input_ids))
print('max sentence length:',max_len)
token_lens=[]
for text in combined:
    tokens=tokenizer.encode(text,max_length=512)
    token_lens.append(len(tokens))
fig,axes=plt.subplots(figsize=(14,6))
sns.distplot(token_lens,color='#e74c3c')
plt.show()
train1=combined[:idx]
test1=combined[idx:]
def tokenize_map(sentence,labs='None'):
    global labels
    input_ids=[]
    attention_masks=[]
    for text in sentence:
        encoded_dict=tokenizer.encode_plus(text,add_special_tokens=True,truncation='longest_first',max_length=84,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        input_ids=torch.cat(input_ids,dim=0)
        attention_masks=torch.cat(attention_masks,dim=0)
        if labs!='None':
            labels=torch.tensor(labels)
            return input_ids,attention_masks,labels
        else:
            return input_ids,attention_masks
        
input_ids,attention_masks,labels=tokenize_map(train,labels)
test_input_ids,test_attention_masks=tokenize_map(test)
dataset=TensorDataset(input_ids,attention_masks,labels)
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
batch_size=32
train_dataloader=DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size)
validation_dataloader=DataLoader(val_dataset,sampler=SequentialSampler(val_datset),batch_size=batch_size)
prediction_data=TensorDataset(test_input_ids,test_attention_masks)
prediction_sampler=SequentialSampler(prediction_data)
prediction_dataloader=DataLoader(prediction_data,sampler=prediction_sampler,batch_size=batch_size)
model=BertForSequenceClassification.from_pretrained('bert-large-uncased',num_labels=2,output_attentions=False,output_hidden_states=False)
model.to(device)
params=list(model.named_parameters())
print('the BERT model has differnt named params:'.format(len(params)))
for p in params[0:5]:
    print('{:<55}{:>12}'.format(p[0],str(tuple(p[1].size()))))
print('first transformer')
for p in params[5:21]:
     print('{:<55}{:>12}'.format(p[0],str(tuple(p[1].size()))))
print('output layer')
for p in params[-4:]:
     print('{:<55}{:>12}'.format(p[0],str(tuple(p[1].size()))))
optimizer=AdamW(model.parameters(),lr=6e-6,eps=1e-8)
epochs=3
total_steps=len(train_dataloader)*epochs
scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
def flat_accuracy(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return accuracy_score(labels_flat,pred_flat)
def flat_f1(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return f1_score(labels_flat,pred_flat)
def format_time(elapsed):
    elapsed_rounded=int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
training_stats=[]
total_t0=time.time()
for epoch_i in range(0,epochs):
    print('epoch'.format(epoch_i+1,epochs))
    t0=time.time()
    total_train_loss=0
    model.train()
    for step,batch in enumerate(train_dataloader):
        if step%50==0 and not step==0:
            elapsed=format_time(time.time()-10)
            print('batch{:>5,} of {:>5,}.elapsed:'.format(step,len(train_dataloader),elapsed))
            b_input_ids=batch[0].to(device).to(torch.int64)
            b_input_mask=batch[1].to(device).to(torch.int64)
            b_labels=batch[2].to(device).to(torch.int64)
            model.zero_grad()
            loss,logits=model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
            total_train_loss+=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            avg_train_loss=total_train_loss/len(train_dataloader)
            training_time=format_time(time.time()-t0)
            print('average training loss:'.format(avg_train_loss))
            print('training epoch took:'.format(training_time))
            t0=time.time()
            model.eval()
            total_eval_accuracy=0
            total_eval_loss=0
            total_eval_f1=0
            nb_eval_steps=0
            for batch in validation_dataloader:
                b_input_ids=batch[0].to(device)
                b_input_mask=batch[1].to(device)
                b_labels=batch[2].to(device)
                with torch.no_grad():
                    (loss,logits)=model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
                    total_eval_loss+=loss.item()
                    logits=logits.detach().cpu().numpy()
                    label_ids=b_labels.to('cpu').numpy()
                    total_eval_accuracy+=flat_accuracy(logits,label_ids)
                    total_eval_f1+=flat_f1(logits,label_ids)
                    avg_val_accuracy=total_eval_accuracy/len(validation_dataloader)
                    print('accuracy:'.format(avg_val_accuracy))
                    avg_val_f1=total_eval_f1/len(validation_dataloader)
                    print('f1:'.format(avg_val_f1))
                    avg_val_loss=total_eval_loss/len(validation_dataloader)
                    print('validation loss:'.format(avg_val_loss))
                    training_stats.append({'epoch':epoch_i+1,'training loss':avg_train_loss,'valid loss':avg_val_loss})
pd.set_option('precision',2)
df_stats=pd.DataFrame(data=training_stats)
df_stats=df_stats.set_index('epoch')
fig,axes=plt.subplots(figsize=(12,8))
plt.plot(df_stats['training loss'],'b-o',label='training')
plt.plot(df_stats['valid loss'],'g-o',label='validation')   
plt.title('training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.xticks([1,2,3])
plt.show()
model.eval()
predictions=[]
for batch in prediction_dataloader:
    batch=tuple(t.to(device) for t in batch)
    b_input_ids,b_input_mask=batch
    with torch.no_grad():
        outputs=model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
        logits=outputs[0]
        logits=logits.detach().cpu().numpy()
        predictions.append(logits)
flat_predictions=[item for sublist in predictions for item in sublist]
flat_predictions=np.argmax(flat_predictions,axis=1).flatten()
submission=pd.read_csv(r'/home/pegah/Desktop/sample_submission.csv')    
submission['target']=flat_predictions
submission.head(10)
submission.to_csv('submission.csv',index=False,header=True)             
                    
                


            
   

    
    
    