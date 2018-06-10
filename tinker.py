#!pip install --upgrade gensim
#!pip install pyldavis

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
import pyLDAvis
import pyLDAvis.gensim

sent = LineSentence('articles.txt')

# learn the dictionary 
article_dict = Dictionary(sent)
    
# filter tokens that are very rare or too common from
# the dictionary (filter_extremes) and reassign integer ids (compactify)
article_dict.filter_extremes(no_below=5, no_above=0.2)
article_dict.compactify()

article_dict.save('articles.dict')
    
# load the finished dictionary from disk
article_dict = Dictionary.load('articles.dict')

def bow(filepath,d): # output bag of words representation
    for review in LineSentence(filepath):
        yield d.doc2bow(review)

# generate bag-of-words representations for all reviews and save them as a matrix
MmCorpus.serialize('articles.mm',
                       bow('articles.txt',article_dict))
    
# load the finished bag-of-words corpus from disk
corpus = MmCorpus('articles.mm')

# Create LDA model
lda = LdaMulticore(corpus,num_topics=10,
                   id2word=article_dict, 
                   workers=2)
    
lda.save('./lda_model')
lda = LdaMulticore.load('./lda_model')
