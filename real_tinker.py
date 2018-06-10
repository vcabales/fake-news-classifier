from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
import pyLDAvis
import pyLDAvis.gensim

def bow(filepath,d): # output bag of words representation
    for review in LineSentence(filepath):
        yield d.doc2bow(review)

real_sent = LineSentence('real.txt')
real_dict = Dictionary(real_sent)
real_dict.filter_extremes(no_below=5, no_above=0.2)
real_dict.compactify()
real_dict.save('real.dict')
real_dict = Dictionary.load('real.dict')

MmCorpus.serialize('real.mm', bow('real.txt',real_dict))
real_corpus = MmCorpus('real.mm')

real_lda = LdaMulticore(real_corpus,num_topics=10,id2word=real_dict,workers=2)
real_lda.save('./real_lda_model')
