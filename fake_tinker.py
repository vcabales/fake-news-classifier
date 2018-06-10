from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
import pyLDAvis
import pyLDAvis.gensim

def bow(filepath,d): # output bag of words representation
    for review in LineSentence(filepath):
        yield d.doc2bow(review)

fake_sent = LineSentence('fake.txt')
fake_dict = Dictionary(fake_sent)
fake_dict.filter_extremes(no_below=5, no_above=0.2)
fake_dict.compactify()
fake_dict.save('fake.dict')
fake_dict = Dictionary.load('fake.dict')

MmCorpus.serialize('fake.mm', bow('fake.txt',fake_dict))
fake_corpus = MmCorpus('fake.mm')

fake_lda = LdaMulticore(fake_corpus,num_topics=10,id2word=fake_dict,workers=2)
fake_lda.save('./fake_lda_model')
