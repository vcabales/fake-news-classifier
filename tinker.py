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
MmCorpus.serialize('articles.mm', bow('articles.txt',article_dict))

# load the finished bag-of-words corpus from disk
corpus = MmCorpus('articles.mm')

# Create LDA model
lda = LdaMulticore(corpus,num_topics=10, id2word=article_dict, workers=2)
lda.save('./lda_model')
lda = LdaMulticore.load('./lda_model')

# It's really slow when they're all together for some reason

# For real and fake dataframes
# fake_sent = LineSentence('fake.txt')
# fake_dict = Dictionary(fake_sent)
# fake_dict.filter_extremes(no_below=5, no_above=0.2)
# fake_dict.compactify()
# fake_dict.save('fake.dict')
# fake_dict = Dictionary.load('fake.dict')
#
# real_sent = LineSentence('real.txt')
# real_dict = Dictionary(real_sent)
# real_dict.filter_extremes(no_below=5, no_above=0.2)
# real_dict.compactify()
# real_dict.save('real.dict')
# real_dict = Dictionary.load('real.dict')
#
# MmCorpus.serialize('fake.mm', bow('fake.txt',fake_dict))
# fake_corpus = MmCorpus('fake.mm')
# MmCorpus.serialize('real.mm', bow('real.txt',real_dict))
# real_corpus = MmCorpus('real.mm')
#
# fake_lda = LdaMulticore(fake_corpus,num_topics=10,id2word=fake_dict,workers=2)
# fake_lda.save('./fake_lda_model')
#
# real_lda = LdaMulticore(real_corpus,num_topics=10,id2word=real_dict,workers=2)
# real_lda.save('./real_lda_model')
