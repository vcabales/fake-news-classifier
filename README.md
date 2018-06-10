# Fake News Analysis

A fake news classifier built with sckit-learn and a bit of NLTK. For a brief overview, check out the [project website](https://ml-fakenews.herokuapp.com/) or [our Jupyter notebook](https://github.com/vcabales/fake-news-classifier/blob/master/Fake%20News%20Analysis.ipynb).

## Dependencies
```
python==2.7.14
Flask==1.0.2
gunicorn
numpy==1.10.4
pandas==0.17.1
scikit-learn==0.17
pyldavis==2.1.2
nltk==3.3
scipy==0.13.3
gensim
```
To run the Flask app locally, run `flask run`.

## LDA Visualizations
A quick note on generating the LDA visualizations locally. You may have difficulty running
`LdaMulticore(corpus,num_topics=10,id2word=article_dict,workers=2)` in Jupyter. As an alternative, you can run the following commands in your terminal to generate the LDA models for the general corpus, fake articles, and real articles respectively:
```
python tinker.py
python fake_tinker.py
python real_tinker.py
```
