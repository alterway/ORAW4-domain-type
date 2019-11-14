# Issues

## Topics detection fails to display on the manual dataset

The process succeed and generate about 224 topics but the front wont display them

## keyword extraction (wip)

- The system's preprocessor is super slow (~24h for 15k issues on an i7)
- The preprocessing handle poorly accentuated characters and maybe spaces
- It load the Word2Vec models using cPickle instead of the Word2Vec.load (resolved)

## Magic dead url (resolved)

- [core/Annotation_Library.py:30](core/Annotation_Library.py#30)
- Fetch a dead [url](http://193.109.207.65:15024/nlp/keywords/supervised)
- SOM & Codebook clustering relies on Word2Vec which rely on this

The magic url was corresponded to a WikipediaDB instance and which returning english wikipedia articles ?!

## Word2Vec crazy training set fetching (bypassed)

- Entities are identified using WikipediaDB english (?!) article matcher. All corresponding Wikipedia articles are downloaded (!!) and a new Word2Vec model is train on those articles

To bypass the crazy workload it represent both for us and WikipediaDB and Wikipedia, we will use
a pre-trained Word2Vec using the same method but with a dump of all Wikipedia french articles

## Bigram model training epochs (hotfixed)

- The bigram model was set to be trained on 5000 epochs which is impossible with the quite biger Word2Vec model we offer

We reduced the number of epochs to 50 since we realised the gain in loss was neglictable after 50 epochs

## Bigram wants to load it_core_news_sm (hotfixed)

- [core/genspacyrank.py:224](core/genspacyrank.py#224)
- 'it_core_news_sm' is hard coded in [core/micro_services/text_ranking_ms.py:29](core/micro_services/text_ranking_ms.py#29)
