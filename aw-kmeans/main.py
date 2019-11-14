import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

# data = pd.read_csv('data/15ke_clean.csv')
data = pd.read_csv('data/data_anonymised.csv')
stop_data = pd.read_csv('data/additional_stop_words.csv')

stop_words = get_stop_words('fr')
stop_words.extend(stop_data['stop_words'])
token_pattern = '[A-Za-zÀ-ÖØ-öø-ÿ]{2,}'

vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=token_pattern)
x = vectorizer.fit_transform(data['text'])

for true_k in range(3, 10):
    print('### %d Clusters\n' % true_k)
    model = KMeans(n_clusters=true_k, init='k-means++', random_state=42)
    model.fit(x)

    print('elbow score: %d\n' % round(model.score(x)))

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(true_k):
        print("- Cluster %d:" % i),
        for j in order_centroids[i, :6]:
            print('\t%s' % terms[j])

    if true_k == 5:
        res = model.predict(x)
        data['class'] = res
        
        for i in range(5):
            data[data['class'] == i].to_csv('results/aw_anon_%d.csv' % i, index=False)

if __name__ == "__main__":
    pass