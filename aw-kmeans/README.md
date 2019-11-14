# oraw4 domain type

Kmeans Unsupervised text classification. With k=5.

## build & run

```bash
docker build . -t awkmeans
docker run -d --rm -p 8080:8080 awkmeans
```

## api

Swagger docs [here](swagger.yaml)

### GET /train

parameters:

- `dataset` url to a csv file with a 'text' column
- `stop_words` (optional) url to a csv file with a 'stop_words' column
- `lang` (optional, default to ('fr')

return:

- `integer` a model id

### GET /predict

parameters:

- `model_id` model id returned by /train
- `text` text to classify

return:

- `integer` class id from 0 to 4 included.

## results

Results summary on the [15k french issue dataset](data/15ke_clean.csv) can be found [here](RESULT.md).
