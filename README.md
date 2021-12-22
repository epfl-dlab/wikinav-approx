# wikinav-approx
Source code for "Approximating Navigation of Readers in Wikipedia Through Public Clickstream Data"

## Resources

All the publicly available *resources* can be downloaded via [google drive](https://drive.google.com/drive/folders/135p_Ufo18Z19TvHh9oPGRE0xCs0X2Od3?usp=sharing) (no sign-in required).
**Important Note: It is not currently possible for us to share the navigation sequences (both 'real' and 'synthetic') used in this work with the general public. Consequently, any resources derived from navigation sequences, such as 'navigation embeddings', 'models for next-article prediction', etc. have not been shared publicly. This is to protect the privacy of Wikipedia users and is consistent with our agreement with the Wikimedia Foundation. If, at a later point in time, the Wikimedia Foundation decides to share this data, an updated link to that resource would be provided in this GitHub repository.**

1. To obtain the Wikipedia *clickstream* dataset, run the `crawl_clickstream.sh` script present within the `data/clickstream` directory. This will fetch the compressed clickstream data corresponding to all the 8 language versions used in this study.
2. To obtain the Wikipedia *wikitext (XML)* dumps, run the `download_dumps.ipynb` iPython notebook within the `data/xml_dumps` directory. This will fetch the compressed XML dumps corresponding to all the 8 language versions used in this study.
3. To obtain pretrained `FastText` word embeddings, run the `crawl_fasttext_embeddings.sh` script present within the `data/pretrained_embeddings` directory. This will fetch the 300-dimensional *aligned* word embeddings for all the 8 languages used in this study.
4. Unzip the *graphs.zip* file from [google drive](https://drive.google.com/drive/folders/135p_Ufo18Z19TvHh9oPGRE0xCs0X2Od3?usp=sharing), which contains the Wikipedia hyperlink graph for the 8 language versions used in this study, into the empty `data/graphs` directory provided with the code repository.
5. The *relatedness* and *similarity* dataset (based on the *WikiSRS* benchmark) is present within the `data/relatedness` directory.
6. Unzip the *topic_prediction_data.zip* file from [google drive](https://drive.google.com/drive/folders/135p_Ufo18Z19TvHh9oPGRE0xCs0X2Od3?usp=sharing), which contains the topic-labels dataset for the 8 language versions used in this study, into the empty `data/topic_prediction` directory provided with the code repository.

## Code
1. To obtain the semantic embeddings of Wikipedia articles, run the `wiki-article-description_embeddings.ipynb` iPython notebook within the `code` directory. This will compute the Wikipedia article representations based on the text in the first paragraph of an article, for all the 8 language versions used in this study.
2. The code corresponding to different analyses and downstream tasks performed in this study is also present within the `code` directory.