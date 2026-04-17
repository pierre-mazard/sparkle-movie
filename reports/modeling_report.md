# Recommender Modeling Report

## Experimental protocol

- Data split: temporal split per user (80% train, 20% test with at least one item in test when possible).
- Relevance threshold for ranking metrics: rating >= 4.0 in test set.
- Ranking cutoff: Top-10.
- Evaluated users with at least one relevant item in test: 592.

## ALS (Spark MLlib)

- Best rank: 40
- Best regParam: 0.1
- Best maxIter: 15
- Validation RMSE: 0.8928
- Test RMSE: 0.9227

## Hybrid strategy

- Fusion method: Weighted Reciprocal Rank Fusion
- Weights: ALS=0.1, Content=0.1, UserKNN=0.8

## Hybrid tuning

- Search space: all weight triplets on a 0.1 grid with ALS + Content + UserKNN = 1
- Selection rule: maximize Precision@K, then Recall@K, then Coverage@K

Top 5 weight settings:

| w_als | w_content | w_knn | precision_at_k | recall_at_k | coverage_at_k |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.1 | 0.8 | 0.0149 | 0.0217 | 0.1532 |
| 0.1 | 0.2 | 0.7 | 0.0149 | 0.0217 | 0.1532 |
| 0.1 | 0.3 | 0.6 | 0.0149 | 0.0217 | 0.1532 |
| 0.2 | 0.1 | 0.7 | 0.0149 | 0.0217 | 0.1532 |
| 0.2 | 0.2 | 0.6 | 0.0149 | 0.0217 | 0.1532 |

## Comparative metrics

| method | precision_at_k | recall_at_k | coverage_at_k | rmse |
| --- | --- | --- | --- | --- |
| ALS | 0.0078 | 0.0078 | 0.1293 | 0.9227 |
| Content-TFIDF | 0.0044 | 0.0062 | 0.1691 | nan |
| UserKNN | 0.0149 | 0.0217 | 0.1532 | nan |
| Hybrid-Weighted | 0.0149 | 0.0217 | 0.1532 | nan |

## Recommendations for fictive users (mapped to real users)

| user_label | real_userId | method | rank | title |
| --- | --- | --- | --- | --- |
| FICTIF_U1 | 414 | ALS | 1 | Belle époque (1992) |
| FICTIF_U1 | 414 | ALS | 2 | Cherish (2002) |
| FICTIF_U1 | 414 | ALS | 3 | Rain (2001) |
| FICTIF_U2 | 599 | ALS | 1 | Mulholland Dr. (1999) |
| FICTIF_U2 | 599 | ALS | 2 | Neon Genesis Evangelion: Death & Rebirth (Shin seiki Evangelion Gekijô-ban: Shito shinsei) (1997) |
| FICTIF_U2 | 599 | ALS | 3 | On the Beach (1959) |
| FICTIF_U3 | 474 | ALS | 1 | Lady Jane (1986) |
| FICTIF_U3 | 474 | ALS | 2 | Crossing Delancey (1988) |
| FICTIF_U3 | 474 | ALS | 3 | Impostors, The (1998) |
| FICTIF_U4 | 448 | ALS | 1 | The Big Bus (1976) |
| FICTIF_U4 | 448 | ALS | 2 | Victory (a.k.a. Escape to Victory) (1981) |
| FICTIF_U4 | 448 | ALS | 3 | Day at the Races, A (1937) |
| FICTIF_U5 | 274 | ALS | 1 | Watermark (2014) |
| FICTIF_U5 | 274 | ALS | 2 | Connections (1978) |
| FICTIF_U5 | 274 | ALS | 3 | Zeitgeist: Moving Forward (2011) |
| FICTIF_U1 | 414 | Content-TFIDF | 1 | Stunt Man, The (1980) |
| FICTIF_U1 | 414 | Content-TFIDF | 2 | Just Before I Go (2014) |
| FICTIF_U1 | 414 | Content-TFIDF | 3 | Adam's Apples (Adams æbler) (2005) |
| FICTIF_U2 | 599 | Content-TFIDF | 1 | Stunt Man, The (1980) |
| FICTIF_U2 | 599 | Content-TFIDF | 2 | Last Boy Scout, The (1991) |
| FICTIF_U2 | 599 | Content-TFIDF | 3 | The Great Train Robbery (1978) |
| FICTIF_U3 | 474 | Content-TFIDF | 1 | Waiting to Exhale (1995) |
| FICTIF_U3 | 474 | Content-TFIDF | 2 | Postman, The (Postino, Il) (1994) |
| FICTIF_U3 | 474 | Content-TFIDF | 3 | Other People's Money (1991) |
| FICTIF_U4 | 448 | Content-TFIDF | 1 | Hunting Party, The (2007) |
| FICTIF_U4 | 448 | Content-TFIDF | 2 | Stunt Man, The (1980) |
| FICTIF_U4 | 448 | Content-TFIDF | 3 | Wasabi (2001) |
| FICTIF_U5 | 274 | Content-TFIDF | 1 | Pusher III: I'm the Angel of Death (2005) |
| FICTIF_U5 | 274 | Content-TFIDF | 2 | Hunting Party, The (2007) |
| FICTIF_U5 | 274 | Content-TFIDF | 3 | Another 48 Hrs. (1990) |
| FICTIF_U1 | 414 | UserKNN | 1 | Enchanted April (1992) |
| FICTIF_U1 | 414 | UserKNN | 2 | Guys and Dolls (1955) |
| FICTIF_U1 | 414 | UserKNN | 3 | Truly, Madly, Deeply (1991) |
| FICTIF_U2 | 599 | UserKNN | 1 | Stranger Than Paradise (1984) |
| FICTIF_U2 | 599 | UserKNN | 2 | Barcelona (1994) |
| FICTIF_U2 | 599 | UserKNN | 3 | Cyrano de Bergerac (1990) |
| FICTIF_U3 | 474 | UserKNN | 1 | Harry Potter and the Order of the Phoenix (2007) |
| FICTIF_U3 | 474 | UserKNN | 2 | Harry Potter and the Deathly Hallows: Part 2 (2011) |
| FICTIF_U3 | 474 | UserKNN | 3 | Once Were Warriors (1994) |
| FICTIF_U4 | 448 | UserKNN | 1 | Ran (1985) |
| FICTIF_U4 | 448 | UserKNN | 2 | Creature Comforts (1989) |
| FICTIF_U4 | 448 | UserKNN | 3 | His Girl Friday (1940) |
| FICTIF_U5 | 274 | UserKNN | 1 | My Neighbor Totoro (Tonari no Totoro) (1988) |
| FICTIF_U5 | 274 | UserKNN | 2 | Creature Comforts (1989) |
| FICTIF_U5 | 274 | UserKNN | 3 | Ran (1985) |
| FICTIF_U1 | 414 | Hybrid-Weighted | 1 | Enchanted April (1992) |
| FICTIF_U1 | 414 | Hybrid-Weighted | 2 | Guys and Dolls (1955) |
| FICTIF_U1 | 414 | Hybrid-Weighted | 3 | Truly, Madly, Deeply (1991) |
| FICTIF_U2 | 599 | Hybrid-Weighted | 1 | Stranger Than Paradise (1984) |
| FICTIF_U2 | 599 | Hybrid-Weighted | 2 | Barcelona (1994) |
| FICTIF_U2 | 599 | Hybrid-Weighted | 3 | Cyrano de Bergerac (1990) |
| FICTIF_U3 | 474 | Hybrid-Weighted | 1 | Harry Potter and the Order of the Phoenix (2007) |
| FICTIF_U3 | 474 | Hybrid-Weighted | 2 | Harry Potter and the Deathly Hallows: Part 2 (2011) |
| FICTIF_U3 | 474 | Hybrid-Weighted | 3 | Once Were Warriors (1994) |
| FICTIF_U4 | 448 | Hybrid-Weighted | 1 | Ran (1985) |
| FICTIF_U4 | 448 | Hybrid-Weighted | 2 | Creature Comforts (1989) |
| FICTIF_U4 | 448 | Hybrid-Weighted | 3 | His Girl Friday (1940) |
| FICTIF_U5 | 274 | Hybrid-Weighted | 1 | My Neighbor Totoro (Tonari no Totoro) (1988) |
| FICTIF_U5 | 274 | Hybrid-Weighted | 2 | Creature Comforts (1989) |
| FICTIF_U5 | 274 | Hybrid-Weighted | 3 | Ran (1985) |

## Interpretation

- ALS is optimized for rating prediction (RMSE), typically stronger on collaborative signals.
- Content-TFIDF exploits genre similarity and is robust for explainability and cold-start items.
- UserKNN captures neighborhood effects and can yield interpretable peer-based recommendations.
- Hybrid-Weighted combines signal diversity and usually improves robustness across user profiles.
- Final production choice should balance precision and catalog coverage depending on product goals.
