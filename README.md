# Generalized Influence Measurement

This repository accompanies the paper and contains a minimal version of the data and code for reproducibility, including; the tweet ids for the arson and covid datasets, the derived empirical influence ranking in the arson dataset, the implementation of the MTURK active learning setup, and functions for computing influence from cascades.


### Structure
In `./01_raw/` we provide the user ids `./01_raw/arson_bushfires/arson_user_ids.txt` and tweet ids `./01_raw/arson_bushfires/arson_tweet_ids.txt` for the arson dataset; as well as the user ids `./01_raw/covid_19/user_ids/` and tweet ids `./01_raw/covid_19/tweet_ids/` for the covid dataset. Additionally, we provide the empirical social influence ranking in `./01_raw/arson_bushfires/arson_with_empirical_influence.csv`.

In the `./02_mturk/` folder we provide the implementation code for the active learning framework.

In the `./03_influence/` folder we provide the core code for computing influence.
