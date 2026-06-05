# Spotify Track Popularity Prediction
#### **Using audio features, genre and metadata to classify popular songs**
**Authors: Anh Tran, Amelia Oo**

## Introduction

Music popularity is difficult to explain with one simple factor. A song may become popular because of the artist, the genre, playlist exposure, social trends, or the way the song sounds. In this project, we focus on the measurable side of music: the audio and metadata features available for Spotify tracks. Our goal is to explore whether these features can help us understand and predict which tracks are more likely to be popular.

We use the Spotify Music Tracks dataset, which contains 114,000 rows and 22 columns. Each row represents a single Spotify track, and the columns contain metadata and audio features associated with that track, such as the artist name, album name, release date, duration, explicit status, genre, and Spotify popularity score. The dataset also includes audio features such as danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, liveness, and tempo. These variables allow us to compare songs not only by genre or popularity, but also by musical characteristics.

Our main data science question is: **Can we predict whether a Spotify track becomes popular using its audio features, genre and track metadata?** <br> <br>
This question is relevant because music platforms, artists, and playlist curators often rely on data to understand listener behavior and identify patterns behind successful songs. While popularity is influenced by many outside factors that are not captured in this dataset, the available audio features still give us a way to study whether certain types of songs are more likely to perform well.

The dataset contains many variables; however, our analysis focuses on the following columns:

| Column | Description |
|----------|-------------|
| `track_id` | Spotify’s unique identifier for the track. |
| `popularity` | Spotify popularity score ranging from 0 to 100. |
| `track_genre` | Genre of the track as labeled by Spotify. |
| `danceability` | Measures how suitable a track is for dancing based on tempo, rhythm stability, and beat strength (0–1). |
| `energy` | Measures the perceptual intensity and activity level of a track (0–1). |
| `loudness` | Measures the overall loudness of a track in decibels (dB). |
| `speechiness` | Measures the presence of spoken words within a track. |
| `acousticness` | Estimates the confidence that the track is acoustic rather than electronically produced (0–1). |
| `instrumentalness` | Estimates the probability that the track contains little to no vocals (0–1). |
| `liveness` | Estimates the probability that the track was recorded with a live audience (0–1). |
| `valence` | Measures the musical positivity of a track, where higher values indicate happier or more cheerful sounds (0–1). |
| `tempo` | Estimated tempo of the track in beats per minute (BPM). |
| `explicit` | Indicates whether the track contains explicit content (True = explicit, False = not explicit).|
| `artists` | Name(s) of the artist(s). |
| `duration_ms` | Duration of the track in milliseconds. |
| `release_date` | Release date of the track. |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning 

To focus on our research question, we kept only the relevant columns described above and removed the remaining columns from the dataset.

1. We start by examining the dataset for missing values. We found that the `tempo` column contained over 22,000 missing values, which could affect analyses involving tempo. Since tempo is an important audio feature, we chose to preserve these missing values for later missingness analysis rather than immediately removing the affected observations.
2. Next, we filtered the `track_genre` column to include only six genres: Classical, Hip-Hop, Country, Electronic, Metal, and Pop. The original dataset contains many genre categories, which can make analysis and visualization difficult to interpret. By focusing on a smaller set of representative genres, we were able to perform clearer genre-based comparisons while keeping the scope of the project manageable.
3. Then, we converted the `release_date` column to a datetime format and extracted the release year into a new column, `release_year`, which allows us to analyze how a track's release period may relate to its popularity.
4. We also created a new column by converting `duration_ms` into a more interpretable feature, `duration_min`, which represents track duration in minutes. 
5. Finally, after examining the distribution of Spotify popularity scores, we created a binary target variable, `is_popular`, where tracks with a popularity score of 50 or greater were labeled as popular and tracks below 50 were labeled as not popular. We chose a threshold of 50 because it approximately corresponds to the top 25% of tracks in our dataset. 

Below is the head of our cleaned DataFrame. 

| track_id               | track_genre   |   popularity |   danceability |   energy |   acousticness | explicit   |   release_year | is_popular   |
|:-----------------------|:--------------|-------------:|---------------:|---------:|---------------:|:-----------|---------------:|:-------------|
| 5SuOikwiRyPMVoIQDJUgSV | acoustic      |           73 |          0.676 |   0.461  |         0.0322 | False      |           1974 | True         |
| 4qPNDBW1i3p13qLCt0Ki3A | acoustic      |           55 |          0.42  |   0.166  |         0.924  | False      |           1995 | True         |
| 1iJBSr7s7jYXzM8EGcbK5b | acoustic      |           57 |          0.438 |   0.359  |         0.21   | False      |           1973 | True         |
| 6lfxq3CG4xtTiEg7opyCyx | acoustic      |           71 |          0.266 |   0.0596 |         0.905  | False      |           2018 | True         |
| 5vjLSffimiIP26QG5WcN2K | acoustic      |           82 |          0.618 |   0.443  |         0.469  | False      |           2017 | True         |

Note: For readability, only a subset of the columns most relevant to our analysis is shown.

### Univariate Analysis

First, we explored the distribution of Spotify popularity scores. 

<iframe
  src="assets/popularity_distribution.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
The histogram above shows a right-skewed distribution, with most tracks receiving relatively low to moderate popularity scores and fewer tracks achieving very high popularity. This suggests that highly popular songs are less common in the dataset, which may make popularity prediction more challenging since there are fewer examples of highly popular tracks for the model to learn from.


We then examined the distribution of danceability scores. 

<iframe
  src="assets/danceability_distribution.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
We observed that most songs have moderate-to-high danceability scores, indicating that Spotify tracks in this dataset tend to have rhythmically engaging music. 



Lastly, we examined the distribution of explicit and non-explicit tracks.

<iframe
  src="assets/explicit_distribution.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
This chart reveals a significant difference between explicit and non-explicit tracks in the dataset. It appears that Non-explicit songs make up the vast majority of tracks, with over 100,000 entries, while explicit songs account for only a much smaller portion of the dataset. This suggests that mainstream music in the dataset is still largely dominated by non-explicit content. This imbalance may also affect later analysis or predictive modeling, where patterns associated with non-explicit tracks could have a stronger influence on the model due to their much larger representation in the dataset.

### Bivariate Analysis

To better understand factors associated with popularity, we examined the relationships between popularity and two track characteristics: danceability and explicit status.

First, we examined the relationship between danceability and popularity. 

<iframe
  src="assets/danceability_popularity.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
The relationship between danceability and popularity does not appear strongly linear. However, highly popular songs are more frequently observed among tracks with moderate-to-high danceability scores, suggesting that danceability may contribute to a track's success, although it is unlikely to be the sole factor influencing popularity.

Next, we compared popularity distributions between explicit and non-explicit tracks.

<iframe
  src="assets/explicit_popularity.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
Interestingly, although non-explicit tracks make up the majority of the dataset, explicit tracks appear to have a slightly higher median popularity. This may suggest that explicit content is relatively common among more popular or commercially successful tracks. However, both groups still show a wide spread in popularity, indicating that explicitness alone is not enough to determine whether a song becomes popular.

### Interesting Aggregates 

We also look at how audio features differ between popular and non-popular songs.

| is_popular   |   danceability |   energy |   loudness |   speechiness |   acousticness |   instrumentalness |   liveness |   valence |   tempo |
|:-------------|---------------:|---------:|-----------:|--------------:|---------------:|-------------------:|-----------:|----------:|--------:|
| False        |           0.56 |     0.64 |      -8.36 |          0.09 |           0.32 |               0.17 |       0.22 |      0.48 |  123.43 |
| True         |           0.59 |     0.64 |      -7.82 |          0.08 |           0.29 |               0.1  |       0.18 |      0.47 |  121.79 |

The table above compares the average audio features between popular and non-popular songs. Popular songs tend to have slightly higher average danceability scores and higher loudness values, suggesting that more rhythmically engaging and louder tracks may perform better on Spotify. Popular songs also show lower average acousticness and instrumentalness, indicating that mainstream songs in the dataset are generally less acoustic and more vocal-focused.

However, many of the differences between the two groups remain relatively small, especially for features such as energy, speechiness, and valence. This suggests that no single audio feature alone strongly determines popularity, and that song popularity is likely influenced by a combination of multiple audio characteristics and genre.


## Assessment of Missingness

### NMAR Analysis

We believe the missing values in the `tempo` column are unlikely to be Not Missing At Random (NMAR). Instead, the missingness may be related to limitations in Spotify's audio analysis process or metadata collection rather than the tempo value itself.

If additional information were available about how Spotify computes audio features, whether a track failed audio processing, or why certain metadata could not be extracted, this information could help explain why tempo values are missing. Since the missingness is likely associated with factors other than the `tempo` itself, we believe the missingness mechanism is more consistent with Missing At Random (MAR) than NMAR.

### Missingness Dependency

We next examined the missingness of the `tempo` column by performing permutation tests to determine whether its missingness depends on other observed variables in the dataset. Specifically, we investigated whether the missingness of tempo depends on `track_genre` or `release_year`.

To conduct these tests, we created a Boolean indicator column, `tempo_missing`, which records whether the `tempo` value is missing for each track. 

#### Test 1: Track Genre and Tempo

**Null Hypothesis**: The missingness of `tempo` does not depend on `track_genre`.  
**Alternative Hypothesis**: The missingness of `tempo` depends on `track_genre`. 

**Test Statistic**: Total Variation Distance (TVD)  between the distribution of genres for tracks with missing tempo values and the distribution of genres for tracks with non-missing tempo values.

**Significance level**: 0.05

<iframe
  src="assets/permutation_test_1.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
After performing the permutation test, we obtained a p-value of 0.0. Since the p-value is less than 0.05, we reject the null hypothesis. This suggests that the missingness of `tempo` does depend on `track_genre`.

#### Test 2: Release Year and Tempo

**Null Hypothesis**: The missingness of `tempo` does not depend on a track's `release_year`. <br>
**Alternative Hypothesis**: The missingness of `tempo` depends on a track's `release_year`. 

**Test Statistic**: Absolute Difference in the mean `release_year` between tracks with missing `tempo` values and tracks with non-missing `tempo` values. 

**Significance level**: 0.05

<iframe
  src="assets/permutation_test_2.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>
After performing the permutation test, we obtained a p-value of 0.0. Since the p-value is less than 0.05, we reject the null hypothesis. This suggests that the missingness of `tempo` does depend on `release_year`.

To further validate our findings, we also explored the relationship between tempo missingness and several other observed variables in the dataset. Across these additional permutation tests, the observed test statistics were consistently much larger than those generated under the null distributions. These results align with the two tests presented above and provide further support for our conclusion that the missingness of tempo is more consistent with a MAR mechanism.

## Hypothesis Testing 

To investigate whether genre is associated with song popularity, we conducted a hypothesis test comparing the genre distributions of popular and non-popular songs.

**Null Hypothesis**: The distribution of `track_genre` is the same for popular and non-popular songs. <br>
**Alternative Hypothesis**: The distribution of `track_genre` differs between popular and non-popular songs.

**Test Statistic**: Total Variation Distance (TVD) between the genre distributions of popular and non-popular songs.

**Significance level**: 0.05

**P-value**: 0.0

<iframe
  src="assets/hypothesis_test.html"
  width="900"
  height="400"
  frameborder="0">
</iframe>

The observed TVD (0.479) is substantially larger than the values generated under the null distribution, resulting in a p-value that is effectively 0. Since the p-value is well below our significance level of 0.05, therefore we rejeuct the null hypothesis. This suggests that the distribution of genres differs between popular and non-popular songs and that genre may be an important factor associated with song popularity.

## Framing a Prediction Problem

The prediction problem is: **Can we predict whether a Spotify track is popular using its audio features, genre, and track metadata?** <br> 
This is a binary classification problem. The response variable is `is_popular`, created by thresholding the popularity score at 50, tracks scoring 50 or above are labeled popular (1), and the rest not popular (0). This threshold labels the top 25% of tracks as popular, giving a reasonable class balance.

At the time of prediction, all audio features, genre, and metadata (duration, explicit status, release year, number of artists) are known properties of the track itself, not outcomes that happen after release.

We chose **F1 score** as our evaluation metric instead of accuracy because the dataset is imbalanced, which only about 25% of tracks are popular. A model could achieve high accuracy by simply predicting "not popular" for every track. F1 balances precision and recall, giving a more honest measure of how well the model identifies popular tracks.

## Baseline Model

For the baseline model, we used a `RandomForestClassifier` inside a single `sklearn Pipeline`. The goal of this baseline was to establish an initial performance benchmark using a small set of features before adding more complexity in the final model.

### Features Used

The baseline model used two features:

| Feature | Type | Encoding / Processing |
|---|---|---|
| `danceability` | Quantitative | Scaled using `StandardScaler` |
| `track_genre` | Nominal | Encoded using `OneHotEncoder` |

The model was trained on 80% of the data and evaluated on the remaining 20% test set.

### Baseline Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.739 |
| Precision | 0.686 |
| Recall | 0.582 |
| F1-score | 0.630 |


The baseline model provides a reasonable starting point, but there is still room for improvement. Our primary metric, **F1 score (0.630)**, reflects this; in particular, the model only identifies about 58% of popular tracks (recall), meaning it misses nearly half of the tracks it should flag as popular. This  suggests that using only `danceability` and `track_genre` is not enough to fully capture the patterns behind track popularity.

For the final model, we plan to improve performance by adding more Spotify audio features, such as `energy`, `valence`, `acousticness`, `loudness`, `tempo`, and other track-level audio characteristics. We also plan to tune model hyperparameters to better capture more complex relationships between audio features, genre, and popularity.

## Final Model

### Engineered Features
We engineered three new features on top of the baseline:

1. `loudness_energy_ratio` (loudness / energy): captures the relationship between loudness and energy that neither feature alone conveys. A song can be loud but low energy, or high energy but quiet.
2. `tempo_filled` : filled 22,114 missing tempo values (19.4% of rows) using the genre median rather than the overall median — because a missing tempo in "classical" is likely different from a missing tempo in "hip-hop".
3. `num_artists` : counts the number of artists on a track by splitting the artists column by semicolons. D

### Feature Selection
Before selecting features, we evaluated each one using two methods:

**1. Correlation** — measures the linear relationship between each feature and popularity.

<iframe
  src="assets/feature_correlation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Not all features relate to popularity in the same way. Features like `release_year`, `loudness`, and `energy` show positive correlations, songs that are louder, more energetic, or more recently released tend to score higher in popularity. On the other side, `instrumentalness` and `acousticness` are negatively correlated, songs that are heavily instrumental or acoustic (think classical or folk) tend to be less popular on Spotify's mainstream charts. It's worth noting that correlation only captures linear relationships, so some features may still be useful even if their correlation looks small.


**2. Permutation Importance** — measures how much the model's F1 score drops when a feature is shuffled.

<iframe
  src="assets/permutation_importance.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

To get a fuller picture of which features actually matter to our model, we used permutation importance by shuffling each feature one at a time and measuring how much the F1 score drops. `track_genre` stands out as the single most important feature by a wide margin, which makes sense given how dramatically popularity varies across genres (pop at 64% vs classical at 4.9%). `release_year` and `acousticness` follow as the next most impactful. Even features like `num_artists` that showed weak linear correlation still contributed positively, a reminder that tree-based models can pick up on patterns that simple correlation misses.

### Hyperparameter Tuning

For the final model, we used `GridSearchCV` with 5-fold cross validation to tune the `RandomForestClassifier`. We tuned: <br> 
  • n_estimators — number of trees (too few → underfitting) <br> 
  • max_depth — depth of each tree (too deep → overfitting) <br> 

The best parameters were `max_depth=20` and `n_estimators=100`.

The best parameters were `max_depth=20` and `n_estimators=200`.

Our primary evaluation metric is **F1 score** — all model comparisons are based on F1. Additional metrics are included for reference.

| Metric | Baseline Model | Final Model |
|---|---:|---:|
| **F1-score (primary)** | **0.630** | **0.745** |
| Accuracy | 0.739 | 0.820 |
| Precision | 0.686 | 0.808 |
| Recall | 0.582 | 0.691 |

Our **F1 score improved from 0.630 to 0.745**, confirming that feature engineering and hyperparameter tuning meaningfully improved the  model's ability to identify popular tracks. Every other metric also improved, further validating this conclusion.


<iframe
  src="assets/confusion_matrix.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


The confusion matrix above shows how our final model performed on the 1,200 test tracks. Out of 457 truly popular tracks, the model correctly identified 317 (69.1% recall) while missing 140. Out of 743 truly non-popular tracks, the model correctly classified 667 (89.8%) while falsely labeling 76 as popular.

The model performs significantly better at identifying non-popular tracks than popular ones. This is expected given the class imbalance, with only 25% of tracks labeled popular, the model has seen fewer examples of what makes a track popular during training. Despite this challenge, an F1 score of 0.745 suggests the model has learned meaningful patterns from audio features, genre, and metadata to distinguish popular from non-popular tracks.

## Fairness Analysis

### Does Our Model Perform Equally Well Across Genre Groups?
Our EDA revealed something striking: popularity varies dramatically across genres. Pop tracks are popular 64% of the time, while classical tracks are popular only 4.9% of the time. This raised an important question: if our model was trained on data where some genres are rarely popular, does it struggle to identify popular tracks within those genres?
To investigate, we split tracks into two groups: lower-popularity genres (classical and country) and higher-popularity genres (electronic, hip-hop, metal, and pop), and tested whether our model performs equally well for both.

**Groups**: <br> 
Group X: Lower-popularity genres — classical, country <br> 
Group Y: Higher-popularity genres — electronic, hip-hop, metal, pop

**Evaluation metric** : F1 score

**Null hypothesis**: The model is fair. Its F1 score for lower-popularity and higher-popularity genres are roughly the same, and any differences are due to random chance.

**Alternative hypothesis** : The model is unfair. Its F1 score for lower-popularity genres is lower than for higher-popularity genres.

**Test statistic** : Difference in F1 scores (lower genres minus higher genres)

**Significance level**: 0.05

**Results**: The observed difference in F1 scores was -0.681. After running 1000 permutation trials, the p-value was 0.0. Since 0.0 < 0.05, we reject the null hypothesis.
<iframe
  src="assets/fairness_permutation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot above shows that the observed difference of -0.681 falls far outside the distribution of simulated differences, confirming that this result is not due to random chance. Our model performs significantly worse on classical and country tracks than on pop, hip-hop, metal, and electronic tracks. This is likely due to class imbalance — lower-popularity genres have very few popular tracks in the training data (classical at 4.9%, country at 15.7%), making it harder for the model to learn what makes them popular compared to genres like pop (64.4%).


### Is Our Model Fair Across Explicit and Non-Explicit Tracks?
Not all songs on Spotify are created equal — some carry an "E" badge indicating explicit content like strong language or adult themes. Since explicit tracks make up a smaller portion of our dataset, we wondered: does our model treat them fairly, or does it struggle to predict their popularity compared to clean tracks?
To test this, we ran a permutation test comparing the F1 score of our model on explicit tracks vs non-explicit tracks.

**Groups**: <br>
Group X: Explicit tracks (explicit=True) <br>
Group Y: Non-explicit tracks (explicit=False)

**Null hypothesis** : The model is fair. Its F1 score for explicit and non-explicit tracks are roughly the same, and any differences are due to random chance. <br>
**Alternative hypothesis** : The model is unfair. Its F1 score for explicit tracks is lower than for non-explicit tracks.

**Test statistic** : Difference in F1 scores (explicit minus non-explicit)

**Significance level** : 0.05

**Results** : The observed difference in F1 scores was -0.017. After running 1000 permutation trials, the p-value was 0.407. Since 0.407 > 0.05, we fail to reject the null hypothesis. The observed statistic falls well within the simulated distribution, suggesting the model performs roughly equally for explicit and non-explicit tracks — any difference is likely due to random chance.
<iframe
  src="assets/fairness_explicit.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot above shows the distribution of simulated F1 score differences across 1000 permutation trials. The dashed line represents our observed difference of -0.017, which falls well within the center of the simulated distribution. This visually confirms that the difference between explicit and non-explicit tracks is well within what we'd expect by random chance — supporting our conclusion that the model is fair across these two groups.

## Conclusion 
In this project, we built a binary classifier to predict whether a Spotify track is popular using its audio features, genre, and metadata. Starting from a simple baseline model using only `danceability` and `track_genre` (F1: 0.630), we improved performance significantly through feature engineering, adding more audio features, and hyperparameter tuning, reaching a final F1 score of 0.745.

Our analysis revealed that `track_genre` is by far the strongest predictor of popularity - pop and hip-hop tracks are dramatically more likely to be popular than classical or country tracks. This finding also surfaced an important limitation: our model performs significantly worse on lower-popularity genres, which is a direct consequence of class imbalance in the training data.

We also attempted to address this genre-based unfairness by setting `class_weight='balanced'` in our `RandomForestClassifier`, which instructs the model to pay more attention to underrepresented classes during training. However, this reduced the overall F1 score from 0.745 to 0.727, highlighting a common tension in machine learning: improving fairness for minority groups often comes at the cost of overall model performance.

While our model shows promise, popularity on Spotify is influenced by many factors beyond audio features, such as artist fame, playlist placement, and social trends all play a role that our dataset cannot capture. Future work could explore more sophisticated fairness techniques such as SMOTE oversampling or genre-specific models, as well as incorporating artist-level features or streaming data to better model popularity dynamics.

