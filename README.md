# Spotify Track Popularity Prediction
#### **Using audio features, genre and metadata to classify popular songs**
**Authors: Anh Tran, Amelia Oo**

## Introduction

Music popularity is difficult to explain with one simple factor. A song may become popular because of the artist, the genre, playlist exposure, social trends, or the way the song sounds. In this project, we focus on the measurable side of music: the audio and metadata features available for Spotify tracks. Our goal is to explore whether these features can help us understand and predict which tracks are more likely to be popular.

We use the Spotify Music Tracks dataset, which contains 114,000 rows and 22 columns. Each row represents a single Spotify track, and the columns contain metadata and audio features associated with that track, such as the artist name, album name, release date, duration, explicit status, genre, and Spotify popularity score. The dataset also includes audio features such as danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, liveness, and tempo. These variables allow us to compare songs not only by genre or popularity, but also by musical characteristics.

Our main data science question is: **Can we predict whether a Spotify track becomes popular using its audio features, genre and metadata?** 
This question is relevant because music platforms, artists, and playlist curators often rely on data to understand listener behavior and identify patterns behind successful songs. While popularity is influenced by many outside factors that are not captured in this dataset, the available audio features still give us a way to study whether certain types of songs are more likely to perform well.

The dataset contains many variables; however, our analysis focuses on the following columns:

| Column | Description |
|----------|-------------|
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
2. Next, we checked for duplicate observations and removed duplicate tracks to ensure that each song was represented only once in the dataset. This helps prevent certain tracks from being overrepresented in both the analysis and predictive modeling process.
3. Then, we converted the `release_date` column to a datetime format and extracted the release year into a new column, `release_year`, which allows us to analyze how a track's release period may relate to its popularity.
4. We also created a new column by converting `duration_ms` into a more interpretable feature, `duration_min`, which represents track duration in minutes. In addition, we created another new feature, `num_artists`, by counting the number of artists associated with each track, allowing us to analyze whether collaborations are related to popularity.
5. Finally, after examining the distribution of Spotify popularity scores, we created a binary target variable, `is_popular`, where tracks with a popularity score of 55 or greater were labeled as popular and tracks below 55 were labeled as not popular. This transformed the problem into a classification task for our predictive modeling.

Below is the head of our cleaned DataFrame.



### Univariate Analysis

First, we explored the distribution of Spotify popularity scores. 


The histogram above shows a right-skewed distribution, with most tracks receiving relatively low to moderate popularity scores and fewer tracks achieving very high popularity. This suggests that highly popular songs are less common in the dataset, which may make popularity prediction more challenging since there are fewer examples of highly popular tracks for the model to learn from.


Next, we looked at the most common genres. 

This barchart above shows the top 10 most common genres in the dataset. The frequences of these genres appear to be very similar, with each genre containing close to 1000 tracks. This suggests that the dataset is relatively balanced across genres rather than being dominated by a few. Having a balanced distribution is crucial for conducting fair analysis and predictive modeling since it reduces the risk of bias toward a specific genre. 

We then examined the distribution of danceability scores. 

We observed that most songs have moderate-to-high danceability scores, indicating that Spotify tracks in this dataset tend to have rhythmically engaging music. 



Lastly, we examined the distribution of explicit and non-explicit tracks.

This chart reveals a significant difference between explicit and non-explicit tracks in the dataset. It appears that Non-explicit songs make up the vast majority of tracks, with over 100,000 entries, while explicit songs account for only a much smaller portion of the dataset. This suggests that mainstream music in the dataset is still largely dominated by non-explicit content. This imbalance may also affect later analysis or predictive modeling, where patterns associated with non-explicit tracks could have a stronger influence on the model due to their much larger representation in the dataset.

### Bivariate Analysis

To better understand factors associated with popularity, we examined the relationships between popularity and two track characteristics: danceability and explicit status.

First, we examined the relationship between danceability and popularity. 

The relationship between danceability and popularity does not appear strongly linear. However, highly popular songs are more frequently observed among tracks with moderate-to-high danceability scores, suggesting that danceability may contribute to a track's success, although it is unlikely to be the sole factor influencing popularity.

Next, we compared popularity distributions between explicit and non-explicit tracks.

Interestingly, although non-explicit tracks make up the majority of the dataset, explicit tracks appear to have a slightly higher median popularity. This may suggest that explicit content is relatively common among more popular or commercially successful tracks. However, both groups still show a wide spread in popularity, indicating that explicitness alone is not enough to determine whether a song becomes popular.

### Interesting Aggregates 

We also look at how audio features differ between popular and non-popular songs


The table above compares the average audio features between popular and non-popular songs. Popular songs tend to have slightly higher average danceability scores and higher loudness values, suggesting that more rhythmically engaging and louder tracks may perform better on Spotify. Popular songs also show lower average acousticness and instrumentalness, indicating that mainstream songs in the dataset are generally less acoustic and more vocal-focused.

However, many of the differences between the two groups remain relatively small, especially for features such as energy, speechiness, and valence. This suggests that no single audio feature alone strongly determines popularity, and that song popularity is likely influenced by a combination of multiple audio characteristics and genre.


## Assessment of Missingness

### NMAR Analysis

We believe the missing values in the `tempo` column are unlikely to be Not Missing At Random (NMAR). Instead, the missingness may be related to limitations in Spotify's audio analysis process or metadata collection. Since the reason a tempo value is missing is likely associated with factors other than the tempo itself, the missingness may be Missing At Random (MAR) rather than NMAR. 

### Missingness Dependency

We next examined the missingness of the `tempo` column by testing whether its missingness depends on other variables in the dataset. Specifically, we investigated whether the missingness of tempo depends on `track_genre` or `release_year`.

To conduct these tests, we created a Boolean indicator column, `tempo_missing`, which records whether the `tempo` value is missing for each track. 

#### Test 1: Track Genre and Tempo

**Null Hypothesis**: The missingness of `tempo` does not depend on `track_genre`.  
**Alternative Hypothesis**: The missingness of `tempo` depends on `track_genre`. 

**Test Statistic**: Total Variation Distance (TVD)  between the distribution of genres for tracks with missing tempo values and the distribution of genres for tracks with non-missing tempo values.

**Significance level**: 0.05


After performing the permutation test, we obtained a p-value of 0.0. Since the p-value is less than 0.05, we reject the null hypothesis. This suggests that the missingness of `tempo` does depend on `track_genre`.

#### Test 2: Release Year and Tempo

**Null Hypothesis**: The missingness of `tempo` does not depend on a track's `release_year`. 
**Alternative Hypothesis**: The missingness of `tempo` depends on a track's `release_year`. 

**Test Statistic**: Absolute Difference in the mean `release_year` between tracks with missing `tempo` values and tracks with non-missing `tempo` values. 

**Significance level**: 0.05

After performing the permutation test, we obtained a p-value of 0.0. Since the p-value is less than 0.05, we reject the null hypothesis. This suggests that the missingness of `tempo` does depend on `release_year`.

## Hypothesis Testing 

## Framing a Prediction Problem

The prediction problem is: can we predict whether a Spotify track is popular using its audio features, genre, and track metadata?
This is a binary classification problem. The response variable is `is_popular`, created by thresholding the popularity score at 50, tracks scoring 50 or above are labeled popular (1), and the rest not popular (0). This threshold labels the top 25% of tracks as popular, giving a reasonable class balance.

At the time of prediction, all audio features, genre, and metadata (duration, explicit status, release year, number of artists) are known properties of the track itself, not outcomes that happen after release.

We chose F1 score as our evaluation metric instead of accuracy because the dataset is imbalanced, which only about 25% of tracks are popular. A model could achieve high accuracy by simply predicting "not popular" for every track. F1 balances precision and recall, giving a more honest measure of how well the model identifies popular tracks.

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

The baseline model provides a reasonable starting point, but there is still room for improvement. In particular, the recall score shows that the model only identifies about 58% of popular tracks. This suggests that using only `danceability` and `track_genre` is not enough to fully capture the patterns behind track popularity.

For the final model, we plan to improve performance by adding more Spotify audio features, such as `energy`, `valence`, `acousticness`, `loudness`, `tempo`, and other track-level audio characteristics. We also plan to tune model hyperparameters to better capture more complex relationships between audio features, genre, and popularity.

## Final Model
### Feature Selection
Before selecting features, we evaluated each one using three methods:

1. Correlation — release_year (0.295), loudness (0.251), and energy (0.241) had the strongest positive correlations with popularity. instrumentalness (-0.209) and acousticness (-0.199) had the strongest negative correlations, highly instrumental or acoustic songs tend to be less popular on Spotify, which is dominated by vocal-heavy genres like pop and hip-hop.

2. Permutation Importance — measures how much the model's F1 score drops when a feature is shuffled. track_genre (0.278) was by far the most important feature, followed by release_year (0.044). Every feature had positive permutation importance, confirming all features contributed to the model.

3. Ablation Testing — removing any single feature dropped model accuracy, even features with low correlation like num_artists (-0.038). This confirms that Random Forest captures non-linear patterns that correlation alone misses.
