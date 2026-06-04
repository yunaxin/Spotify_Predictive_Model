# Spotify Track Popularity Prediction
#### **Using audio features and genre to classify popular songs**

## Introduction

Music popularity is difficult to explain with one simple factor. A song may become popular because of the artist, the genre, playlist exposure, social trends, or the way the song sounds. In this project, we focus on the measurable side of music: the audio and metadata features available for Spotify tracks. Our goal is to explore whether these features can help us understand and predict which tracks are more likely to be popular.

We use the Spotify Music Tracks dataset, which contains 114,000 rows and 22 columns. Each row represents a single Spotify track, and the columns contain metadata and audio features associated with that track, such as the artist name, album name, release date, duration, explicit status, genre, and Spotify popularity score. The dataset also includes audio features such as danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, liveness, and tempo. These variables allow us to compare songs not only by genre or popularity, but also by musical characteristics.

Our main data science question is: **Can we predict whether a Spotify track becomes popular using its audio features and genre?** 
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

To focus on our research question, we kept only the columns described above and removed the remaining columns from the dataset.

1. We start by examining the dataset for missing values. We found that the `tempo` column contained over 22,000 missing values, which could affect analyses involving tempo. Since tempo is an important audio feature, we chose to preserve these missing values for later missingness analysis rather than immediately removing the affected observations.
2. Next, we checked for duplicate observations and removed duplicate tracks to ensure that each song was represented only once in the dataset. This helps prevent certain tracks from being overrepresented in both the analysis and predictive modeling process.
3. Then, we converted the `release_date` column to a datetime format and extracted the release year into a new column, `release_year`, which allows us to analyze how a track's release period may relate to its popularity.
4. We also created a new column by converting `duration_ms` into a more interpretable feature, `duration_min`, which represents track duration in minutes. In addition, we created another new feature, `num_artists`, by counting the number of artists associated with each track, allowing us to analyze whether collaborations are related to popularity.
5. Finally, after examining the distribution of Spotify popularity scores, we created a binary target variable, `is_popular`, where tracks with a popularity score of 55 or greater were labeled as popular and tracks below 55 were labeled as not popular. This transformed the problem into a classification task for our predictive modeling.

Below is the head of our cleaned DataFrame.



### Univariate Analysis

First, we explored the distribution of Spotify popularity scores. 


The histogram above shows a right-skewed distribution, with most tracks receiving relatively low to moderate popularity scores and fewer tracks achieving very high popularity. This suggests that highly popular songs are less common in the dataset, which may make popularity prediction more challenging since there are fewer examples of highly popular tracks for the model to learn from.


Next, 

## Framing a Prediction Problem

The prediction problem is: can we predict whether a Spotify track is popular using its audio features and metadata?

This is a binary classification problem. The response variable is `is_popular`, a column we created by thresholding the popularity score at 50, tracks with a score of 50 or above are labeled popular (1), and the rest are labeled not popular (0). We chose this threshold because it labels the top 25% of tracks as popular, giving a reasonable class balance.

We chose F1 score as our evaluation metric instead of accuracy because the dataset is imbalanced which is only about 25% of tracks are popular. A model could achieve high accuracy by simply predicting "not popular" for every track, which is not useful. F1 balances precision and recall, giving a more honest measure of how well the model identifies popular tracks.

Prediction Problem

'Can we predict whether a Spotify track will be popular (popularity ≥ 55) using its audio features, genre, and metadata?'

This is a **binary classification** problem because the response variable, `is_popular` can take only one of two possible values: 

`True(1)`: The track has a popularity score greater than or equal to 55.
`False(0)`: The track has a popularity score less than 55. 
