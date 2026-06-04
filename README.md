# Spotify Track Popularity Prediction
#### **Using audio features and genre to classify popular songs**

## Introduction

Music popularity is difficult to explain with one simple factor. A song may become popular because of the artist, the genre, playlist exposure, social trends, or the way the song sounds. In this project, we focus on the measurable side of music: the audio and metadata features available for Spotify tracks. Our goal is to explore whether these features can help us understand and predict which tracks are more likely to be popular.

We use the Spotify Music Tracks dataset, which contains 114,000 tracks and 22 columns. Each row represents one track and includes information such as the artist name, album name, release date, duration, explicit status, genre, and Spotify popularity score. The dataset also includes audio features such as danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, liveness, and tempo. These variables allow us to compare songs not only by genre or popularity, but also by musical characteristics.

Our main data science question is: Can we predict whether a Spotify track becomes popular using its audio features and genre? This question is relevant because music platforms, artists, and playlist curators often rely on data to understand listener behavior and identify patterns behind successful songs. While popularity is influenced by many outside factors that are not captured in this dataset, the available audio features still give us a way to study whether certain types of songs are more likely to perform well.

## Relevant Columns

| Column             | Type              | Description                                                                           | Why It Matters                                                                        |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `popularity`       | Numeric           | Spotify popularity score from 0 to 100. Higher values mean the track is more popular. | Used as the original popularity measure and to create the prediction target.          |
| `is_popular`       | Binary / derived  | `True` if a track’s popularity score is at least 55, otherwise `False`.               | This is the response variable for the classification model.                           |
| `track_genre`      | Categorical       | Genre label assigned to each track.                                                   | Used to compare popularity patterns across genres and as a model feature.             |
| `danceability`     | Numeric           | Measures how suitable a track is for dancing, from 0 to 1.                            | Helps capture rhythm and beat-related qualities of a song.                            |
| `energy`           | Numeric           | Measures the intensity and activity of a track, from 0 to 1.                          | Useful for understanding whether more energetic tracks are more likely to be popular. |
| `valence`          | Numeric           | Measures the musical positivity of a track, from 0 to 1.                              | Helps capture whether happier-sounding tracks differ in popularity.                   |
| `acousticness`     | Numeric           | Measures how likely a track is to be acoustic, from 0 to 1.                           | Useful for comparing acoustic versus more electronically produced tracks.             |
| `instrumentalness` | Numeric           | Estimates whether a track contains no vocals, from 0 to 1.                            | Helps distinguish instrumental tracks from vocal-heavy tracks.                        |
| `loudness`         | Numeric           | Overall loudness of the track in decibels.                                            | Captures intensity and production style.                                              |
| `tempo`            | Numeric           | Estimated beats per minute of the track.                                              | Represents song speed and is also used in the missingness analysis.                   |
| `duration_min`     | Numeric / derived | Track duration converted from milliseconds to minutes.                                | Easier to interpret than `duration_ms` and useful for analyzing song length.          |
| `explicit`         | Boolean           | Indicates whether the track contains explicit content.                                | Can help test whether explicit tracks differ in popularity.                           |



## Framing a Prediction Problem

The prediction problem is: can we predict whether a Spotify track is popular using its audio features and metadata?

This is a binary classification problem. The response variable is is_popular, a column we created by thresholding the popularity score at 50, tracks with a score of 50 or above are labeled popular (1), and the rest are labeled not popular (0). We chose this threshold because it labels the top 25% of tracks as popular, giving a reasonable class balance.

We chose F1 score as our evaluation metric instead of accuracy because the dataset is imbalanced which is only about 25% of tracks are popular. A model could achieve high accuracy by simply predicting "not popular" for every track, which is not useful. F1 balances precision and recall, giving a more honest measure of how well the model identifies popular tracks.

Prediction Problem

'Can we predict whether a Spotify track will be popular (popularity ≥ 55) using its audio features, genre, and metadata?'

This is a **binary classification** problem because the response variable, `is_popular` can take only one of two possible values: 

`True(1)`: The track has a popularity score greater than or equal to 55.
`False(0)`: The track has a popularity score less than 55. 
