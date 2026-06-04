# Spotify Track Popularity Prediction
### *Using audio features and genre to classify popular songs*

## Introduction

Music popularity is difficult to explain with one simple factor. A song may become popular because of the artist, the genre, playlist exposure, social trends, or the way the song sounds. In this project, we focus on the measurable side of music: the audio and metadata features available for Spotify tracks. Our goal is to explore whether these features can help us understand and predict which tracks are more likely to be popular.

We use the Spotify Music Tracks dataset, which contains 114,000 tracks and 22 columns. Each row represents one track and includes information such as the artist name, album name, release date, duration, explicit status, genre, and Spotify popularity score. The dataset also includes audio features such as danceability, energy, valence, acousticness, instrumentalness, loudness, speechiness, liveness, and tempo. These variables allow us to compare songs not only by genre or popularity, but also by musical characteristics.

Our main data science question is: Can we predict whether a Spotify track becomes popular using its audio features and genre? This question is relevant because music platforms, artists, and playlist curators often rely on data to understand listener behavior and identify patterns behind successful songs. While popularity is influenced by many outside factors that are not captured in this dataset, the available audio features still give us a way to study whether certain types of songs are more likely to perform well.



Prediction Problem

'Can we predict whether a Spotify track will be popular (popularity ≥ 55) using its audio features, genre, and metadata?'

This is a **binary classification** problem because the response variable, `is_popular` can take only one of two possible values: 

`True(1)`: The track has a popularity score greater than or equal to 55.
`False(0)`: The track has a popularity score less than 55. 
