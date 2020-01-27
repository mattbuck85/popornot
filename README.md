# Pop or Not

Binary classifier for Pop music with Logistic Regression and data from the Spotify API.  Pop music is defined as chart-topping hits from the 80s and 90s, and Eclectic music is defined as songs that spent limited or no time on the charts, however achieved some cult popularity, and is known for more complex rhythms and changes in the music.

This classifier postulates that metrics based on musical complexity can help predict music's popularity.  No hypothesis has yet been formulated on this assumption, because it is based on other assumptions that are beyond the scope of this project to provide.  It can be said that all of these assumptions are based on the perceptions of experts.

Despite a lack of evidentiary foundation, it has been found that complexity features are important to this model.  Music scouting is fundamental to signing artists, and these features could create better automation tools for that.

# Dependencies

1. Pandas
2. Seaborn
3. matplotlib
4. Scikit-Learn
5. Flask

# Spotify Playlists

## Pop

251 Tracks

1. 80s Pop Hits
2. 90s Pop Party
3. Indie Pop

## Eclectic

159 Tracks

1. Prog Rock Monsters
2. Custom additions

# Features

Spotify's API provided these features which were directly used in Logistic Regression modeling:

1. Danceability
2. Valence

These API features were used in Feature Engineering to determine musical complexity.

1. Track Duration
2. Tempo, Time Signature
3. Key, Mode
4. Tempo Confidence, Time Signature Confidence

These come from the Sections API, and subfeatures are derived by looking at the changes over time:

1. key_mode_changes
2. time_signature_changes
3. unique_time_signatures: nunique(time_signature)
4. key_mode_variety: nunique(mode + key)

Then the complexity features:

1. changes_per_minute: (key_mode_changes + time_signature_changes) / duration
2. variety: (unique_time_signatures + key_mode_variety) / duration

First the confidence penalty is generated, then all complexity scores are penalized

Confidence Penalty:  min(time_signature_confidence**-1 + tempo_confidence**-1, 1)

Giving us our final complexity features:

1. Duration Score: min(duration / 60, 6) * confidence_penalty
2. penalized_cpm = changes_per_minute * confidence_penalty
3. penalized_variety = changes_per_minute * confidence_penalty

# Model Performance

A logistic regression model was chosen due to the linear seperability of the features, as well as provide the interpretability needed for feature importance and any future hypothesis testing.


sklearn train_test_split(X, y, test_size=.30, random_state=12)

random state chosen to more evenly distribute the pop/eclectic group between train and test.  Precision/recall was tuned to 0.35 to optimize for the F1 and ROC/AUC, as well as get as many positive results for pop music as possible.

Precision:  0.850,  Recall:  0.986, F1:  0.913
ROC/AUC:  0.882
