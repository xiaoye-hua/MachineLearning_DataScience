## TODO & Questions

1. Authors introduced Click Feedback feature (CF)
    1. COEC in different segmentation
    2. expected clicks

## Problem Definition

1. Main objective: the design of **personalized** click prediction 
2. Essential part: development of new user-related features.
    1. No user-related features in baseline model

## Previous Methods & new Methods

1. New method: add user features
    1. demographic group features 
    2. user-specific features
        1. activity normalized COEC -> worth to explore
## Impacts

1. Another interesting point is that
adding demographic features to the model that already included user-specific features helped improve accuracy, especially for users without enough user-specific click feedback.

## Further Improvement
1. Another direction for future research is to take into account session-based information.
2. group users in different ways -> clustering maybe
