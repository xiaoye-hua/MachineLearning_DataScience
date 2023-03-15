

## TODO & Questions

2. [x] how to eval impact? -> downstream recommendation problem
3. [x] how to get the embedding -
4. [x] how many data are used
    1. 1M hotels
    2. 60M click sessions
5. [x] to read
    1. [x] Chapter 2
    2. [x] Real-Time Personalization Using Embeddings for Search Ranking at Airbnb 

## Problem Definition

1. [x] key metrics: downstream recommendation problem

## Previous Methods & New Methods

1. Previous: Embedding only based on user-item interaction (click sequence)
    1. Basic idea: we seek to predict the context hotels (words), given a target hotel (word) in the session (sentence).
    2. shortcoming
        1. unseen hotels
        2. [ ] In addition, the model may be forced to learn certain semantic features which capture aspects of user interest, hotel geographic information, hotel attributes, and so on,
2. New: Except user-item interaction, supplement, structure hotel information has been included, such as 
    1. hotel attributes (e.g., property type, star rating, average user rating)
    2. amenity information (e.g., if the hotel has free Wi-Fi or free breakfast)
    3. geographic information that leverages an hexagonal geospatial system as well as spatial encoders.
        1. [ ] spatial encoder ??
3. How to train it:     
    1. Each aspect of the hotel 1) the one-hot
encoding of the clicked property ids within the same session, 2) the associated amenities and 3) the geographical
information is embedded separately, and these representations are later concatenated and further compressed before
being used for context prediction.
        1. Vc(click), Va(amenity), Vg(geographcal) -> Ve (final embedding)

## Impacts

1. Most important advantage: addresses the cold-start problem for hotels with insufficient
historical click information by incorporating additional hotel attributes, which are available for all hotels.
    1. [x] how???
        1. . For newly listed hotels with no click-session information, one can simply
choose 𝑉𝑐 for new hotels at random and hotel2vec computes 𝑉𝑒 using the randomly initialized 𝑉𝑐 and the other hotel
attributes which are known even for recently listed hotels
    2. [x] test result, several method
        1. MF based vector
        2. session-only based vector 

## Further Improvement

