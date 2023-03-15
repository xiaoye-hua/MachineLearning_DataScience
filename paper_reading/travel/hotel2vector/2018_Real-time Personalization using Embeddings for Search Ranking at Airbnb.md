

## TODO & Questions

1. [x] data info
    1. listing embedding: 
        1. 800M click sessions
        2. 4.5M listings
2. [x] how to eval?? -> While some listing characteristics, such as price, do not need to be learned because they can be extracted from listing meta-data, other types of listing characteristics, such as architecture, style and feel are much harder to extract in form of listing features.
    1. Clustering -> shown on map 
    2. Similarity difference in difference dimensions
    3. Embedding evaluation tools 
3. [Video demostration of this paper](https://www.youtube.com/watch?v=aWjsUEX7B1I&t=415s)
## Problem Definition

1. [x] key metrics
2. Two application
    1. real-time personalization rec
    2. Similar listing rec

## Previous Methods & New Methods

1. Key contribution
    1. real-time embedding 
    2. adapt it to their marketing
        1.  Congregated training
        2. more weight to booking item
    3. short & long term interest
        1. short: item embedding based on clicked listings
        2. long: user_type, item_type embedding based on booked listings

## Impacts

1. similar hotel rec
    1. 21% increase in CTR
    2. 4.6% increase in bookings

## Further Improvement

