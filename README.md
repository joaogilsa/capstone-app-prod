# Capstone project for Lisbon Data Science Academy

The model deployed receives information of individuals and search motive, returning true if such search has a likelihood of being successful >10%.
The information is saved into a database, along with the predicted outcome. The real outcome of the search can also be stored afterwards.

Two endpoints:
- should_search: predicts whether the search should be conducted
- search_outcome: updates the database entry with the actual search result.
