# Aligning Reddit and Wikipedia
A dataset that aligns knowledge from Wikipedia in the form of sentences with sequences of Reddit utterances. The dataset consists of sequences of comments and a number of Wikipedia sentences that were allocated randomly from the Wikipedia pages to which each sequence is aligned. The resultant dataset consists of 15k sequences of comments that are aligned with 75k Wikipedia sentences.

## Contents
* The HDF5 files (i) `Aligned-Dataset/reddit.h5` and (ii) `Aligned-Dataset/wikipedia.h5` are built in such a way that each sequence of comments on Reddit is aligned with 20 Wikipedia sentences.
* `Inspect-Dataset.ipynb` is a Python script on iPython Notebook that allows easier inspection of the above aligned dataset.
* The folders `Data/Reddit` and `Data/Wikipedia` contain the respective Reddit sequences of comments and Wikipedia sentences as these have been initially extracted by utilising the search feature of both their APIs (i.e. https://www.reddit.com/dev/api/ and https://www.mediawiki.org/wiki/API:Main_page)

## License
This project is licensed under the terms of the Apache 2.0 License.
