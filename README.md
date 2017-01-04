# Aligning Reddit and Wikipedia
A dataset that aligns knowledge from Wikipedia in the form of sentences with sequences of Reddit utterances. The dataset consists of sequences of comments and a number of Wikipedia sentences that were allocated randomly from the Wikipedia pages to which each sequence is aligned. The resultant dataset consists of 15k sequences of comments that are aligned with 75k Wikipedia sentences.

For a detailed description of this dataset please refer to original paper: https://aclweb.org/anthology/C/C16/C16-1318.pdf.

## Contents
* The HDF5 files (i) `Aligned-Dataset/reddit.h5` and (ii) `Aligned-Dataset/wikipedia.h5` are built in such a way that each sequence of comments on Reddit is aligned with 20 Wikipedia sentences.
* `Inspect-Dataset.ipynb` is a Python script on iPython Notebook that allows easier inspection of the above aligned dataset.
* The folders `Data/Reddit` and `Data/Wikipedia` contain the respective Reddit sequences of comments and Wikipedia sentences as these have been initially extracted by utilising the search feature of both their APIs (i.e. https://www.reddit.com/dev/api/ and https://www.mediawiki.org/wiki/API:Main_page).

## BibTeX
Please cite the following paper should you use this dataset in your work.
```
InProceedings{vougiouklis-hare-simperl:2016:COLING,
  author    = {Vougiouklis, Pavlos  and  Hare, Jonathon  and  Simperl, Elena},
  title     = {A Neural Network Approach for Knowledge-Driven Response Generation},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  publisher = {The COLING 2016 Organizing Committee},
  pages     = {3370--3380},
  url       = {http://aclweb.org/anthology/C16-1318}
}
```

## License
This project is licensed under the terms of the Apache 2.0 License.
