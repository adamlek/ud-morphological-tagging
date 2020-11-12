# ud-morphological-tagging

Code for the paper "Composing Byte-Pair Encodings for Morphological Sequence Classification"

## How to
You can run the system by ``python3 main.py``

in ``train_and_test_all_languages()`` you can choose which treebanks to run, and which composition functions to use.

Hyperparameters can be changes in ``args.py``

The UD treebank data is available in .data/, all languages in UD are here, so if you want to run a language with this model, simply add the language in the function ``train_and_test_all_languages()``.

## Note-to-self:
* Clean the code
* Improve ease of use
