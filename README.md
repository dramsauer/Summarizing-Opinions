# Code for the paper "Exploring the Incorporation of Opinion Polarity for Abstractive Multi-Document Summarisation"

We use the [OpenNMT](https://opennmt.net/) neural machine translation system, but adapted it in some places.

The main steps of a OpenNMT-pipeline are
- Preprocessing 
- Model training
- Translate/Inference

You can find the documentation [here](https://opennmt.net/OpenNMT-py/quickstart.html#step-1-preprocess-the-data).


See also the python implementation of OpenNMT [here](https://github.com/OpenNMT/OpenNMT-py).

Afterwards, the translated summaries are scored via files2rouge package.

Data, model files and outputs files can be found here:
https://drive.google.com/drive/folders/1jMyCgCT8HR9etUib5fwaINi7QKU05TXO?usp=sharing
