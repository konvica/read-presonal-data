# read-presonal-data

## Goal

Extraction of Text and Identification of face in documents. Given a document of type PDF, PNG or JPG, the program should

1. Extract all the text present in the document
2. Classify if the document is Personal or Non-Personal
3. Identify if the document contains any face in it

## Setup

Install conda environment from env.yml

```bash
conda env create -f env.yml
python -m spacy download en_core_web_lg
mkdir require
curl -o require/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
streamlit run streamlit_app.py
```

## References

- https://pymupdf.readthedocs.io/en/latest/intro.html
- https://github.com/pymupdf/PyMuPDF
- https://docs.streamlit.io/en/stable/api.html
- https://github.com/streamlit/streamlit
- https://towardsdatascience.com/nlp-approaches-to-data-anonymization-1fb5bde6b929
- https://github.com/microsoft/presidio
- https://microsoft.github.io/presidio/supported_entities/
- https://microsoft.github.io/presidio/getting_started/
- https://spacy.io/api
- https://www.johnsnowlabs.com/simpler-more-accurate-deidentification-in-spark-nlp-for-healthcare/
