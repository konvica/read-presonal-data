import io
import logging
import os

import cv2 as cv
import en_core_web_lg
import fitz
import numpy as np
import streamlit as st
from PIL import Image
from presidio_analyzer import AnalyzerEngine
from presidio_image_redactor import ImageAnalyzerEngine

SCORE_THRESHOLD = 0.5


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# Load models - Because this is cached it will only happen once.
@st.cache(allow_output_mutation=True)
def load_nlp():
    logging.info("loading spacy nlp")
    return en_core_web_lg.load()


@st.cache(allow_output_mutation=True)
def load_analyzer():
    logging.info("loading analyzer")
    return AnalyzerEngine()


@st.cache(allow_output_mutation=True)
def load_image_analyzer():
    logging.info("loading image analyzer")
    return ImageAnalyzerEngine()


@st.cache(allow_output_mutation=True)
def load_face_detector():
    logging.info("loading face cascade detector")
    face_cascade = cv.CascadeClassifier()
    face_cascade.load("require/haarcascade_frontalface_default.xml")
    return face_cascade


def process_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    face_detector = load_face_detector()
    faces = face_detector.detectMultiScale(gray)
    return faces


def process_image_ocr(image):
    ocr_analyzer = load_image_analyzer()
    res = ocr_analyzer.analyze(image)
    return np.array([[spot.left, spot.top, spot.width, spot.height] for spot in res])


def process_image(image):
    faces = process_face(image)
    spots = process_image_ocr(image)
    if (len(faces) > 0) & (len(spots) > 0):
        return np.concatenate((faces, spots), axis=0)
    elif len(faces) > 0:
        return faces
    elif len(spots) > 0:
        return spots
    else:
        return np.array([])


def process_text(text):
    analyzer = load_analyzer()
    # Call analyzer to get results
    results = analyzer.analyze(text=text, language='en', )
    return filter(lambda pii: pii.score > SCORE_THRESHOLD, results)


def process_page(page):
    ## scan for images
    data = page.get_text('dict')
    img_data = list(filter(lambda block: block['type'] == 1, data['blocks']))

    for i, img in enumerate(img_data):
        img_bytes = img['image']
        if img['bpc'] == 8:
            decoded = cv.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        elif img['bpc'] == 16:
            decoded = cv.imdecode(np.frombuffer(img_bytes, np.uint16), -1)
        else:
            logging.error(f"Unknown bitrate - img.bpc:{img['bpc']}")
            raise (f"Unknown bitrate - img.bpc:{img['bpc']}")
        faces_spots = process_image(decoded)
        img_data[i]['faces_spots'] = faces_spots

    ## scan text
    text = page.get_text('text')
    # text_data = list(filter(lambda block: block['type'] == 0, data['blocks']))
    # text_lines = [span for block in text_data for line in block['lines'] for span in line['spans']]
    piis = process_text(text)
    return img_data, piis


def process_pdf(doc):
    anot_texts = []
    page_imgs = []
    for j in range(doc.page_count):
        page = doc.load_page(j)
        pdf_img_data, pdf_page_piis = process_page(page)

        ## convert page to numpy image
        pix = page.get_pixmap()
        with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as pil_:
            page_img = np.array(pil_)
        ## draw faces on page
        for block in pdf_img_data:
            x0, y0, x1, y1 = block['bbox']
            for face in block['faces_spots']:
                x, y, w, h = face
                rel_x0 = (x / block['width']) * (x1 - x0)
                rel_x1 = ((x + w) / block['width']) * (x1 - x0)
                rel_y0 = (y / block['height']) * (y1 - y0)
                rel_y1 = ((y + h) / block['height']) * (y1 - y0)
                cv.rectangle(page_img, (int(x0 + rel_x0), int(y0 + rel_y0)),
                             (int(x0 + rel_x1), int(y0 + rel_y1)), (255, 0, 0), 2)
        # construct colored text
        text = page.get_text('text')
        i = 0
        pretty_text = "<div>"
        for pii in pdf_page_piis:
            pretty_text += text[i:pii.start]
            pretty_text += " <span class='highlight red'>" + \
                           text[pii.start:pii.end] + \
                           f"<span class='bold'>{str(pii.entity_type)}</span> </span>"
            i = pii.end
        pretty_text += "</div>"

        page_imgs.append(page_img)
        anot_texts.append(pretty_text)
    return anot_texts, page_imgs


def process_image_final(image):
    faces_spots = process_image(image)
    output_img = image.copy()
    for spot in faces_spots:
        x, y, w, h = spot
        cv.rectangle(output_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    anot_texts = []  # text from ocr is not used now, only detected PII
    page_imgs = [output_img]
    return anot_texts, page_imgs


def process_file(buff, ext):
    anot_texts, page_imgs = [], []
    if ext == 'pdf':
        with fitz.open(stream=buff.read(), filetype="pdf") as doc:
            anot_texts, page_imgs = process_pdf(doc)
    elif ext in ["jpg", "jpeg", "png", "PNG", 'JPG']:
        with Image.open(buff) as buf:
            image = np.array(buf)
        if image.shape[-1] == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
        anot_texts, page_imgs = process_image_final(image)
    with st.beta_expander("Analyzed text"):
        # svg = page.get_svg_image()
        # b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        # html = r'<img src="data:image/svg+xml;base64,{}"/>'.format(b64)
        # st.write(html, unsafe_allow_html=True)
        for pretty_text in anot_texts:
            st.markdown(pretty_text, unsafe_allow_html=True)

    for page_img in page_imgs:
        st.image(page_img)


def main():
    st.title('Detect Personal Information')
    img_file_buffer = st.file_uploader("Upload an pdf/image", type=["png", "jpg", "jpeg", "pdf", "PNG", 'JPG'])

    if img_file_buffer is not None:
        ext = os.path.splitext(img_file_buffer.name)[-1].replace(".", "")
        file_buff = img_file_buffer
    else:
        logging.info("No input file - using sample image.")
        ext = 'png'
        with open("data/tds_jessica.png", 'rb') as buf:
            file_buff = io.BytesIO(buf.read())
    process_file(file_buff, ext)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.info(fitz.__doc__)
    local_css("style.css")
    main()
