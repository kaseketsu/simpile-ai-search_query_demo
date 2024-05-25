import PyPDF2
def extract_text(file_name):
    paragraphs = []
    with open(file_name,'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for pages in pdf_reader.pages:
            text = pages.extract_text()
            if text:
                paragraph = text.split('\n\n')
                paragraphs.extend(paragraph)
    return paragraphs
