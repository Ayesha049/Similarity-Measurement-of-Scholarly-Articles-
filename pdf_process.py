from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('./documents/s10875-024-01707-8.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[1]
parts = []

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if 50 < y < 750:
        parts.append(text)

# extracting text from page
text = page.extract_text(visitor_text=visitor_body)
text_body = "".join(parts)
print(text_body)
