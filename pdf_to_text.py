import PyPDF2
import glob, os,re


pdf_names = []
pdf_txt_names = []
os.chdir("/home/rajas/PDF_spider")
for file in glob.glob("*.pdf"):
    pdf_names.append(file)
    pdf_txt_names.append(file[:-3]+"txt")

zipped = zip(pdf_names,pdf_txt_names)
# creating a pdf file object
for i,j in zipped:

    pdfFileObj = open(i, 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    # printing number of pages in pdf file
    num = pdfReader.numPages

    # extracting text from page

    with open(j, "a") as myfile:
        for i in range(num):
            pageObj = pdfReader.getPage(i)
            myfile.write(pageObj.extractText())
    # closing the pdf file object
    pdfFileObj.close()
