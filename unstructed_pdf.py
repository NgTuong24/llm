# import fitz  # PyMuPDF
# from PIL import Image
# import io
#
# # Define the PDF path
# pdf_path = r'F:\CMC\CMC_Study\Code\data\kt.pdf'
#
# # Define bounding boxes for cropping (left, top, right, bottom)
# bboxes = [
#     (50, 50, 200, 200),  # First area
#     (250, 50, 400, 200),  # Second area
#     (50, 250, 200, 400),  # Third area
#     (250, 250, 400, 400)  # Fourth area
# ]
#
# # Open the PDF file
# pdf_document = fitz.open(pdf_path)
#
# # Iterate over each page
# for page_number in range(len(pdf_document)):
#     page = pdf_document.load_page(page_number)
#
#     # Iterate over each bounding box
#     for i, bbox in enumerate(bboxes):
#         # Crop the page using the bounding box
#         page.set_cropbox(bbox)
#
#         # Render the cropped page to an image
#         pix = page.get_pixmap()
#
#         # Convert to PIL Image
#         image = Image.open(io.BytesIO(pix.tobytes()))
#
#         # Save the image
#         image_filename = f'F:\CMC\CMC_Study\Code\data\output\page_{page_number + 1}_crop_{i + 1}.png'
#         image.save(image_filename)
#         print(f'Saved: {image_filename}')
#
# # Close the PDF document
# pdf_document.close()


# _-----------------------
#
# import fitz  # PyMuPDF
# from PIL import Image
# import io
#
# # Define the PDF path
# pdf_path = r'F:\CMC\CMC_Study\Code\data\kt.pdf'
#
# # Define bounding boxes for cropping (left, top, right, bottom)
# bboxes = [
#     (50, 50, 200, 200),  # First area
#     (250, 50, 400, 200),  # Second area
#     (50, 250, 200, 400),  # Third area
#     (250, 250, 400, 400)  # Fourth area
# ]
#
# # Open the PDF file
# pdf_document = fitz.open(pdf_path)
#
# # Iterate over each page
# for page_number in range(len(pdf_document)):
#     page = pdf_document.load_page(page_number)
#
#     # Iterate over each bounding box
#     for i, bbox in enumerate(bboxes):
#         # Define the rectangle for cropping
#         rect = fitz.Rect(bbox)
#
#         # Get the pixmap for the specific area
#         pix = page.get_pixmap(clip=rect)
#
#         # Convert to PIL Image
#         image = Image.open(io.BytesIO(pix.tobytes()))
#
#         # Save the image
#         image_filename = f'F:\\CMC\\CMC_Study\\Code\\data\\output\\page_{page_number + 1}_crop_{i + 1}.png'
#         image.save(image_filename)
#         print(f'Saved: {image_filename}')
#
# # Close the PDF document
# pdf_document.close()

# --------------------------------
import fitz  # PyMuPDF

# Define the PDF path
pdf_path = r'F:\CMC\CMC_Study\Code\data\kt.pdf'

# Define bounding boxes for cropping (left, top, right, bottom)
bboxes = [
    (50, 50, 200, 200),  # First area
    (250, 50, 400, 200),  # Second area
    (50, 250, 200, 400),  # Third area
    (250, 250, 400, 400)  # Fourth area
]

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Iterate over each page
for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)

    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        # Define the rectangle for cropping
        rect = fitz.Rect(bbox)

        # Get the pixmap for the specific area
        pix = page.get_pixmap(clip=rect)

        # Save the pixmap directly as a PNG
        image_filename = f'F:\\CMC\\CMC_Study\\Code\\data\\output\\page_{page_number + 1}_crop_{i + 1}.png'
        pix.save(image_filename)
        print(f'Saved: {image_filename}')

# Close the PDF document
pdf_document.close()
