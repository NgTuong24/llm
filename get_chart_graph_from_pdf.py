# import fitz  # PyMuPDF
# from PIL import Image
# from io import BytesIO
#
# def extract_images_from_pdf(pdf_path, output_folder, min_width=50, min_height=50):
#     # Mở tài liệu PDF
#     document = fitz.open(pdf_path)
#     image_count = 0
#
#     # Lặp qua từng trang trong tài liệu PDF
#     for page_index in range(len(document)):
#         page = document.load_page(page_index)
#         images = page.get_images(full=True)
#
#         # Lặp qua từng hình ảnh trong trang
#         for img_index, img in enumerate(images):
#             xref = img[0]
#             base_image = document.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]
#             image = Image.open(BytesIO(image_bytes))
#
#             # Kiểm tra kích thước hình ảnh
#             if image.width >= min_width and image.height >= min_height:
#                 image_filename = f"{output_folder}/image_{page_index + 1}_{img_index + 1}.{image_ext}"
#
#                 # Lưu hình ảnh vào tệp
#                 image.save(image_filename)
#                 image_count += 1
#
#     return image_count
#
#
# # Sử dụng hàm
# pdf_path = r'F:\CMC\CMC_Study\Code\data\kt.pdf'
# output_folder = r'F:\CMC\CMC_Study\Code\data_output'
# extracted_image_count = extract_images_from_pdf(pdf_path, output_folder)
# print(f"Extracted {extracted_image_count} images.")


import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# Đường dẫn đến file PDF
pdf_path = r'F:\CMC\CMC_Study\Code\data\kt.pdf'

# Mở file PDF
pdf_document = fitz.open(pdf_path)

# Duyệt qua các trang để tìm kiếm biểu đồ
for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)

    # Chuyển đổi trang thành hình ảnh
    zoom = 2  # Zoom vào để tăng độ phân giải
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Chuyển đổi pixmap thành numpy array để dùng với OpenCV
    img_data = pix.samples
    img_size = (pix.height, pix.width, pix.n)
    img = np.frombuffer(img_data, np.uint8).reshape(img_size)

    if pix.alpha:  # Nếu ảnh có alpha channel, chuyển sang BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Tạo một bản sao có thể ghi của hình ảnh
    img_copy = np.array(img, copy=True)

    # Chuyển đổi ảnh sang grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Dùng GaussianBlur để làm mờ ảnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dùng Canny edge detection để phát hiện các cạnh
    edged = cv2.Canny(blurred, 50, 150)

    # Tìm các đường viền trong ảnh
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ các đường viền lên ảnh gốc
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Chỉ lấy những contour có diện tích lớn hơn 1000
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lưu hình ảnh kết quả với các đường viền được vẽ
    result_image_path = f'F:\CMC\CMC_Study\Code\data_output/page_{page_number + 1}_contours.png'
    cv2.imwrite(result_image_path, img_copy)

    # Hiển thị hình ảnh kết quả (nếu cần)
    result_image = Image.open(result_image_path)
    result_image.show()

print("Đã phát hiện và đánh dấu các biểu đồ trên các trang PDF.")
