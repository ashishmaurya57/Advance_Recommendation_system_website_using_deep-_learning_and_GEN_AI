import pikepdf
from django.core.files.base import ContentFile
from io import BytesIO

def compress_pdf(file):
    """
    Compress a PDF file using pikepdf.
    :param file: Uploaded PDF file
    :return: Compressed PDF file
    """
    try:
        # Read the uploaded PDF file
        input_pdf = BytesIO(file.read())
        output_pdf = BytesIO()

        # Compress the PDF using pikepdf
        with pikepdf.open(input_pdf) as pdf:
            pdf.save(output_pdf, optimize_image=True)

        # Return the compressed PDF as a Django ContentFile
        return ContentFile(output_pdf.getvalue(), name=file.name)
    except Exception as e:
        print(f"Error compressing PDF: {e}")
        return file  # Return the original file if compression fails

# from .models import ProductInteraction, product

# def handle_like_dislike(user, product_id, action):
#     product_instance = product.objects.get(id=product_id)
#     interaction, created = ProductInteraction.objects.get_or_create(user=user, product=product_instance)

#     if action == "like":
#         if not interaction.liked:
#             interaction.liked = True
#             interaction.disliked = False
#             product_instance.likes += 1
#             if not created and interaction.disliked:
#                 product_instance.dislikes -= 1
#     elif action == "dislike":
#         if not interaction.disliked:
#             interaction.disliked = True
#             interaction.liked = False
#             product_instance.dislikes += 1
#             if not created and interaction.liked:
#                 product_instance.likes -= 1

#     interaction.save()
#     product_instance.save()    

# app_name/utils.py

# user/utils.py
# views.py
# from django.http import JsonResponse
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from PyPDF2 import PdfReader
# from django.conf import settings
# import torch
# import os
# from yourapp.models import UploadedPDF  # Adjust based on your model

# # Load model only once
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# LANG_CODE_MAP = {
#     "english": "eng_Latn",
#     "hindi": "hin_Deva",
#     "french": "fra_Latn",
#     "german": "deu_Latn",
#     "telugu": "tel_Telu",
#     "tamil": "tam_Taml",
#     "bengali": "ben_Beng",
#     "marathi": "mar_Deva",
#     "gujarati": "guj_Gujr",
# }

# def translate_visible_pages(request):
#     if request.method == "POST":
#         pdf_id = request.POST.get("pdf_id")
#         language = request.POST.get("language", "hindi").lower()
#         pages = request.POST.getlist("pages[]")  # Expecting array like ["3", "4", "5"]

#         if not pdf_id or not pages or language not in LANG_CODE_MAP:
#             return JsonResponse({"error": "Invalid input"}, status=400)

#         try:
#             pdf_instance = UploadedPDF.objects.get(id=pdf_id, user=request.user)
#             pdf_path = os.path.join(settings.MEDIA_ROOT, str(pdf_instance.file))
#             reader = PdfReader(pdf_path)

#             lang_code = LANG_CODE_MAP[language]
#             translated = []

#             for page_no in map(int, pages):
#                 if page_no < 1 or page_no > len(reader.pages):
#                     translated.append({"page": page_no, "text": "[Invalid page number]"})
#                     continue

#                 page = reader.pages[page_no - 1]
#                 text = page.extract_text()
#                 if not text or len(text.strip()) < 10:
#                     translated.append({"page": page_no, "text": "[No meaningful text found]"})
#                     continue

#                 inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
#                 translated_ids = model.generate(
#                     **inputs,
#                     forced_bos_token_id=tokenizer.lang_code_to_id[lang_code],
#                     max_length=1024
#                 )
#                 output = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
#                 translated.append({"page": page_no, "text": output})

#             return JsonResponse({"translated_pages": translated})

#         except UploadedPDF.DoesNotExist:
#             return JsonResponse({"error": "PDF not found or access denied"}, status=404)
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
