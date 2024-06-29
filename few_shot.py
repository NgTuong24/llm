
import google.generativeai as genai
import os

# Đặt API key của bạn
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAk2SGsbPm5H-6K-rNgnIhQsBYwkm2GHhE'
genai.configure(api_key='AIzaSyAk2SGsbPm5H-6K-rNgnIhQsBYwkm2GHhE')

# from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser
# import google.generativeai as genai
# import os
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyAjgSFL4cLuqfZ9gkDhh7MV535edefCTZw'
# genai.configure(api_key='AIzaSyAjgSFL4cLuqfZ9gkDhh7MV535edefCTZw')
# examples = [
#     {'question': 'Dog is',
#      'answer': 'Pet animal'
#      },
#     {'question': 'Cat is',
#      'answer': 'Pet animal'
#      },
#     {'question': 'Tiger is',
#      'answer': 'Wild animal'
#      },
#     {'question': 'Lion is',
#      'answer': 'Wild animal'
#      }
# ]
#
# prompt = """
# My question: {question}
# {answer}
# """
# example_prompt = PromptTemplate.from_template(prompt)
#
# # print(example_prompt.format(**examples[0]))
#
# few_shot_prompt = FewShotPromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
#     suffix='Question: {input}',
#     input_variables=['input']
# )
# # Sử dụng few_shot_prompt với đầu vào
# input_example = {'input': 'Elephant is'}
# formatted_prompt = few_shot_prompt.format(**input_example)
#
#
# model = GoogleGenerativeAI(model="gemini-pro")
# chain = (few_shot_prompt
#          |model
#          |StrOutputParser()
# )
#
# res = chain.invoke('Elephant is')
# print(res)






########################
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# Các ví dụ
# examples = [
#     {
#         "question": """
#         {
#             "Tiền mặt và tiền gửi": {
#                 "Tiền gửi thông thường": {
#                     "Ngân hàng A": {
#                         "value": [
#                             1,
#                             2,
#                             3,
#                             4
#                         ]
#                     },
#                     "value": [
#                         5,
#                         6,
#                         7,
#                         8
#                     ]
#                 },
#                 "value": [
#                     "A",
#                     "B",
#                     "C",
#                     "D"
#                 ]
#             }
#         }
#         """,
#         "answer":
#         """
#         {
#             "Tiền mặt và tiền gửi": {
#                 "Tiền gửi thông thường": {
#                     "value": [
#                         8
#                     ]
#                 },
#                 "value": [
#                     "D"
#                 ]
#             }
#         }
#         """
#     }
# ]
examples = [{"question": {"Tiền mặt và tiền gửi": {"Tiền gửi thông thường": {"Ngân hàng A": {"value": [1,2,3,4]},"value": [5,6,7,8]},"value": ["A","B","C","D"]}},
        "answer":{"Tiền mặt và tiền gửi": {"Tiền gửi thông thường": {"value": [8]},"value": ["D"]}}}
]

# Định nghĩa prompt mẫu cho mỗi ví dụ
prompt = """
My question: {question}
{answer}
"""
example_prompt = PromptTemplate.from_template(prompt)

# Kiểm tra định dạng ví dụ
# print(example_prompt.format(**examples[0]))

# Tạo FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix='Question: {input}',
    input_variables=['input']
)

# # Sử dụng few_shot_prompt với đầu vào
# input_example = {'input': """
# {
#     "Tiền mặt và tiền gửi": {
#         "Tiền gửi thông thường": {
#             "Ngân hàng A": {
#                 "value": [
#                     1,
#                     2,
#                     3,
#                     4
#                 ]
#             },
#             "value": [
#                 5,
#                 6,
#                 7,
#                 8
#             ]
#         },
#         "value": [
#             "A",
#             "B",
#             "C",
#             "D"
#         ]
#     }
# }
# """}
# formatted_prompt = few_shot_prompt.format(**input_example)
# print(formatted_prompt)

# Tạo mô hình Google Generative AI
model = GoogleGenerativeAI(model="gemini-pro")
# model = GoogleGenerativeAI(model="gemini-1.5-pro-001")

# Tạo chuỗi
chain = (few_shot_prompt | model | StrOutputParser())

# Đặt câu hỏi
# ques = """
# {
#     "Tiền mặt và tiền gửi": {
#         "Tiền gửi thông thường": {
#             "Ngân hàng A": {
#                 "value": [
#                     1,
#                     2,
#                     3,
#                     4
#                 ]
#             },
#             "value": [
#                 5,
#                 6,
#                 7,
#                 8
#             ]
#         },
#         "value": [
#             "A",
#             "B",
#             "C",
#             "D"
#         ]
#     }
# }
# """

ques2 = {"Tiền mặt và tiền gửi": {"Tiền gửi thông thường": {"Ngân hàng A": {"value": [1,2,3,4]},"value": [5,6,7,8]},"value": ["A","B","C","D"]}}
res = chain.invoke({'input': ques2})
print(res)
