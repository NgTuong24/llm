from langchain.retrievers import MultiVectorRetriever
from langchain_text_splitters import CharacterTextSplitter

with open(r"F:\CMC\CMC_Study\Code\data\state_of_the_union.txt", encoding='utf-8') as f:
    text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=0, separator="", strip_whitespace=False)

text = text_splitter.create_documents([text])

print(text)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

retriever.metadata.items()