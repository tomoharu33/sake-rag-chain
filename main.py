import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

llm = ChatOpenAI(model="gpt-4o-mini")

# テキストのみの場合
docs = [
    Document(page_content="山廃・生酛は、醸造用乳酸を添加せずに乳酸菌を増殖させる伝統的な醸造方法である。1910年に速醸酛が開発されたことで、手間のかかる生酛や山廃は少数派になり、現状のシェアは速醸90%、山廃9%、生酛1%である。しかし、豊かで複雑みのある香味を求めて、生酛系酒母に取り組む蔵が目立ってきている。和食だけでなく、中華や洋食との相性も良いため、幅広いペアリングの対象として好まれる可能性がある。"),
    Document(page_content="セルレニン耐性酵母は、リンゴ様の香り・カプロン酸エチルを多く生成する酵母で、「香り酵母」とも呼ばれる。1990年代中期に全国に広まった。代表的な酵母として、アルプス酵母やきょうかい1801号が挙げられる。香り酵母は全国各地で開発されており、地域色を打ち出したものも増えている。フルーティーな香りは日本酒初心者にも好まれ、新たな消費者が日本酒を飲むきっかけとなる可能性がある。"),
    Document(page_content="村米制度は、酒造家と農家が直接契約して酒米を栽培する制度であり、「山田錦」の故郷である兵庫県では明治20年代から行われていた。農家は酒造家が好む酒米を生産するために品質向上を図る。テロワールによる集落ごとの格付けも行われ、集落内外での競争が活発化した。現在は「特A-a地区」と「特A-b地区」に分けられ、「特A-a地区」は吉川町、口吉川町、東条、社の91集落で構成されている。"),
    Document(page_content="美山錦は、1978年に長野県農事試験場で「たかね錦」の種籾にγ線を照射して生み出された突然変異種の酒米である。醸造用玄米の中では「山田錦」「五百万石」に次ぎ生産量第３位を誇る。大粒で心白発現率が良いため、高精白が可能である。また耐冷性があるため、長野のほか東北地方が主な産地となっている。「亀ノ尾」など歴史ある品種を先祖にもち、「出羽燦々」「越の雫」「秋の精」など他県が開発した品種の親株でもある。"),
    Document(page_content="奈良県は清酒発祥の地とされている。日本最古の神社・大神神社は酒造りの神で、杉玉の発祥の地でもある。奈良時代には造酒司が設けられ、酒造りの中心地となった。室町時代には酒母製法の一つである「菩提酛」が菩提山正暦寺で生み出された。菩提酛は「そやし水」と呼ばれる乳酸酸性水を使用して酒母を作る製法で、近年奈良県内の蔵元が再現している。酒米は自県産より他県からの移入が多いが、「露葉風」の生産量は日本一である。"),
]

# webページの場合
# loader_note = WebBaseLoader(
#     web_paths=(
#         "https://note.com/sake_diploma_23/n/n0e5ff61a2c6b",
#         "https://note.com/sake_diploma_23/n/n988b0381d910",
#     ),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("note-common-styles__textnote-body")
#         )
#     ),
# )

# loader_wiki = WebBaseLoader(
#     web_paths=("https://ja.wikipedia.org/wiki/%E5%9D%82%E5%8F%A3%E8%AC%B9%E4%B8%80%E9%83%8E",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(class_="mw-body")
#     ),
# )

# docs = loader_note.load() + loader_wiki.load()
# docs = loader_wiki.load() + loader_note.load()

# テキストを追加する場合
# docs.extend([
#     Document(page_content="山廃・生酛は、醸造用乳酸を添加せずに乳酸菌を増殖させる伝統的な醸造方法である。1910年に速醸酛が開発されたことで、手間のかかる生酛や山廃は少数派になり、現状のシェアは速醸90%、山廃9%、生酛1%である。しかし、豊かで複雑みのある香味を求めて、生酛系酒母に取り組む蔵が目立ってきている。和食だけでなく、中華や洋食との相性も良いため、幅広いペアリングの対象として好まれる可能性がある。"),
#     Document(page_content="セルレニン耐性酵母は、リンゴ様の香り・カプロン酸エチルを多く生成する酵母で、「香り酵母」とも呼ばれる。1990年代中期に全国に広まった。代表的な酵母として、アルプス酵母やきょうかい1801号が挙げられる。香り酵母は全国各地で開発されており、地域色を打ち出したものも増えている。フルーティーな香りは日本酒初心者にも好まれ、新たな消費者が日本酒を飲むきっかけとなる可能性がある。"),
#     Document(page_content="村米制度は、酒造家と農家が直接契約して酒米を栽培する制度であり、「山田錦」の故郷である兵庫県では明治20年代から行われていた。農家は酒造家が好む酒米を生産するために品質向上を図る。テロワールによる集落ごとの格付けも行われ、集落内外での競争が活発化した。現在は「特A-a地区」と「特A-b地区」に分けられ、「特A-a地区」は吉川町、口吉川町、東条、社の91集落で構成されている。"),
#     Document(page_content="美山錦は、1978年に長野県農事試験場で「たかね錦」の種籾にγ線を照射して生み出された突然変異種の酒米である。醸造用玄米の中では「山田錦」「五百万石」に次ぎ生産量第３位を誇る。大粒で心白発現率が良いため、高精白が可能である。また耐冷性があるため、長野のほか東北地方が主な産地となっている。「亀ノ尾」など歴史ある品種を先祖にもち、「出羽燦々」「越の雫」「秋の精」など他県が開発した品種の親株でもある。"),
#     Document(page_content="奈良県は清酒発祥の地とされている。日本最古の神社・大神神社は酒造りの神で、杉玉の発祥の地でもある。奈良時代には造酒司が設けられ、酒造りの中心地となった。室町時代には酒母製法の一つである「菩提酛」が菩提山正暦寺で生み出された。菩提酛は「そやし水」と呼ばれる乳酸酸性水を使用して酒母を作る製法で、近年奈良県内の蔵元が再現している。酒米は自県産より他県からの移入が多いが、「露葉風」の生産量は日本一である。"),
#     ])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# print(rag_chain.invoke("縄文時代の晩期に、すでに水稲は始まっていましたか？"))
print(rag_chain.invoke("山廃・生酛の現状と将来の展望について200字以内で述べよ。"))
# print(rag_chain.invoke("坂口謹一郎と日本酒について、450~500文字程度で要約してください。"))

# cleanup
vectorstore.delete_collection()