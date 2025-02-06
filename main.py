from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

model = ChatOllama(
    model="llama3.2",
    temperature=0,
)
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""


def main():
    text ="""
    
2
BUILDING BLOCKS STUDENT HANDOUT 1 of 1
Sample credit card statement
 BUILDING BLOCKS STUDENT HANDOUT
Sample credit card statement
Name: Susan Doe
Address: 1234 Main Street, Anytown, USA
Account Number: 12345-67-8907
For Lost or Stolen Card, Call:
1-800-XXX-XXXX
Payment Information
Date: 12/30/XX
Payment Due Date: 1/23/XX
New Balance: $1392.71
Minimum Payment: $25
Account Summary Payment Information
Previous Balance $482.42 New Balance $1,392.71
Payment, Credits -$350.42 Payment Due Date 1/23/XX
Purchases $1,258.56 Minimum Payment Due $25
Cash Advances $0
Balance Transfers $0
Fees Charged $0
Interest Charged $2.15
New Balance $1,392.71
Opening/Closing Date 11/27/XX – 12/26/XX
Credit Access Line $12,000
Available Credit $10,607.29
Cash Access Line $2,000
Available for Cash $2,000
Past Due Amount $0
Balance Over the Credit Access Line $0
Finance Charge Summary Purchases Advances
Periodic Rate 1.65% 0.54%
Annual Percentage Rate (APR) 19.80% 6.48%

 
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    chain = (
            RunnablePassthrough.assign(context=lambda input: input["context"])
            | rag_prompt
            | model
            | StrOutputParser()
    )

    # question = "What are the approaches to Task Decomposition?"
    question = "tell me the account name and address?"

    # Run
    output = chain.invoke({"context": text, "question": question})
    # output1 = chain.invoke({"context": docs, "question": 'what is the source'})

    print(output)


if __name__ == '__main__':
    print('running')
    main()
