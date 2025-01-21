# langchain_utils.py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# A simple prompt template or chain for demonstration
prompt_template = """
You are a specialized model that outputs JSON schemas for the AdvancedFormNode.
User requirement: {user_requirement}
Return the valid JSON schema. 
"""


def get_llm_chain():
    # Read fine-tuned model name
    with open("fine_tuned_model_name.txt", "r") as f:
        fine_tuned_model = f.read().strip()

    # Setup ChatOpenAI to use your fine-tuned model
    llm = ChatOpenAI(
        temperature=0.0,
        model=fine_tuned_model,  # Changed from model_name to model
        streaming=False,
    )

    # Create the chain using the new LCEL syntax
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["user_requirement"]
    )

    chain = (
        {"user_requirement": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    return chain
