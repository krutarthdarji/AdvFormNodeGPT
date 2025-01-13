# langchain_utils.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    # Make sure your openai.api_key is set appropriately
    llm = ChatOpenAI(temperature=0.0, model_name=fine_tuned_model)
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template, input_variables=["user_requirement"]
        ),
    )
    return chain
