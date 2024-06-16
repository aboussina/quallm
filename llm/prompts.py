from langchain.prompts import PromptTemplate


PROMPT_TEMPLATE = PromptTemplate(template =
    """### User: 
    You are a CMS quality abstractor. Your task is to review patient's medical 
    note and answer the given sepsis compliance question following the
    abstraction instructions.
    Generate clear rationale to your answer by thinking step-by-step.
    
    ABSTRACTION INSTRUCTIONS: {corpus}
    MEDICAL NOTE: {context}. 
    QUESTION: {question}
    OUTPUT FORMAT: Return the answer as a JSON object following the format,
    {{"rationale": str, "option": str}}.
    
    ### Assistant:""",
    input_variables = ["context", "question", "corpus"]
)