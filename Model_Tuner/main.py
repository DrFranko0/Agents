import streamlit as st
from huggingface_hub import list_models
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from bs4 import BeautifulSoup
import requests
import pandas as pd

llm = OllamaLLM(model="llama3.1")

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a model selection assistant. Based on the query: '{query}', "
        "analyze and summarize the available fine-tuned Llama models. "
        "Provide a detailed comparison including instruction-following capabilities, "
        "fine-tuning datasets, parameters, and other distinguishing features."
    ),
)

chain = RunnableSequence(prompt_template | llm)

def fetch_models_from_hf(query):
    models = list_models(search=query)
    return [{"Model ID": model.modelId, "Tags": model.tags, "Card": model.cardData} for model in models]


def fetch_web_resources(query):
    search_url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        search_results = []
        for a in soup.find_all('a', class_='result__a', href=True):
            search_results.append({
                'title': a.get_text(),
                'url': a['href']
            })
        
        return search_results
    else:
        return []



st.title("Llama Finder - Fine-Tuned Model Selector")
st.markdown(
    """
    Small ML models can outperform larger ones when fine-tuned for specific tasks. 
    Use this app to find the best fine-tuned Llama models or alternatives tailored to your needs.
    """
)

user_prompt = st.text_input("Describe your specific use case:", "")

if user_prompt:
    st.write("### Fetching Fine-Tuned Models from HuggingFace...")
    with st.spinner("Searching HuggingFace models..."):
        hf_results = fetch_models_from_hf(user_prompt)
    if not hf_results:
        st.warning("No fine-tuned models found on HuggingFace. Try using a broader query or adjust filters.")
        st.stop()
    if hf_results:
        st.write("#### Results from HuggingFace:")
        hf_df = pd.DataFrame(hf_results)
        st.dataframe(hf_df)
    else:
        st.warning("No models found on HuggingFace for your query.")

    st.write("### About...")
    with st.spinner("Fetching web resources..."):
        web_data = fetch_web_resources(user_prompt)
    if web_data:
        st.write("#### Example Resources:")
        for result in web_data:
            st.markdown(f"**[{result['title']}]({result['url']})**")
    else:
        st.warning("No web resources found.")

    st.write("### Advanced Analysis...")
    with st.spinner("Analyzing models using LangChain's OllamaLLM..."):
        try:
            reasoning_result = chain.invoke({"query": user_prompt})
            st.write("#### Analysis Result from OllamaLLM:")
            st.text(reasoning_result)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.write("### Generate Comparison Table")
    if hf_results:
        st.write("The comparison table summarizes models fetched from HuggingFace:")
        comparison_data = [
            {
                "Model ID": model["Model ID"],
                "Fine-tuned for Instruction Following": "Yes" if "instruction" in (model["Tags"] or []) else "No",
                "Parameters": len(model["Card"]) if model["Card"] else "Unknown",
                "Datasets Used": ", ".join(model["Card"].get("datasets", [])) if model["Card"] else "N/A",
            }
            for model in hf_results
        ]
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
    else:
        st.warning("No data available to create a comparison table.")

else:
    st.info("Enter a use case to start the search.")
