import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -------------------------------
# STEP 1: API Key Setup
# -------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="🥗 Diet Chatbot", page_icon="🥗")
st.title("🥗 Diet Recommendation Chatbot")

if not API_KEY:
    st.warning("⚠️ No API key found. Please set GOOGLE_API_KEY in your environment.")
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY

    # -------------------------------
    # STEP 2: Load and process dataset
    # -------------------------------
    df = pd.read_csv("All_Diets.csv")
    df = df.dropna(subset=["Recipe_name", "Diet_type", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)"])

    # Optional: limit for speed during dev
    # df = df.sample(50)

    # Convert into structured text
    text_data = "\n\n".join(
        f"Recipe: {row['Recipe_name']}\nDiet Type: {row['Diet_type']}\nCuisine: {row['Cuisine_type']}\n"
        f"Protein: {row['Protein(g)']}g\nCarbs: {row['Carbs(g)']}g\nFat: {row['Fat(g)']}g"
        for _, row in df.iterrows()
    )

    # -------------------------------
    # STEP 3: Split into chunks
    # -------------------------------
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([text_data])

    # -------------------------------
    # STEP 4: Embeddings & Vector DB
    # -------------------------------
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embedding_model, persist_directory="db/")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # -------------------------------
    # STEP 5: Custom Prompt
    # -------------------------------
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful and professional diet and nutrition assistant. Use the provided information to answer the user's question as clearly and accurately as possible.

If the answer is not found in the context, say "I'm not sure about that from the given information."

Examples:
Q: Suggest a French paleo recipe with moderate fat.
A: Paleo Chicken Provençal — Fat: 42g, Cuisine: French, Diet: Paleo

Q: I want something from the keto diet that is high in protein.
A: Keto Beef Bowl — Protein: 95g, Diet: Keto

Always recommend recipes strictly from the given context. Do not invent new recipes.

Interpret patient health conditions as follows when matching recipes:
- "Diabetes" or "blood sugar control" → Suggest low-carb options
- "Hypertension" or "high blood pressure" → Suggest DASH diet options
- "Cholesterol issues" → Prefer DASH or Paleo diet recipes
- "Weight loss" → Prefer Keto or low-carb recipes
- "Heart health" → Prefer DASH or Mediterranean recipes

Examples:
Q: I have diabetes and want something Indian.
A: Blackened Catfish — Cuisine: Indian, Diet: Low-Carb

Q: I am on the DASH diet for high blood pressure. What Mediterranean recipe can I try?
A: Mediterranean Chickpea Salad — Cuisine: Mediterranean, Diet: DASH
---------------------
Context:
{context}

---------------------
User Question: {question}

Answer:
"""
    )

    # -------------------------------
    # STEP 6: Gemini LLM
    # -------------------------------
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=API_KEY,
        temperature=0.4,
        max_output_tokens=512,
    )

    # -------------------------------
    # STEP 7: QA Chain
    # -------------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # -------------------------------
    # STEP 8: Streamlit Chat UI
    # -------------------------------
    st.subheader("💬 Ask me about diets & recipes")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Your question:")

    if query:
        with st.spinner("🤔 Thinking..."):
            try:
                response = qa_chain.run(query)
                st.session_state.chat_history.append(("You", query))
                st.session_state.chat_history.append(("Bot", response.strip("— ").strip()))
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"🧑 **You:** {message}")
        else:
            st.markdown(f"🤖 **Bot:** {message}")