# diet-recommendation-chatbot
Diet Recommendation Chatbot powered by LangChain, Hugging Face embeddings, ChromaDB, and Google Gemini.   Built to assist patients with dietary needs by providing tailored recipe suggestions through a Streamlit interface.  


The chatbot understands natural language dietary queries and retrieves recipes from a structured dataset, providing **personalized, disease-oriented suggestions** through a **Streamlit web interface**.  

---

## 🚀 Features  
- Understands natural language questions about diets and meals  
- Retrieves and suggests recipes based on dietary needs and health conditions  
- Uses semantic embeddings for better search & retrieval  
- Provides a simple, user-friendly web interface  
- Focused on helping patients with **disease-specific meal planning**  

---

## 🛠️ Tech Stack  
This project integrates multiple modern AI technologies:  

- **Python** – core programming language  
- **LangChain** – framework for orchestration of LLMs, embeddings & retrievers  
- **Hugging Face Sentence Transformers (all-MiniLM-L6-v2)** – semantic embeddings for recipes  
- **ChromaDB** – vector database for storage & retrieval of embeddings  
- **Google Gemini 1.5 Flash (via API)** – LLM for response generation  
- **Streamlit** – interactive frontend for the chatbot  
- **CSV Dataset (All_Diets.csv)** – structured recipe dataset with nutritional values  

---

## 📂 Project Structure  







---

## ⚙️ Setup Instructions  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/diet-recommendation-chatbot.git
   cd diet-recommendation-chatbot

2. Install dependencies:
   ```
    pip install -r requirements.txt

3.Add your Google API Key in the .env file:

     GOOGLE_API_KEY=your_api_key_here
     
 4. Run the Streamlit app:
    ```
    streamlit run chatbot_final2.py


Dataset:

The dataset All_Diets.csv includes:
	•	Recipe Name
	•	Diet Type (Keto, Paleo, DASH, etc.)
	•	Cuisine Type (Indian, French, Mediterranean, etc.)
	•	Nutritional Info (Protein, Carbs, Fat in grams)

This data is pre-processed into structured text and then embedded using Hugging Face for semantic search.

⸻

 Example Queries:
 
	•	“Suggest me an Indian dish suitable for a diabetic patient.”
	•	“What Mediterranean recipes are good for someone with high blood pressure?”
	•	“I want a keto meal option for weight loss.”

⸻


🔒Note
	•	Keep your .env file private (never commit your API keys).
	•	The dataset can be extended with more recipes and nutritional details.
