

### **RAG Chatbot with Streamlit & Groq API**  

This is a **Retrieval-Augmented Generation (RAG) chatbot** built using **Streamlit** and **Groq API**, designed to provide intelligent answers based on uploaded PDF documents.  

---

## **🚀 Features**  
✅ **Uses LLaMA 3 (llama3-8b-8192) for responses**  
✅ **Retrieves context from PDFs using embeddings & vector search**  
✅ **Streamlit-based chat UI for user-friendly interaction**  
✅ **Fast & efficient retrieval using HuggingFaceEmbeddings**  

---

## **📌 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/vaishnavidesai09/rag-chatbot-streamlit.git
cd rag-chatbot-streamlit
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**  
```sh
python -m venv venv  
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**  
Create a `.env` file in the root directory and add:  
```
GROQ_API_KEY=your_groq_api_key_here
```

### **5️⃣ Run the Chatbot**  
```sh
streamlit run main.py
```

---

## **📦 Dependencies**  
- `streamlit` – UI for the chatbot  
- `langchain-community` – LLM and vector store support  
- `sentence-transformers` – Embeddings for document retrieval  
- `pypdf` – PDF document parsing  
- `dotenv` – Loads environment variables  

To install all dependencies manually:  
```sh
pip install streamlit langchain-community sentence-transformers pypdf python-dotenv
```

---

## **💡 How It Works**  
1. The chatbot reads a PDF (`main.pdf`).  
2. Splits the document into **chunks** using `RecursiveCharacterTextSplitter`.  
3. Embeds the chunks using **HuggingFaceEmbeddings**.  
4. Stores the embeddings in a **vector database**.  
5. Uses Groq’s `llama3-8b-8192` to generate responses.  
6. Retrieves relevant PDF content for context before answering.  

---

## **🎯 Future Improvements**  
- ✅ Support for multiple PDFs  
- ✅ Improve chunking strategy for better retrieval  
- ✅ Deploy on a cloud platform like **Hugging Face Spaces**  

---

## **🤝 Contributing**  
Feel free to open issues or submit pull requests!  

---

## **📜 License**  
MIT License  

