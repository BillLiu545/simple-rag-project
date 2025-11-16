# simple-rag-project
In this repository is a Vue.js project that demonstrates a RAG procedure (extracts text from document to analyse)

# How does it work?
On the main Vue page, the user uploads a PDF file. To let AI analyze the text, the user can click on "Send Query." Once clicked, the AI model (Qwen 3B) ingests the document to analyze the content inside. When counting words, the number is displayed in a pie chart, in which it is displayed after being exported as a JSON file.

# Remark
I wasn't able to upload the entire Vue project especially considering the backend Python files and extra Vue config files would already make it too much. OpenAI API key sections were replaced with placeholders.
