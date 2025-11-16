<template>
    <div class="container">
        <h1>RAG Application</h1>
        <p>This is a simple RAG application built with Vue.js.</p>

        <!--Upload a PDF File-->
        <div class="upload">
            <h2>Upload a PDF File</h2>
            <input type="file" @change="uploadPdf" />
        </div>

        <!--Display Uploaded PDF Content-->
        <div class="pdf-content" v-if="pdfContent">
            <h2>PDF Content</h2>
            <p>{{ pdfContent }}</p>
        </div>

        <!--Query-->
        <div class="query">
            <h2>Chat Interface</h2>
            <input v-model = "query" placeholder="Enter some text to query" />
            <button @click="sendQuery">Send Query</button>
        </div>

        <!--Display Response-->
        <div class="response" v-if="response">
            <p class="response_text">{{ response }}</p>
        </div>
    </div>
</template>

<script>
import axios from 'axios';
export default {
    data() {
        return {
            pdfContent: '',
            query: '',
            response: ''
        }
    },
    methods: {
        async uploadPdf(event) {
            const file = event.target.files[0];
            let formData = new FormData();
            formData.append('file', file);

            await axios.post("http://localhost:8000/upload_pdf", formData, {
                headers: {'Content-Type': 'multipart/form-data' ,}
            });
            alert("PDF file uploaded successfully!");
            
        },
        async sendQuery() {
            if (!this.query) {
                alert("Please enter a query.");
                return;
            }
            let formData = new FormData();
            formData.append('query', this.query);
            const res = await axios.post("http://localhost:8000/query", formData)
            this.response = res.data.response;
        }

    }
}
</script>

<style>
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.upload, .query, .response {
    margin-bottom: 20px;
}
.upload input, .query input {
    width: 100%;
    padding: 10px;
    margin-top: 10px;
}
.query button {
    padding: 10px 20px;
    margin-top: 10px;
}
.response_text {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
}
.pdf-content {
    background-color: #f9f9f9;
    padding: 10px;
    border-radius: 5px;
}
.pdf-content p {
    white-space: pre-wrap; /* Preserve whitespace */
    word-break: break-word; /* Break long words */
}
</style>