# 🚀 ziporg.ai

### Prompt In. Project Out.

Generate complete, production-ready software projects from a single prompt.

**ziporg.ai** is an AI-powered software generation platform that transforms natural language prompts into complete, downloadable project codebases. Instead of generating isolated code snippets, ziporg.ai plans the project architecture, generates every required source file, packages everything into a ZIP archive, and delivers a ready-to-run application.

---

## ✨ Features

* 🧠 AI-powered project planning
* 📂 Multi-file code generation
* 📦 Automatic ZIP packaging
* ⚡ FastAPI backend
* 🕸️ LangGraph workflow orchestration
* 🤖 Google Gemini integration
* 🌐 Simple web interface
* 📥 One-click project download

---

# 🏗️ Architecture

```
                User Prompt
                     │
                     ▼
            Prompt Planner (LLM)
                     │
                     ▼
         Project File Architecture
                     │
                     ▼
        Sequential File Generation
                     │
                     ▼
          Complete Project Folder
                     │
                     ▼
             ZIP Packaging
                     │
                     ▼
           Download project.zip
```

---

# ⚙️ Tech Stack

## Backend

* Python
* FastAPI
* LangGraph
* LangChain
* Google Gemini 2.5 Flash
* Pydantic

## Frontend

* HTML
* CSS
* JavaScript

## AI

* Google Gemini API

---

# 📁 Project Structure

```
ziporg-ai/

├── backend/
│   ├── main.py
│   ├── phase1.py
│   ├── requirements.txt
│   ├── .env
│   └── ...
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
└── README.md
```

---

# 🚀 How It Works

### 1. User enters a prompt

Example:

```
Build me a personal portfolio website with HTML, CSS and JavaScript.
```

---

### 2. AI Planner

The planner analyzes the request and determines:

* Project type
* Required files
* Programming language
* Dependencies
* Generation order

---

### 3. Code Generation

Each file is generated individually while maintaining awareness of the entire project structure.

This produces consistent multi-file applications instead of disconnected code snippets.

---

### 4. Packaging

The generated project is automatically packaged into a ZIP archive.

---

### 5. Download

The user downloads a ready-to-run project.

---

# 📦 Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/ziporg-ai.git
```

Move into the project.

```bash
cd ziporg-ai
```

Create a virtual environment.

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside the backend folder.

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

# ▶️ Running the Backend

```bash
uvicorn main:app --reload
```

The API will start at

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# 🌐 Running the Frontend

Open the frontend using a local web server such as VS Code Live Server.

Alternatively, after deployment, access it through your hosted URL.

---

# 📡 API Endpoint

## Generate Project

**POST**

```
/generate
```

Request

```json
{
  "prompt": "Build me a calculator application."
}
```

Response

```
project.zip
```

---

# 🧠 Workflow

```
START
   │
   ▼
Planner Node
   │
   ▼
Code Generator Node
   │
   ▼
ZIP Builder
   │
   ▼
END
```

---

# 📷 Screenshots

> Add screenshots of your application here.

Suggested screenshots:

* Landing Page
* Prompt Input
* Generation Progress
* Download Button
* Generated Project

---

# 🔮 Future Improvements

* User Authentication
* Project History
* Cloud Storage
* Stripe Subscription
* Team Workspaces
* Custom AI Models
* Project Templates
* Streaming Generation
* Docker Export
* Git Repository Export
* One-click Deployment
* Version History

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve ziporg.ai, feel free to open an issue or submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps the project reach more developers and motivates future development.

---

# 👨‍💻 Author

**Om Prakash Gupta**

AI Engineer | Full-Stack AI Developer

GitHub:
https://github.com/omprakash20071125-a11y

---

## 💡 Vision

The long-term vision of **ziporg.ai** is to become an AI software engineer capable of transforming ideas into production-ready applications.

Rather than generating isolated code snippets, the platform focuses on producing complete, structured, maintainable software projects that developers can immediately run, extend, and deploy.

**Prompt In. Project Out.**
