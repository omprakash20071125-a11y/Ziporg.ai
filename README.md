<div align="center">

# 🚀 ziporg.ai

### Prompt In. Project Out.

AI-powered software generation platform that turns a single natural-language prompt into a complete, downloadable, production-ready codebase.
<img width="1450" height="832" alt="image" src="https://github.com/user-attachments/assets/84395a0f-4f62-4d8a-9f84-36db35e5244f" />
<img width="1450" height="675" alt="image" src="https://github.com/user-attachments/assets/0eb19f8b-1a70-40ef-8653-b9687bb93e21" />



**[🔗 Live Demo](https://ziporg-ai.vercel.app)**

</div>

---

## 📖 Overview

Most AI coding tools generate isolated snippets that still need to be wired together by hand. **ziporg.ai** takes a different approach: it plans an entire project's architecture, generates every required file with awareness of the full codebase, packages the result into a ZIP archive, and hands back something you can actually run.

Give it a prompt like *"Build me a personal portfolio website with HTML, CSS and JavaScript"* and get back a structured, ready-to-run project — not a fragment.

---

## ✨ Features

| | |
|---|---|
| 🧠 | AI-powered project planning |
| 📂 | Multi-file, context-aware code generation |
| 📦 | Automatic ZIP packaging |
| ⚡ | FastAPI backend |
| 🕸️ | LangGraph workflow orchestration |
| 🤖 | Google Gemini integration |
| 🌐 | Simple, lightweight web interface |
| 📥 | One-click project download |
| 🧩 | Prompt refinement for vague/underspecified requests |

---

## 🏗️ Architecture

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

**Workflow graph (LangGraph):**

```
START → Planner Node → Code Generator Node → END
```

---

## ⚙️ Tech Stack

**Backend**
- Python
- FastAPI
- LangGraph
- LangChain
- Google Gemini 2.5 Flash
- Pydantic

**Frontend**
- HTML
- CSS
- JavaScript

**AI**
- Google Gemini API

---

## 📁 Project Structure

```
ziporg-ai/
├── backend/
│   ├── .gitignore
│   ├── main.py
│   ├── phase1.py
│   ├── phase2.py
│   ├── phase2_planner.py
│   ├── requirements.txt
│
├── frontend/
│   └── index.html
│
└── README.md
```

---

## 🚀 How It Works

1. **User enters a prompt**
   ```
   Build me a personal portfolio website with HTML, CSS and JavaScript.
   ```

2. **AI Planner** analyzes the request and determines:
   - Project type
   - Required files
   - Programming language(s)
   - Dependencies
   - Generation order

3. **Code Generation** — each file is generated individually while the model stays aware of the overall project structure, producing consistent, connected output instead of disjointed snippets.

4. **Packaging** — the finished project is automatically zipped.

5. **Download** — the user receives a ready-to-run `project.zip`.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/omprakash20071125-a11y/ziporg-ai.git
cd ziporg-ai

# Create a virtual environment
python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file inside `backend/` (this file is git-ignored and should never be committed):

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## ▶️ Running the Backend

```bash
cd backend
uvicorn main:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 🌐 Running the Frontend

Open `frontend/index.html` with a local server (e.g. VS Code Live Server), or use the hosted version at the [live demo](https://ziporg-ai.vercel.app).

---

## 📡 API Reference

### Generate Project

`POST /generate`

**Request**
```json
{
  "prompt": "Build me a calculator application."
}
```

**Response**

Returns a downloadable `project.zip` containing the generated codebase.

---

## 🗺️ Roadmap

**Phase 1 — Complete**
- Prompt-to-project generation
- Multi-file, context-aware code output
- ZIP packaging & download
- FastAPI + LangGraph pipeline

**Phase 2 — Complete**
- Prompt query optimization — refine and clarify vague/underspecified prompts with the user before generation
- Support for cloning/replicating real-world web app structures and UX patterns from a description
- Streaming generation with live progress updates
- Project templates and reusable scaffolds

**Phase 3 — In Progress**
- Image analysis — accept an uploaded image (e.g. a design mockup, screenshot, or reference UI) and let the AI planner interpret it as part of the generation input
- Image-to-code assistance — pull layout, color, and component cues from an analyzed image to inform the generated frontend
- Improved web-page content reading — stronger extraction and understanding of an existing site's structure, text, and layout when a URL or page is provided as reference
- Reference-aware regeneration — use the read page content to keep generated pages consistent with the source's tone, sections, and information architecture
- Basic accessibility pass on generated pages (alt text for images, semantic HTML checks)
- Prompt history within a session, so follow-up prompts can refine the same project instead of starting over

**Future**
- User authentication
- Project history & versioning
- Cloud storage for generated projects
- Stripe subscription / usage tiers
- Team workspaces
- Custom/pluggable AI models
- Docker export
- Git repository export
- One-click deployment

---

## 🤝 Contributing

Contributions are welcome. If you'd like to improve ziporg.ai, please open an issue or submit a pull request.

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Om Prakash Gupta**
AI Engineer · Full-Stack AI Developer

[GitHub](https://github.com/omprakash20071125-a11y)

---

## ⭐ Support

If this project helped you, consider giving it a star on GitHub — it helps others discover it and supports continued development.

---

<div align="center">

### 💡 Vision

The long-term vision of **ziporg.ai** is to become an AI software engineer capable of turning ideas into production-ready applications — not isolated snippets, but complete, structured, maintainable projects developers can run, extend, and deploy immediately.

**Prompt In. Project Out.**

</div>
