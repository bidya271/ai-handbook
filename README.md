# AI Mastery: Field Guide & Accelerated Playbook 🚀

A mobile-first, chapter-based interactive course reader and playbook engineered to take engineers from prompt users to production AI systems architects.

---

## ⚡ What's New & Upgraded (What Was Missing & Fixed)

1. **Integrated 5-Layer Core Competency Stack:**
   - Data & Math Foundations (PyTorch, NumPy, Linear Algebra)
   - Model Architecture & PEFT (Attention, KV Caching, LoRA, Quantization)
   - Agentic Systems & Orchestration (State Graphs, LangGraph, Pydantic schemas)
   - AI Engineering & Infrastructure (vLLM, PagedAttention, Vector DBs, Tracing)
   - Product, Risk & Governance (RAG Triad, LLM-as-a-Judge, Promptfoo, Ragas)

2. **The Hyper-Accelerated Fast-Track Roadmap:**
   - Compresses the traditional **9-month timeline into an aggressive 8-Week / 60-Day Sprint**.
   - Structured with the **30-60-30 Daily Deliberate Practice Protocol** (30 min paper/architecture deconstruction, 60 min line-by-line coding, 30 min documentation/benchmarking).
   - Milestone checklist with verifiable exit criteria for each sprint.

3. **Production Math Rendering (KaTeX):**
   - LaTeX equations (Scaled Dot-Product Attention, LoRA rank decomposition, RRF scoring, KV cache calculations) rendered crisply across mobile and desktop.

4. **True Offline PWA Support (`sw.js` & `manifest.json`):**
   - Background service worker caching for offline access during commutes or flights.
   - Installable on iOS (Share → "Add to Home Screen") and Android (3 dots → "Install App").

5. **Enhanced Mobile UX:**
   - One-tap code copy buttons on every code block.
   - Real-time chapter search and filter inside the drawer.
   - Dynamic font resizing (15px to 21px).
   - Seamless Dark and Light reading themes.
   - LocalStorage progress persistence (remembers completed chapters and reading position).

---

## 📂 Repository File Structure

```text
ai-handbook/
├── index.html       # Full mobile-first reader with KaTeX, Lucide, and 9 course chapters
├── manifest.json    # Progressive Web App (PWA) manifest configuration
├── package.json     # Project definition and static serve scripts
├── sw.js            # Offline service worker cache engine
├── vercel.json      # Production Vercel headers and clean routing
└── README.md        # Documentation and deployment guide
```

---

## 🚀 Instant Deployment to Vercel

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: complete mobile ai handbook reader with accelerated roadmap"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ai-handbook.git
   git push -u origin main
   ```

2. Open [vercel.com](https://vercel.com) and log in.
3. Click **Add New...** → **Project**.
4. Import the `ai-handbook` repository.
5. Leave all settings at defaults and click **Deploy**.
6. Your handbook will be live on a production URL (e.g. `ai-handbook.vercel.app`) within seconds!

---

## 💻 Local Development

Run locally with any static web server:

```bash
# Using npx serve
npx serve .

# Or using Python's built-in server
python3 -m http.server 3000
```

Open `http://localhost:3000` in your mobile simulator or browser.

---

## 📖 Complete Curriculum Overview

- **Chapter 1:** The 5-Layer Core Competency Stack & Executive Blueprint
- **Chapter 2:** Foundational Mechanics & Model Architecture
- **Chapter 3:** Vector Search & Advanced Production RAG
- **Chapter 4:** Stateful Agentic Systems & State Machines
- **Chapter 5:** Inference Optimization & Serving at Scale
- **Chapter 6:** Automated Evaluations & Security Guardrails
- **Chapter 7:** End-to-End Production Capstone Blueprint (*Autonomous Financial Ledger Auditor*)
- **Chapter 8:** The Accelerated Fast-Track Roadmap (*Compress 9 Months into 60 Days*)
- **Chapter 9:** The "Build in Public" Portfolio & Technical Post-Mortem Playbook

---

## 📜 License
MIT License. Free to adapt, build upon, and distribute.
