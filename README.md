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

3. **Production Math & Formula Cards (KaTeX):**
   - LaTeX equations rendered inside dedicated, responsive `.formula-block` cards with horizontal scrolling and protection against Markdown character-corruption.
   - Inline math equations styled as elegant semantic badges.

4. **Interactive Architecture Flowcharts (Mermaid.js):**
   - All ASCII architectural blocks upgraded to responsive, vector SVG flowcharts across all chapters.
   - Dynamic dark/light mode synchronization and horizontal touch scrolling for mobile devices.

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
├── chapter_01_foundations.md                     # Standalone deep dive on Tensors, Tokenization & Latent Space
├── chapter_02_attention_and_kv_cache.md           # Standalone deep dive on Attention, GQA & KV-Cache Sizing
├── chapter_03_training_lora_and_alignment.md      # Standalone deep dive on Loss, AdamW, LoRA/QLoRA & DPO
├── chapter_04_production_rag_and_retrieval.md     # Standalone deep dive on Chunking, HNSW/IVF, BM25 & Rerankers
├── chapter_05_agentic_systems_and_orchestration.md# Standalone deep dive on State Graphs, Tools, Memory & HITL
├── chapter_06_serving_and_inference_infrastructure.md # Standalone deep dive on vLLM, PagedAttention, Quantization & Latency
├── chapter_07_evals_safety_and_governance.md     # Standalone deep dive on RAG Triad, LLM-as-a-Judge & EU AI Act
├── chapter_08_capstone_financial_auditor.md      # Standalone deep dive on Autonomous Auditor, pgvector, LangGraph & HITL
├── chapter_09_accelerated_roadmap_and_sprint.md  # Standalone deep dive on 60-Day Sprint, 30-60-30 Protocol & Gate Milestones
├── index.html                                    # Full mobile-first reader with KaTeX, Lucide, and 10 course chapters
├── manifest.json                                 # Progressive Web App (PWA) manifest configuration
├── package.json                                  # Project definition and static serve scripts
├── sw.js                                         # Offline service worker cache engine
├── vercel.json                                   # Production Vercel headers and clean routing
└── README.md                                     # Documentation and deployment guide
```

---

## 🚀 Instant Deployment to Vercel

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete mobile ai handbook reader with accelerated roadmap"
   git push origin main
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

- **Chapter 1:** The Mechanical Foundation: Tensors, BBPE Tokenization, $W_E$ Lookups, RoPE & Latent Space Geometry
- **Chapter 2:** The Attention Engine: Projections ($Q, K, V$), Scaled Dot-Product, GQA & Production KV Cache Math
- **Chapter 3:** Adaptation & Optimization: Cross-Entropy Loss, AdamW, LoRA, QLoRA (NF4) & DPO Alignment
- **Chapter 4:** Production RAG & Hybrid Retrieval: Semantic Window Chunking, HNSW/IVF Graphs, BM25 & Cross-Encoder Rerankers
- **Chapter 5:** Stateful Agentic Systems & Orchestration: Cyclic State Graphs, Pydantic Tool Schemas, Memory Architecture & HITL
- **Chapter 6:** Serving Engines & High-Throughput Infrastructure: vLLM, PagedAttention, Continuous Batching, AWQ/FP8 & Latency SLAs (with Interactive Roofline Sizing Simulator)
- **Chapter 7:** Production Evals, Safety & Enterprise Governance: The RAG Triad, LLM-as-a-Judge, Dual Guardrails & EU AI Act (with Interactive NLI Proposition Auditor)
- **Chapter 8:** Capstone Architecture: Autonomous Financial Risk & Compliance Auditor (with Interactive End-to-End Audit Workbench & HITL Simulator)
- **Chapter 9:** The Accelerated Fast-Track Roadmap: The 30-60-30 Protocol, 4 Gate Milestones & 60-Day Sprint (with Interactive Deliberate Practice Calculator)
- **Chapter 10:** The "Build in Public" Portfolio & Technical Post-Mortem Playbook

---

## 📜 License
MIT License. Free to adapt, build upon, and distribute.
