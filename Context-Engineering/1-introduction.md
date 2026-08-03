https://youtu.be/xlUIiTSaFKI?si=s655G5vJbFyLkGZf

### 1. Paradigm Shift: Prompting vs. Context Engineering

- **Prompt Engineering Limitations**:
    - Focuses primarily on optimizing user text input in a single step.
    - Constrained to one-off API calls or standard chat interfaces without explicit control over peripheral inputs.
        
- **Context Engineering Focus**:
    - Constructs multi-dimensional input environments for LLMs.
    - Orchestrates inputs across user prompts, system instructions, RAG outputs, tools (MCPs), memory states, and conversation history.
        
- **The LLM OS Analogy**:
    - **CPU**: Large Language Model itself (e.g., Claude, GPT, Gemini).
    - **RAM**: Context Window (fast, dynamic, finite, and temporary memory space).
    - **File System / Storage**: External Vector Databases, Knowledge Bases, and RAG pipelines.
    - **System Calls / Applications**: Tool calls, MCP (Model Context Protocol) integrations, and autonomous agents.
        

### 2. The 6 Core Layers of an LLM Context

1. **User Input / Current Query**:
    - The immediate user request or incoming trigger event.
2. **System Prompt / Instructions**:
    - Defines agent persona, operational boundaries, formatting schemas, and non-negotiable business logic.
3. **Retrieved Knowledge (RAG)**:
    - Chunks, docs, or metadata fetched from external databases relevant to the active query.
4. **Tool Definitions & Outputs**:
    - API schemas, function-calling contracts (e.g., MCPs), and returned tool execution payloads.
5. **Conversation History**:
    - Multi-turn message trails (`user` / `assistant` roles) maintaining short-term continuity.
6. **State & Memory**:
    - Persistent user profiles, ongoing feedback logs, session state flags, and global variables.

### 3. Context Capacity & Context Rot Mechanics
- **Window Size vs. Information Density**:
    - Large context windows (1M to 10M+ tokens) allow thousands of pages of raw text, but stuffing them blindly degrades performance.
    - 1 token \approx 0.75 words (or 3/4 of a word).

- **Context Rot**:
    - Performance degradation caused by excessive, noisy, redundant, or stale tokens polluting the context window.
    - Leads to higher financial costs, higher latency, instructions being ignored, and hallucinations.
        
- **Lost in the Middle Effect**:
    - Empirical phenomenon where LLMs attend strongly to tokens at the **beginning** (start) and **end** (finish) of a prompt sequence, while overlooking details located in the **middle**.
    - **Prompt Ordering Strategy**: Place critical system rules and core instructions at the very beginning, and place the active user query at the very end. Place dynamic context chunks or tool outputs in the middle.

### 4. Implementation & Production Patterns

- **Context Budgeting**:
    - Leaving room for generation tokens is critical; stuffing a window to capacity leaves no budget for model outputs.
    - Operating within a target budget range (e.g., 50k–100k tokens in a 200k window) yields better output quality than maxing out the budget.
    
- **Prompt Doubling Technique**:
    - Research shows repeating crucial system prompts or core user instructions twice within the context window can reinforce attention weights and improve compliance, at the cost of doubling input token consumption.
    
- **Memory Architecture Strategies**:
    - **Short-Term Memory**: JSON files or direct message arrays tracking recent turns (e.g., last 20–30 interactions).
    - **Long-Term Memory**: Markdown (`.md`) or XML files containing condensed summaries of past user interactions, preferences, and system states.
        
- **Context Maintenance Commands**:
    - `/clear`: Wipes the context window clean when switching entirely to a new task domain.
    - `/compact`: Summarizes and compresses active context when approaching token thresholds.