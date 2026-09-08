https://youtu.be/HvBCcmzS_vs?si=L9J2wHXJ1P_okyD1

### 1. Six Layers of Context Architecture

- **Stack Composition**:
    1. **System Prompt** (Rules, Guardrails, Persona).
    2. **Memory / State** (Persistent long-term preferences, user data across sessions).
    3. **RAG Output** (Dynamic context fetched on demand from vector databases/knowledge bases).
    4. **Tools / MCP** (Function calling outputs, Model Context Protocol integration).
    5. **Conversation History / Session Memory** (Short-term chat thread exchanges).
    6. **User Prompt** (Immediate user instruction/query).

- **Positional Priority & Lost-in-the-Middle Effect**:
    - The **System Prompt** sits at the absolute start (highest priority for rules and guardrails).
    - The **User Prompt** is intentionally placed at the absolute end.
    - Arranging critical instructions at the two extremes minimizes the "lost-in-the-middle" attention degradation found in large context windows.
        
- **Memory vs. Conversation History**:
    - **Conversation History (Session Memory)**: Non-persistent, short-term context representing the immediate chat exchange (e.g., 25 previous email turns in a thread).
    - **State / Memory**: Persistent across sessions, containing long-term profiles, writing style preferences, or historical interaction patterns over months/years.

### 2. Anatomy of a System Prompt (`CLAUDE.md`)

A structured configuration file (such as `CLAUDE.md` or `AGENTS.md`) is built around five distinct structural elements:

1. **Identity / Project Overview**:
    - Defines what the tool or agent is built to accomplish (e.g., "Email response agent" or "Automated site tracker").
    - Does not require anthropomorphic human descriptions (e.g., "You are a senior developer").
        
2. **Rules & Guardrails**:
    - Strict behavioral constraints and operational boundaries (e.g., "Never approve refunds without checking policy").
    - Represents the highest priority component in the system prompt; rules are the last item removed when context is constrained.
        
3. **Format / Project Structure**:
    - Dictates the response structure, file formats (JSON, Python, Markdown), or repo directory layouts.
        
4. **Knowledge Base**:
    - Represents the factual domain constraints and environment specs (e.g., tech stack details, environmental variables, development commands).
    - _Knowledge vs. Rules Distinction_: Knowledge defines the **reality/state** of the application, whereas Rules define **behavioral control**.
5. **Tools**:
    - Explicitly defines accessible function calls, scripts, packages, APIs, or external library integrations.

### 3. Compute, Memory, & Context Selection

- **Context Budget Constraints**:
    - Keep `CLAUDE.md` under ~200 lines (loads into every request, each line costs tokens).
    - Keep the `SKILL.md` body under ~500 lines (loads only when invoked; split longer content into separate files one level deep).
    - Respecting these limits avoids context rot and excessive API usage costs.

- **Selective Context Loading (Dynamic vs. Static)**:
    - System prompts do not need to inject the complete configuration into every API call.
    - **Static Rules**: Universal guardrails (e.g., safety, output formats, core constraints) are injected permanently into every single call.
    - **Dynamic Sections**: Domain-specific or phase-specific sections (e.g., Phase 1 features, specific API schemas) are retrieved selectively via keyword matching or RAG.
        
- **Sub-Agent Context Isolation**:
    - Hierarchical sub-agents maintain isolated context windows.
    - Spinning off a dedicated sub-agent (e.g., a code executor or paper writer) prevents global conversation history from cluttering specialized API calls.

### 4. Right-Altitude Principle for System Instructions

- **Concept**: Instructions must avoid being "too high" (overly vague) or "too low" (overly specific and brittle).
    - **Too High (Vague)**: _"Be helpful with pricing"_ or _"Do not hallucinate"_ (provides zero operational boundaries).
    - **Too Low (Brittle)**: _"Reply that plans start at $9.99 per month"_ (breaks immediately if prices or policies update).
    - **Right Altitude**: _"When asked about pricing, query the pricing database first; if no exact match exists, suggest the closest plan."

### 5. Multi-Agent Hierarchy & Standards

- **Universal Standards (`AGENTS.md` & `SKILL.md`)**:
    - While `CLAUDE.md` is standard in Claude Code, `AGENTS.md` is a cross-platform standard (formalized by OpenAI) honored across multiple tools and open ecosystems.
    - **Hierarchical Overrides**: Nested configuration files in subdirectories inherit from parent/root configurations, but local subdirectory rules override global root rules in cases of direct conflict.
        
- **Few-Shot Examples Strategy**:
    - **Diversity over Quantity**: Injecting 3 highly diverse, multi-category examples yields optimal performance.
    - Adding non-diverse or redundant examples wastes token budget, risks overindexing on specific classes, and suffers from steep diminishing returns.
        
    - **Pattern Selection**:
        - _Classification_: Input-output pairs.
        - _Complex Reasoning / Policies_: Chain-of-Thought (Input \to Reasoning steps \to Output).
        - _Structured Formats_: Output templates/schemas.
            

### 6. Common Anti-Patterns & Pitfalls
- **Speculative Prompt Over-Engineering**: Writing massive, 1000+ line system prompts upfront leads to self-contradictory rules.
- **Incremental Construction**: Start minimal, test edge cases, and add explicit guardrails sequentially as failures occur.
- **Anti-Patterns in Codebases**: Adding a "Common Mistakes to Avoid" section provides high value per token by encoding tacit project conventions that cannot be inferred purely from reading existing source files.
