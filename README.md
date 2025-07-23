# A Cognitive Framework for Vision-Language Navigation (Cognitive-VLN-Framework)
## Overview

This repository provides a research framework for implementing, testing, and analyzing sophisticated cognitive architectures for Vision-Language Navigation (VLN) agents. The project is built upon the foundational concepts of the **MapGPT** algorithm but extends significantly into a modular, highly configurable platform designed for in-depth research into the emergent behaviors and failure modes of Large Language Model (LLM) based embodied agents.

The primary goal of this framework is not to present a state-of-the-art model, but to offer a powerful **"glass-box" toolkit** for researchers. It enables the systematic investigation of how different cognitive modules—such as strategic planning, tool-augmented perception, and structured reasoning—influence an agent's navigation behavior in complex 3D environments.

## Core Features

*   **Extended MapGPT Implementation**: A functional implementation of the MapGPT agent, which uses a dynamically generated textual map and history to inform an LLM's navigation decisions.
*   **Detailed Logging**: A non-invasive logging system that captures rich, step-by-step debug information (including candidate views and full LLM outputs) without altering the core algorithm logic.
*   **Failure Diagnosis Tool**: A script (`diagnose_failures.py`) that automatically analyzes experimental results and categorizes common failure modes, such as "Early Stopping" and "Initial Misunderstanding".
*   **Visualization Dashboard**: A script (`create_debug_demo.py`) to generate detailed, multi-panel videos that visualize the agent's perception, thought process, and final decision at every step of its trajectory.

## Framework Enhancements over Baseline MapGPT

This framework introduces several modular enhancements designed to probe and improve the core reasoning capabilities of the agent:

🧠 **1. Strategic Planning Module**: Implements a pre-navigation phase where the LLM acts as a "mission planner" to decompose a high-level instruction into a structured list of key objects to find.

👁️ **2. Tool-Augmented Perception Module**: Provides a "plug-and-play" interface for external vision tools like **Grounding SAM**. This allows the agent to perform instruction-driven visual searches, providing structured, verifiable evidence to the LLM.

🤔 **3. Structured Reasoning & Prompt Engine**: A flexible prompt engineering engine that implements advanced strategies like **Forced Comprehensive Observation** and **MCP (Maximizing Core Prompt)** to guide the LLM's reasoning process and mitigate long-context issues.
