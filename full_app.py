"""Gradio entry point for the full Scientific RAG research pipeline."""

from __future__ import annotations

import os

import gradio as gr

from new_scientific_rag import Orchestrator, ResearchAgent, VectorService


services: tuple[VectorService, ResearchAgent, Orchestrator] | None = None


def get_services() -> tuple[VectorService, ResearchAgent, Orchestrator]:
    global services
    if services is None:
        vector_service = VectorService()
        research_agent = ResearchAgent(vector_service)
        services = (vector_service, research_agent, Orchestrator(vector_service, research_agent))
    return services


def format_sources(documents: list) -> str:
    if not documents:
        return "No external academic sources were used for this answer."
    lines = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = metadata.get("title", "Untitled source")
        source = metadata.get("source", "unknown")
        identifier = metadata.get("doc_id", metadata.get("id", ""))
        lines.append(f"{index}. **{title}** — {source} {identifier}".strip())
    return "\n".join(lines)


async def run_research(
    question: str,
    clear_knowledge_base: bool,
    generate_evaluation_dataset: bool,
    progress=gr.Progress(),
) -> tuple[str, str, str, list]:
    if not question.strip():
        return "Please enter a research question.", "", "", []
    try:
        progress(0.05, desc="Starting the full research pipeline…")
        _, _, orchestrator = get_services()
        progress(0.15, desc="Planning and validating sub-questions…")
        result = await orchestrator.execute_full_study(
            original_query=question,
            clear_db=clear_knowledge_base,
            generate_evaluation_dataset=generate_evaluation_dataset,
        )
        progress(0.9, desc="Formatting report and sources…")
        report = result["report"]
        plan = result.get("metadata", {}).get("plan_executed", [])
        progress(1.0, desc="Done")
        return (
            report.get("answer", "No answer was generated."),
            format_sources(report.get("raw_docs_used", [])),
            "\n".join(f"- {item}" for item in plan) or "No external sub-queries were required.",
            result.get("evaluation_dataset", []),
        )
    except Exception as error:
        return f"Research request failed: {error}", "", "", []


with gr.Blocks(title="Scientific RAG — Full Pipeline") as app:
    gr.Markdown(
        "# Scientific RAG — Full Pipeline\n"
        "Planner → gatekeeper → PubMed/arXiv mining → hybrid retrieval → cited synthesis. "
        "This tool is for research assistance, not medical advice."
    )
    question = gr.Textbox(label="Research question", lines=3, placeholder="Enter a scientific research question…")
    with gr.Row():
        clear_database = gr.Checkbox(label="Clear knowledge base before research", value=False)
        generate_evaluation = gr.Checkbox(label="Generate synthetic evaluation Q&A dataset", value=False)
    submit = gr.Button("Run full research study", variant="primary")
    answer = gr.Markdown(label="Research report")
    with gr.Accordion("Sources used", open=False):
        sources = gr.Markdown()
    with gr.Accordion("Executed research plan", open=False):
        plan = gr.Markdown()
    with gr.Accordion("Generated evaluation dataset", open=False):
        evaluation_data = gr.JSON()
    inputs = [question, clear_database, generate_evaluation]
    outputs = [answer, sources, plan, evaluation_data]
    submit.click(run_research, inputs=inputs, outputs=outputs)
    question.submit(run_research, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
