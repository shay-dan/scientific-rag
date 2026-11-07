"""Gradio user interface for the full research pipeline."""

from __future__ import annotations

import logging

import gradio as gr

from .application import ResearchApplication

logger = logging.getLogger(__name__)


def create_ui(application: ResearchApplication) -> gr.Blocks:
    def format_sources(sources: list[dict[str, str]]) -> str:
        if not sources:
            return "No external academic sources were used for this answer."
        return "\n".join(
            f"{index}. **{source['title']}** — {source['source']} {source['id']}".strip()
            for index, source in enumerate(sources, start=1)
        )

    async def run_research(
        question: str,
        clear_knowledge_base: bool,
        generate_evaluation_dataset: bool,
        progress=gr.Progress(),
    ) -> tuple[str, str, str, list]:
        try:
            progress(0.05, desc="Starting the full research pipeline…")
            progress(0.15, desc="Planning and validating sub-questions…")
            result = await application.run(
                question,
                clear_knowledge_base=clear_knowledge_base,
                generate_evaluation_dataset=generate_evaluation_dataset,
            )
            progress(0.9, desc="Formatting report and sources…")
            progress(1.0, desc="Done")
            plan = "\n".join(f"- {item}" for item in result.plan) or "No external sub-queries were required."
            return result.answer, format_sources(result.sources), plan, result.evaluation_dataset
        except Exception as error:
            logger.exception("Research request failed")
            return f"Research request failed: {error}", "", "", []

    with gr.Blocks(title="Scientific RAG — Full Pipeline") as app:
        gr.Markdown(
            "# Scientific RAG — Full Pipeline\n"
            "Planner → gatekeeper → PubMed/arXiv mining → hybrid retrieval → cited synthesis. "
            "This tool is for research assistance, not medical advice."
        )
        question = gr.Textbox(label="Research question", lines=3, placeholder="Enter a scientific research question…")
        with gr.Row():
            clear_database = gr.Checkbox(
                label="Clear knowledge base before research",
                value=False,
                visible=application.settings.allow_database_reset,
            )
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
    return app
