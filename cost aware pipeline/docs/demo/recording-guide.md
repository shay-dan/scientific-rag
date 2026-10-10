# Scientific RAG: 75-second demo

## Recording status

This is the recording plan, not a completed video. Use actual saved application results. Do not present simulated progress or invented answers as a live run.

## Storyboard and on-screen captions

| Time | Screen | Caption / narration |
| --- | --- | --- |
| 0–8s | Research page with the question filled in | Scientific RAG: research answers grounded in academic sources. |
| 8–18s | Standard selected; submit a real request | Plans the research and checks existing evidence before searching external sources. |
| 18–27s | Actual stage/model indicator, if this request performs research | Background research with visible progress. Waiting time shortened in this recording. |
| 27–45s | Completed answer; scroll to citations and source records | Inspect the answer and trace its citations to retrieved papers. |
| 45–65s | Open the saved completed 10-case Standard benchmark | Internal benchmark: 10 questions. Inspect aggregate and per-question scores. |
| 65–75s | README Docker instructions | Run locally with Docker. Bring your own Gemini API key. |

Question: **What are the main risks and limitations of CAR-T cell therapy?**

If the request returns a cached answer immediately, show and label that behavior. Do not fabricate a progress segment. Opening saved evaluation results does not require rerunning evaluation. A fresh research request can incur API costs.

## Evaluation caption

For the historical Standard run supplied during development, the reported averages were:

| Metric | Mean |
| --- | ---: |
| Faithfulness | 0.9672 |
| Answer relevancy | 0.8641 |
| Context precision | 0.5047 |
| Context recall | 0.5933 |

Nine cases used Ragas and one used a Gemini-judge fallback. These are scores from one internal run, not accuracy percentages or a verified before/after improvement. Use these figures only alongside that matching saved run; for a different run, show its actual figures.

## Capture

- Open only the application window in the recording area, at a readable zoom.
- Sign in before capture. Keep keys, `.env`, terminals with secrets, and personal browser tabs out of frame.
- Record the application at 1080p if possible. Export an MP4 and a PNG cover screenshot.
- Retain readable pauses on the answer and results. Label cuts in long waits.
- Review the full exported file before publishing.

## Publish on GitHub

Upload the reviewed MP4 as a GitHub Release asset, then link the exact asset URL from the app README. Store the real cover screenshot here as `preview.png` and link that image to the video. Do not add a broken Watch demo button before the asset exists.

The media is a viewing option; retain the Docker instructions for people who want to run the app. Do not commit raw recordings, keys, private exports, or a database dump.
