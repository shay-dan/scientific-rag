const question = document.querySelector('#question');
const button = document.querySelector('#research-button');
const status = document.querySelector('#status');
const stage = document.querySelector('#stage');
const result = document.querySelector('#result');
const errorBox = document.querySelector('#error');
const answer = document.querySelector('#answer');
const sourceList = document.querySelector('#source-list');
const resultMeta = document.querySelector('#result-meta');
const costNote = document.querySelector('#cost-note');

const notes = { fast: 'Fast mode uses the smallest research budget and is the default.', standard: 'Standard mode balances coverage and API use.', deep: 'Deep mode mines more sources and is intentionally restricted by the server.' };
const stages = ['Planning the research approach…', 'Searching the knowledge base…', 'Mining academic sources…', 'Reading and synthesizing evidence…', 'Writing the final report…'];

document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', () => { costNote.textContent = notes[input.value]; }));

function selectedMode() { return document.querySelector('input[name="mode"]:checked').value; }
function show(element) { element.classList.remove('hidden'); }
function hide(element) { element.classList.add('hidden'); }

button.addEventListener('click', async () => {
  const text = question.value.trim();
  if (text.length < 3) { errorBox.textContent = 'Please enter a research question of at least three characters.'; show(errorBox); return; }
  hide(errorBox); hide(result); show(status); button.disabled = true; button.textContent = 'Researching…';
  let index = 0; stage.textContent = stages[index];
  const timer = setInterval(() => { index = Math.min(index + 1, stages.length - 1); stage.textContent = stages[index]; }, 6500);
  try {
    const response = await fetch('/v1/research', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, mode: selectedMode(), generate_evaluation_dataset: false }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The research request failed.');
    answer.textContent = data.answer || 'No report was generated.';
    resultMeta.textContent = data.cached ? `Cached ${data.mode} result` : `${data.mode} research`;
    sourceList.replaceChildren(...data.sources.map(source => {
      const item = document.createElement(source.source_url ? 'a' : 'div');
      item.className = 'source'; if (source.source_url) { item.href = source.source_url; item.target = '_blank'; item.rel = 'noreferrer'; }
      const title = document.createElement('span'); title.className = 'source-title'; title.textContent = source.title;
      const meta = document.createElement('span'); meta.className = 'source-meta'; meta.textContent = source.source + (source.external_id ? ` · ${source.external_id}` : '');
      item.append(title, meta); return item;
    }));
    show(result); result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { errorBox.textContent = error.message; show(errorBox); }
  finally { clearInterval(timer); hide(status); button.disabled = false; button.innerHTML = 'Start research <span>→</span>'; }
});
