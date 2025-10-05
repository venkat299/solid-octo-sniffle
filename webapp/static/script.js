const form = document.getElementById('analyze-form');
const resultsContainer = document.getElementById('results');
const placeholder = document.getElementById('results-placeholder');
const roleIdEl = document.getElementById('role-id');
const roleSummaryEl = document.getElementById('role-summary');
const competencyList = document.getElementById('competency-list');
const footerYear = document.getElementById('footer-year');

footerYear.textContent = new Date().getFullYear();

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector('.button');
  submitButton.classList.add('button--loading');
  submitButton.disabled = true;

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  payload.years_of_experience = Number(payload.years_of_experience);

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Unable to analyze job role.');
    }

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    alert(error.message);
  } finally {
    submitButton.classList.remove('button--loading');
    submitButton.disabled = false;
  }
});

function renderResults(data) {
  placeholder.hidden = true;
  resultsContainer.hidden = false;
  roleIdEl.textContent = data.job_role_id;
  roleSummaryEl.textContent = data.normalized_job_role_summary;

  competencyList.innerHTML = '';
  data.competencies.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'competency-card';

    const name = document.createElement('div');
    name.className = 'competency-card__name';
    name.textContent = item.name;

    const meta = document.createElement('div');
    meta.className = 'competency-card__meta';

    const level = document.createElement('span');
    level.className = 'competency-card__level';
    level.innerHTML = `Level ${item.level}`;

    const indicator = document.createElement('span');
    indicator.className = 'level-indicator';

    for (let i = 1; i <= 5; i += 1) {
      const dot = document.createElement('span');
      if (i <= item.level) {
        dot.classList.add('is-active');
      }
      indicator.appendChild(dot);
    }

    const type = document.createElement('span');
    type.textContent = item.type ? item.type : 'technical';

    level.appendChild(indicator);
    meta.appendChild(level);
    meta.appendChild(type);

    card.appendChild(name);
    card.appendChild(meta);
    competencyList.appendChild(card);
  });
}
