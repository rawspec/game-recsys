const state = { mode: 'rec', offset: 0, loading: false, done: false, total: 0, algo: 'hybrid' };

const feed = document.getElementById('feed');
const metricsBox = document.getElementById('metrics-box');
const sentinel = document.getElementById('sentinel');
const endMsg = document.getElementById('end');
const hint = document.getElementById('hint');

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, opts) {
  const init = opts
    ? { method: opts.method || 'POST', headers: { 'Content-Type': 'application/json' },
        body: opts.body ? JSON.stringify(opts.body) : undefined }
    : undefined;
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function placeholderImg(img) {
  img.onerror = null;
  img.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="460" height="215">' +
    '<rect width="100%" height="100%" fill="#1b2230"/>' +
    '<text x="50%" y="50%" fill="#5b6b85" font-size="20" text-anchor="middle" font-family="sans-serif">нет обложки</text></svg>');
}

function attachImg(img, src) {
  img.onerror = () => placeholderImg(img);
  if (src) img.src = src; else placeholderImg(img);
}

function makeStars(value, onRate) {
  const wrap = el('div', 'stars');
  for (let i = 1; i <= 10; i++) {
    const b = el('button', 'star' + (value && i <= Math.round(value) ? ' on' : ''), '★');
    b.type = 'button';
    b.title = `Оценка ${i}/10`;
    b.addEventListener('click', (e) => { e.stopPropagation(); onRate(i); });
    wrap.appendChild(b);
  }
  wrap.appendChild(el('span', 'stars-val', value ? String(value) : ''));
  return wrap;
}

function card(game) {
  const c = el('article', 'card');
  const img = el('img', 'cover');
  attachImg(img, game.image);
  c.appendChild(img);

  const body = el('div', 'card-body');
  body.appendChild(el('h3', 'card-title', game.name));
  const meta = [];
  if (game.year) meta.push(String(game.year));
  if (game.genres && game.genres.length) meta.push(game.genres.slice(0, 3).join(' · '));
  body.appendChild(el('p', 'meta', meta.join(' · ')));

  const chips = el('div', 'chips');
  if (game.score !== undefined && game.score !== null) {
    chips.appendChild(el('span', 'badge', `совпадение ${Math.round(game.score * 100)}%`));
  }
  if (game.reason) chips.appendChild(el('span', 'reason', game.reason));
  body.appendChild(chips);

  body.appendChild(makeStars(game.rating, (v) => rate(game.game_id, v)));
  c.appendChild(body);

  const hideBtn = el('button', 'hide-btn', '✕');
  hideBtn.type = 'button';
  hideBtn.title = 'Не показывать';
  hideBtn.addEventListener('click', (e) => { e.stopPropagation(); hide(game.game_id); });
  c.appendChild(hideBtn);

  c.addEventListener('click', () => openModal(game));
  return c;
}

async function rate(gameId, rating) {
  await api('/api/rate', { body: { game_id: gameId, rating } });
  await refreshStats();
  if (state.mode === 'rec') reloadFeed(); else renderRated();
}

async function unrate(gameId) {
  await fetch(`/api/rate/${gameId}`, { method: 'DELETE' });
  await refreshStats();
  if (state.mode === 'rec') reloadFeed(); else renderRated();
}

async function hide(gameId) {
  await api('/api/hide', { body: { game_id: gameId } });
  closeModal();
  await refreshStats();
  if (state.mode === 'rec') reloadFeed();
}

async function loadMore() {
  if (state.loading || state.done || state.mode !== 'rec') return;
  state.loading = true;
  try {
    const data = await api(`/api/recommendations?offset=${state.offset}&limit=24&algo=${encodeURIComponent(state.algo)}`);
    state.total = data.total;
    for (const g of data.items) feed.appendChild(card(g));
    state.offset += data.items.length;
    state.done = data.items.length === 0 || state.offset >= data.total;
    endMsg.hidden = !state.done;
  } catch (e) {
    state.loading = false;
    throw e;
  }
  state.loading = false;
  // бутстрап ленты: если контент не создаёт скролл (страница ниже экрана) —
  // догружаем ещё, иначе подгрузка по скроллу никогда не сработает
  if (!state.done && document.documentElement.scrollHeight <= window.innerHeight + 200) {
    return loadMore();
  }
}

function reloadFeed() {
  feed.textContent = '';
  state.offset = 0;
  state.done = false;
  endMsg.hidden = true;
  loadMore();
}

async function renderRated() {
  state.loading = true;
  try {
    const data = await api('/api/ratings');
    feed.textContent = '';
    for (const g of data.items) {
      const c = card(g);
      const clear = el('button', 'btn-ghost small', 'убрать оценку');
      clear.type = 'button';
      clear.addEventListener('click', (e) => { e.stopPropagation(); unrate(g.game_id); });
      c.querySelector('.card-body').appendChild(clear);
      feed.appendChild(c);
    }
    if (!data.items.length) {
      feed.appendChild(el('p', 'hint',
        'Пока пусто. Найдите игры через поиск сверху и поставьте оценки — тогда появятся рекомендации.'));
    }
    endMsg.hidden = true;
  } finally {
    state.loading = false;
  }
}

function setMode(mode) {
  state.mode = mode;
  document.getElementById('tab-rec').classList.toggle('active', mode === 'rec');
  document.getElementById('tab-rated').classList.toggle('active', mode === 'rated');
  document.getElementById('tab-metrics').classList.toggle('active', mode === 'metrics');
  metricsBox.hidden = mode !== 'metrics';
  sentinel.style.display = mode === 'rec' ? '' : 'none';
  feed.style.display = mode === 'metrics' ? 'none' : '';
  if (mode === 'rec') {
    hint.textContent = 'Листайте ленту и оценивайте игры от 1 до 10 — рекомендации подстроятся мгновенно.';
    reloadFeed();
  } else if (mode === 'rated') {
    hint.textContent = 'Игры, которые вы оценили:';
    renderRated();
  } else {
    hint.textContent = 'Офлайн-оценка алгоритмов на holdout-сплите (скрипт scripts/09_evaluate.py):';
    renderMetrics();
  }
}

async function renderMetrics() {
  metricsBox.textContent = '';
  const data = await api('/api/metrics');
  if (!data.ready) {
    metricsBox.appendChild(el('p', 'hint',
      'Метрики ещё не посчитаны. Запустите: .venv/Scripts/python scripts/09_evaluate.py'));
    return;
  }
  const names = { popularity: 'Popularity', content: 'Content (MiniLM)', itemknn: 'ItemKNN',
    als: 'ALS (implicit)', bpr: 'BPR-MF', twotower: 'Two-Tower (DL)', lightfm: 'LightFM' };
  const keys = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20', 'map@10', 'coverage@20', 'novelty', 'diversity'];
  const table = el('table', 'metrics-table');
  const head = el('tr');
  head.appendChild(el('th', null, 'Модель'));
  for (const k of keys) head.appendChild(el('th', null, k));
  table.appendChild(head);
  const order = ['popularity', 'content', 'itemknn', 'als', 'bpr', 'twotower', 'lightfm'];
  for (const name of order) {
    const r = data.items[name];
    if (!r) continue;
    const tr = el('tr');
    tr.appendChild(el('td', 'model-name', names[name] || name));
    for (const k of keys) tr.appendChild(el('td', null, (r[k] ?? 0).toFixed(4)));
    table.appendChild(tr);
  }
  metricsBox.appendChild(table);
  const img = el('img', 'metrics-img');
  img.src = '/results/metrics.png';
  img.alt = 'Графики метрик';
  metricsBox.appendChild(img);
  metricsBox.appendChild(el('p', 'hint',
    'Recall/NDCG/MAP — качество ранжирования для пользователей из теста. Coverage — доля каталога в рекомендациях. Novelty — неожиданность (выше = менее популярные советы). Diversity — разнообразие списка.'));
}

const observer = new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting)) loadMore();
}, { rootMargin: '800px' });
observer.observe(sentinel);

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimer = null;

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (!q) { searchResults.hidden = true; return; }
  searchTimer = setTimeout(async () => {
    try {
      const data = await api(`/api/search?q=${encodeURIComponent(q)}&limit=20`);
      searchResults.textContent = '';
      if (!data.items.length) searchResults.appendChild(el('div', 'dd-empty', 'Ничего не найдено'));
      for (const g of data.items) {
        const row = el('div', 'dd-row');
        attachImg(el('img'), g.image);
        const info = el('div', 'dd-info');
        info.appendChild(el('div', 'dd-name', g.name));
        info.appendChild(el('div', 'meta',
          [g.year ? String(g.year) : '', (g.genres || []).slice(0, 2).join(' · ')]
            .filter(Boolean).join(' · ')));
        row.appendChild(info);
        if (g.rating) row.appendChild(el('span', 'badge', `${g.rating}/10`));
        row.addEventListener('click', () => {
          searchResults.hidden = true;
          searchInput.blur();
          openModal(g);
        });
        searchResults.appendChild(row);
      }
      searchResults.hidden = false;
    } catch (e) { /* молча пропускаем */ }
  }, 250);
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search')) searchResults.hidden = true;
});

const modal = document.getElementById('modal');
let modalGame = null;

function renderModalStars() {
  const box = document.getElementById('modal-stars');
  box.textContent = '';
  box.appendChild(makeStars(modalGame.rating, async (v) => {
    await rate(modalGame.game_id, v);
    modalGame.rating = v;
    document.getElementById('modal-rating-val').textContent = `${v}/10`;
    renderModalStars();
  }));
}

function openModal(game) {
  modalGame = game;
  attachImg(document.getElementById('modal-img'), game.image);
  document.getElementById('modal-name').textContent = game.name;
  document.getElementById('modal-meta').textContent =
    [game.year ? String(game.year) : '', (game.genres || []).join(' · ')]
      .filter(Boolean).join(' · ');
  document.getElementById('modal-desc').textContent = game.description || 'Описание отсутствует.';
  const tagsBox = document.getElementById('modal-tags');
  tagsBox.textContent = '';
  for (const t of (game.tags || [])) tagsBox.appendChild(el('span', 'chip', t));
  document.getElementById('modal-rating-val').textContent = game.rating ? `${game.rating}/10` : '—';
  renderModalStars();
  document.getElementById('modal-hide').onclick = () => hide(game.game_id);
  modal.hidden = false;
}

function closeModal() { modal.hidden = true; }

document.getElementById('modal-close').addEventListener('click', closeModal);
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

async function refreshStats() {
  const s = await api('/api/stats');
  const pill = document.getElementById('rated-count');
  pill.hidden = !s.rated;
  pill.textContent = s.rated || '';
  document.getElementById('stats').textContent = `${s.catalog.toLocaleString('ru-RU')} игр в базе`;
}

async function fillAlgoSelect() {
  const sel = document.getElementById('algo-select');
  let data;
  try {
    data = await api('/api/algos');
  } catch (e) {
    return;
  }
  sel.textContent = '';
  const names = { popularity: 'Популярность', content: 'Контент (MiniLM)', itemknn: 'ItemKNN',
    als: 'ALS', bpr: 'BPR-MF', twotower: 'Two-Tower (DL)', lightfm: 'LightFM', hybrid: 'Гибрид (ансамбль)' };
  for (const info of data.items) {
    const opt = el('option', null, names[info.name] || info.display_name);
    opt.value = info.name;
    sel.appendChild(opt);
  }
  const hybrid = [...sel.options].find((o) => o.value === 'hybrid');
  if (hybrid) sel.value = 'hybrid';
  sel.addEventListener('change', () => {
    state.algo = sel.value;
    if (state.mode === 'rec') reloadFeed();
  });
}

document.getElementById('tab-rec').addEventListener('click', () => setMode('rec'));
document.getElementById('tab-rated').addEventListener('click', () => setMode('rated'));
document.getElementById('tab-metrics').addEventListener('click', () => setMode('metrics'));

refreshStats().catch(() => {});
fillAlgoSelect().finally(() => setMode('rec'));
