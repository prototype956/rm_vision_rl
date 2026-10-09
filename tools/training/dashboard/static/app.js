(() => {
  const $ = (id) => document.getElementById(id);
  const state = { presets: [], runs: [], jobs: [], tools: [] };
  const statusText = { queued: '排队', starting: '启动中', running: '运行中', stop_requested: '停止中', complete: '完成', interrupted: '已中断', failed: '失败', stale: '失联', unmanaged: '未托管' };
  const toolNames = ['analysis', 'compare', 'diagnose', 'benchmark', 'view', 'manual'];

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
    const text = await response.text();
    let body = {};
    try { body = text ? JSON.parse(text) : {}; } catch (_) { body = { detail: text }; }
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    return body;
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  }
  function showMessage(message, error = false, target = 'message') {
    const node = $(target); if (!node) return;
    node.textContent = message; node.classList.toggle('hidden', !message);
    node.style.background = error ? '#fde2e2' : '#fff3cd'; node.style.color = error ? '#8e2525' : '#735400';
  }
  function preset() { return state.presets.find(item => item.name === $('preset').value); }
  function updatePreview() {
    const item = preset(); if (!item) return;
    const envs = Math.max(1, Number($('n-envs').value || item.n_envs));
    const value = Math.max(1, Number($('budget-value').value || 1));
    const samples = item.n_steps * envs;
    const requested = $('budget-mode').value === 'updates' ? value * samples : value;
    const actual = Math.ceil(requested / samples) * samples;
    $('budget-preview').textContent = `每次更新 ${samples} 样本；预计 ${Math.ceil(requested / samples)} 次更新；实际预算 ${actual}。batch_size ${samples % item.batch_size === 0 ? '满足' : '不满足'} n_steps × n_envs 的整除约束。`;
  }
  function makeOption(value, text) { const option = document.createElement('option'); option.value = value; option.textContent = text; return option; }
  function fillPresets() {
    const valid = state.presets.filter(item => !item.error);
    ['preset', 'benchmark-preset', 'manual-preset'].forEach(id => {
      const node = $(id); if (!node) return; const current = node.value;
      node.replaceChildren(...valid.map(item => makeOption(item.name, `${item.name} · ${item.policy_kind} · ${item.n_steps} steps`)));
      if (current && valid.some(item => item.name === current)) node.value = current;
    }); updatePreview();
  }
  function checkpointItems() {
    const items = [];
    for (const run of state.runs) for (const name of run.checkpoints || []) items.push({ value: `artifacts/training/${run.run_id}/${name}`, label: `${run.run_id}/${name}` });
    return items;
  }
  function fillCheckpointSelect(id, optional = false) {
    const node = $(id); if (!node) return; const current = node.value;
    const options = checkpointItems().map(item => makeOption(item.value, item.label));
    if (optional) options.unshift(makeOption('', '从随机策略开始'));
    node.replaceChildren(...options); if (current && options.some(option => option.value === current)) node.value = current;
  }
  function fillCheckpoints() {
    fillCheckpointSelect('checkpoint');
    ['compare-checkpoint', 'compare-baseline', 'diagnose-checkpoint', 'view-checkpoint'].forEach(id => fillCheckpointSelect(id));
    fillCheckpointSelect('manual-checkpoint', true);
  }
  function fillRuns() {
    const completeRuns = state.runs.filter(run => run.status === 'complete'); const node = $('tool-analysis-run'); if (!node) return;
    const current = node.value; node.replaceChildren(...completeRuns.map(run => makeOption(run.run_id, `${run.run_id} · ${run.last_completed_timesteps ?? 0} steps`)));
    if (current && completeRuns.some(run => run.run_id === current)) node.value = current;
  }
  function renderRows() {
    const rows = [];
    for (const job of state.jobs) {
      const button = ['running', 'starting', 'stop_requested', 'queued'].includes(job.status) ? `<button class="danger" data-stop="${escapeHtml(job.job_id)}">停止</button>` : '';
      const kind = job.mode === 'tool' && job.tool ? `tool:${job.tool}` : (job.mode || 'job');
      rows.push(`<tr><td>${escapeHtml(kind)}</td><td>${escapeHtml(job.job_id)}</td><td class="status ${escapeHtml(job.status)}">${statusText[job.status] || escapeHtml(job.status)}</td><td>${escapeHtml(job.requested_timesteps ?? job.timesteps ?? '—')}</td><td>${escapeHtml(job.created_at || '—')}</td><td><button class="secondary" data-job="${escapeHtml(job.job_id)}">详情</button>${button}</td></tr>`);
    }
    for (const run of state.runs) rows.push(`<tr><td>run</td><td>${escapeHtml(run.run_id)}</td><td class="status ${escapeHtml(run.status)}">${statusText[run.status] || run.status}</td><td>${escapeHtml(run.last_completed_timesteps ?? '—')}</td><td>—</td><td><button class="secondary" data-run="${escapeHtml(run.run_id)}">详情</button>${run.status === 'complete' ? `<button class="secondary" data-analyze="${escapeHtml(run.run_id)}">分析</button>` : ''}</td></tr>`);
    $('jobs').innerHTML = rows.join('') || '<tr><td colspan="6">暂无任务或训练运行</td></tr>';
    document.querySelectorAll('[data-job]').forEach(node => node.onclick = () => showJob(node.dataset.job)); document.querySelectorAll('[data-run]').forEach(node => node.onclick = () => showRun(node.dataset.run)); document.querySelectorAll('[data-stop]').forEach(node => node.onclick = () => stopJob(node.dataset.stop)); document.querySelectorAll('[data-analyze]').forEach(node => node.onclick = () => analyze(node.dataset.analyze));
  }
  async function refreshTools() {
    try { const response = await api('/api/tools'); state.tools = response.tools || []; }
    catch (_) { state.tools = toolNames.map(name => ({ name })); }
  }
  async function refresh() {
    try {
      const [presets, runs] = await Promise.all([api('/api/configs/training'), api('/api/runs')]);
      state.presets = presets.presets || []; state.runs = runs.runs || []; state.jobs = runs.jobs || [];
      fillPresets(); fillCheckpoints(); fillRuns(); renderRows();
    } catch (error) { showMessage(error.message, true); }
  }
  async function submitNew(event) {
    event.preventDefault();
    try { const job = await api('/api/jobs/train', { method: 'POST', body: JSON.stringify({ preset: $('preset').value, budget_mode: $('budget-mode').value, budget_value: Number($('budget-value').value), n_envs: Number($('n-envs').value), device: $('device').value, seed: Number($('seed').value), checkpoint_updates: Number($('checkpoint-updates').value), auto_analysis: $('auto-analysis').checked }) }); showMessage(`训练已启动：${job.job_id}`); await refresh(); showJob(job.job_id); }
    catch (error) { showMessage(error.message, true); }
  }
  async function submitResume(event) {
    event.preventDefault();
    try { const body = { checkpoint: $('checkpoint').value, budget_mode: $('resume-budget-mode').value, budget_value: Number($('resume-budget-value').value), auto_analysis: $('resume-auto-analysis').checked }; if ($('resume-n-envs').value) body.n_envs = Number($('resume-n-envs').value); if ($('resume-device').value) body.device = $('resume-device').value; const job = await api('/api/jobs/resume', { method: 'POST', body: JSON.stringify(body) }); showMessage(`恢复训练已启动：${job.job_id}`); await refresh(); showJob(job.job_id); }
    catch (error) { showMessage(error.message, true); }
  }
  async function stopJob(id) { try { await api(`/api/jobs/${encodeURIComponent(id)}/stop`, { method: 'POST', body: '{}' }); showMessage('已请求优雅停止'); await refresh(); } catch (error) { showMessage(error.message, true); } }
  async function analyze(id) { try { const job = await api(`/api/runs/${encodeURIComponent(id)}/analyze`, { method: 'POST', body: '{}' }); showMessage(`分析任务已启动：${job.job_id}`); await refresh(); showJob(job.job_id); } catch (error) { showMessage(error.message, true); } }
  function commaNumbers(id) { const raw = $(id).value.trim(); if (!raw) return undefined; const values = raw.split(',').map(item => Number(item.trim())); if (values.some(value => !Number.isInteger(value) || value < 0)) throw new Error('seed/环境数必须是逗号分隔的非负整数'); return values; }
  async function submitTool(tool, payload, label) {
    try { const job = await api(`/api/tools/${encodeURIComponent(tool)}/run`, { method: 'POST', body: JSON.stringify(payload) }); showMessage(`${label}任务已启动：${job.job_id}`, false, 'tools-message'); await refresh(); showJob(job.job_id); }
    catch (error) { showMessage(error.message, true, 'tools-message'); }
  }
  async function submitAnalysis(event) { event.preventDefault(); await submitTool('analysis', { run_dir: `artifacts/training/${$('tool-analysis-run').value}`, evaluation_scenes: Number($('analysis-scenes').value), device: $('analysis-device').value || undefined }, 'analysis'); }
  async function submitCompare(event) { event.preventDefault(); await submitTool('compare', { checkpoint: $('compare-checkpoint').value, baseline_checkpoint: $('compare-baseline').value, device: $('compare-device').value || undefined, workers: Number($('compare-workers').value) }, 'compare'); }
  async function submitDiagnose(event) { event.preventDefault(); await submitTool('diagnose', { checkpoint: $('diagnose-checkpoint').value, rollouts: Number($('diagnose-rollouts').value), device: $('diagnose-device').value || undefined }, 'diagnose'); }
  async function submitBenchmark(event) {
    event.preventDefault();
    try {
      const body = { preset: $('benchmark-preset').value, n_envs: commaNumbers('benchmark-envs'), repeats: Number($('benchmark-repeats').value), rollout_mode: $('benchmark-rollout-mode').value };
      const budget = Number($('benchmark-budget-value').value);
      body[$('benchmark-budget-mode').value] = budget;
      await submitTool('benchmark', body, 'benchmark');
    } catch (error) { showMessage(error.message, true, 'tools-message'); }
  }
  async function submitView(event) { event.preventDefault(); try { await submitTool('view', { checkpoint: $('view-checkpoint').value, scene_count: Number($('view-scene-count').value), scene_seeds: commaNumbers('view-seeds'), workers: Number($('view-workers').value), device: $('view-device').value || undefined }, 'view'); } catch (error) { showMessage(error.message, true, 'tools-message'); } }
  async function submitManual(event) {
    event.preventDefault();
    const checkpoint = $('manual-checkpoint').value;
    const body = { mode: checkpoint ? 'checkpoint' : 'config', episode_steps: Number($('manual-episode-steps').value), scene_seed: Number($('manual-scene-seed').value) };
    if (checkpoint) body.checkpoint = checkpoint;
    else body.preset = $('manual-preset').value;
    await submitTool('manual', body, 'manual');
  }
  function artifactLink(item) { if (!item) return null; if (typeof item === 'string') return { href: item.startsWith('/') || item.startsWith('http') ? item : `/api/artifacts/${item}`, label: item }; if (item.url || item.href) return { href: item.url || item.href, label: item.label || item.path || item.url || item.href }; return null; }
  function renderLinks(container, data) { container.replaceChildren(); const links = []; for (const item of data.artifacts || []) { const link = artifactLink(item); if (link) links.push(link); } for (const key of ['report_url', 'report_path', 'html_url', 'replay_url', 'results_url', 'command_url']) { const link = artifactLink(data[key]); if (link) links.push(link); } const unique = new Map(links.map(link => [link.href, link])); for (const link of unique.values()) { const a = document.createElement('a'); a.href = link.href; a.textContent = `打开 ${link.label}`; a.target = '_blank'; a.rel = 'noopener'; container.append(a, document.createElement('br')); } }
  async function showJob(id) { try { const job = await api(`/api/jobs/${encodeURIComponent(id)}`); $('detail').classList.remove('hidden'); $('detail-json').textContent = JSON.stringify(job, null, 2); $('detail-log').textContent = await fetch(`/api/jobs/${encodeURIComponent(id)}/log`).then(response => response.text()); renderLinks($('detail-links'), job); } catch (error) { showMessage(error.message, true); } }
  async function showRun(id) { try { const run = await api(`/api/runs/${encodeURIComponent(id)}`); $('detail').classList.remove('hidden'); $('detail-json').textContent = JSON.stringify(run, null, 2); $('detail-log').textContent = ''; const links = $('detail-links'); links.replaceChildren(); for (const item of run.analysis || []) { const path = item.replace(/analysis\.json$/, 'report.html'); const a = document.createElement('a'); a.href = `/api/artifacts/${encodeURIComponent(id)}/${path}`; a.textContent = `打开 ${path}`; a.target = '_blank'; a.rel = 'noopener'; links.append(a, document.createElement('br')); } } catch (error) { showMessage(error.message, true); } }
  function setupTabs() { document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => { document.querySelectorAll('[data-tab]').forEach(item => item.classList.toggle('active', item === button)); ['training-panel', 'tools-panel'].forEach(id => $(id).classList.toggle('hidden', id !== button.dataset.tab)); }); }
  $('new-form').onsubmit = submitNew; $('resume-form').onsubmit = submitResume; $('refresh').onclick = refresh; $('refresh-tools').onclick = async () => { await refreshTools(); await refresh(); }; $('close-detail').onclick = () => $('detail').classList.add('hidden'); $('preset').onchange = updatePreview; $('budget-mode').onchange = updatePreview; $('budget-value').oninput = updatePreview; $('n-envs').oninput = updatePreview;
  $('analysis-form').onsubmit = submitAnalysis; $('compare-form').onsubmit = submitCompare; $('diagnose-form').onsubmit = submitDiagnose; $('benchmark-form').onsubmit = submitBenchmark; $('view-form').onsubmit = submitView; $('manual-form').onsubmit = submitManual;
  setupTabs(); refreshTools(); refresh(); setInterval(refresh, 2000);
})();
