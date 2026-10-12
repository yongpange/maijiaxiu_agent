const previewBtn = document.getElementById('previewBtn');
const generateBtn = document.getElementById('generateBtn');
const statusBadge = document.getElementById('statusBadge');
const progressArea = document.getElementById('progressArea');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const tableArea = document.getElementById('tableArea');
const approvalArea = document.getElementById('approvalArea');

const ACTIVE_BATCH_KEY = 'maijiaxiu_active_batch';

let previewRows = [];
let requireConfirmation = false;
let trackingBatchId = '';

previewBtn.addEventListener('click', loadPreview);
generateBtn.addEventListener('click', startBatchGenerate);
loadPendingApprovals();
restoreActiveBatch();

async function loadPreview() {
  previewBtn.disabled = true;
  previewBtn.textContent = '读取中...';
  generateBtn.disabled = true;

  try {
    const resp = await fetch('/batch-generate/preview');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    requireConfirmation = !!data.require_confirmation;
    previewRows = data.rows || [];
    renderPreviewTable(previewRows);
    await loadPendingApprovals();

    if (!trackingBatchId && previewRows.length > 0) {
      generateBtn.disabled = false;
    }
  } catch (e) {
    tableArea.innerHTML = `<div class="empty-state">读取失败: ${escapeHtml(e.message)}</div>`;
  } finally {
    previewBtn.disabled = false;
    previewBtn.textContent = '读取需求表';
  }
}

function renderPreviewTable(rows) {
  if (!rows.length) {
    tableArea.innerHTML = '<div class="empty-state">需求表为空</div>';
    return;
  }

  let html = '<div class="table-scroll"><table class="data-table">';
  html += `<thead><tr>
    <th>#</th>
    <th>商品标题</th>
    <th>平铺图</th>
    <th>穿搭图</th>
    <th>评价数</th>
    <th>晒图数</th>
    <th>状态</th>
    <th>备注</th>
  </tr></thead><tbody>`;

  rows.forEach((row, i) => {
    const statusText = (row.processing_status || '').trim();
    const shouldProcess = !!row.should_process;
    const uiStatusText = shouldProcess ? (statusText || '待处理') : (statusText || '已处理');
    const statusClass = shouldProcess ? 'pending' : 'success';
    const noteText = shouldProcess ? '' : '本轮将跳过';

    html += `<tr id="row-${row.record_id}">
      <td>${i + 1}</td>
      <td class="cell-title" title="${escapeHtml(row.product_title)}">${escapeHtml(row.product_title)}</td>
      <td>${row.has_image
        ? '<span class="has-image-yes">有</span>'
        : '<span class="has-image-no">无</span>'}</td>
      <td>${row.has_outfit_images
        ? `<span class="has-image-yes">${row.outfit_image_count || 0}张</span>`
        : '<span class="has-image-no">无</span>'}</td>
      <td>${row.review_count}</td>
      <td>${row.image_count}</td>
      <td><span class="row-status ${statusClass}">${escapeHtml(uiStatusText)}</span></td>
      <td>${escapeHtml(noteText)}</td>
    </tr>`;
  });

  html += '</tbody></table></div>';
  tableArea.innerHTML = html;
}

function rememberBatch(batchId) {
  trackingBatchId = batchId || '';
  if (batchId) localStorage.setItem(ACTIVE_BATCH_KEY, batchId);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function restoreActiveBatch() {
  let batchId = localStorage.getItem(ACTIVE_BATCH_KEY) || '';
  try {
    const resp = await fetch('/batch-generate-v2/current');
    if (resp.ok) {
      const current = await resp.json();
      if (current.batch_id && (current.status === 'running' || !batchId)) {
        batchId = current.batch_id;
      }
    }
  } catch (e) {
    if (!batchId) return;
  }
  if (!batchId) return;
  rememberBatch(batchId);
  if (!previewRows.length) {
    await loadPreview();
  }
  await followBatch(batchId);
}

async function startBatchGenerate() {
  previewBtn.disabled = true;
  generateBtn.disabled = true;
  progressArea.classList.remove('hidden');
  progressBar.style.width = '8%';
  progressText.textContent = '正在开始批量生成...';
  setBadge('running', '生成中...');

  try {
    const resp = await fetch('/batch-generate-v2', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
    if (!data.batch_id) throw new Error('没有拿到批次号');
    rememberBatch(data.batch_id);
    await followBatch(data.batch_id);
  } catch (e) {
    progressText.textContent = `批量生成失败: ${e.message}`;
    setBadge('error', '失败');
    previewBtn.disabled = false;
    generateBtn.disabled = previewRows.length === 0;
  }
}

async function followBatch(batchId) {
  rememberBatch(batchId);
  previewBtn.disabled = true;
  generateBtn.disabled = true;
  progressArea.classList.remove('hidden');
  setBadge('running', '生成中...');

  while (trackingBatchId === batchId) {
    let data;
    try {
      const resp = await fetch(`/batch-generate-v2/status?batch_id=${encodeURIComponent(batchId)}`);
      data = await resp.json();
      if (resp.status === 404) {
        localStorage.removeItem(ACTIVE_BATCH_KEY);
        trackingBatchId = '';
        progressText.textContent = '这次生成的进度已经找不到了';
        setBadge('error', '已中断');
        previewBtn.disabled = false;
        generateBtn.disabled = previewRows.length === 0;
        return;
      }
      if (!resp.ok || data.error && data.status !== 'error') {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
    } catch (e) {
      progressText.textContent = `正在重连生成进度: ${e.message}`;
      await sleep(2000);
      continue;
    }

    const total = data.total || previewRows.filter((row) => row.should_process).length || 0;
    const finished = (data.results || []).length;
    if (data.status === 'running') {
      const percent = total ? Math.max(8, Math.round((finished / total) * 100)) : 12;
      progressBar.style.width = `${Math.min(percent, 95)}%`;
      progressText.textContent = total
        ? `正在生成 ${finished}/${total}，刷新页面会保持这个进度`
        : '正在生成，刷新页面会保持这个进度';
      updateTableWithResults(data.results || []);
      await sleep(2000);
      continue;
    }

    requireConfirmation = !!data.require_confirmation;
    progressBar.style.width = '100%';
    if (data.status === 'error') {
      progressText.textContent = data.error || data.message || '批量生成失败';
      setBadge('error', '失败');
    } else {
      const successCount = data.success_count || 0;
      progressText.textContent = requireConfirmation
        ? `完成: ${successCount}/${total} 已生成，等待人工审核`
        : `完成: ${successCount}/${total} 成功`;
      setBadge('completed', requireConfirmation ? `${successCount}/${total} 已生成，待审核图片` : `${successCount}/${total} 成功`);
    }
    updateTableWithResults(data.results || []);
    if (requireConfirmation) await loadPendingApprovals();
    previewBtn.disabled = false;
    generateBtn.disabled = previewRows.length === 0;
    return;
  }
}

function updateTableWithResults(results) {
  results.forEach((r) => {
    const tr = document.getElementById(`row-${r.record_id}`);
    if (!tr) return;

    const cells = tr.querySelectorAll('td');
    const statusCell = cells[6];
    const noteCell = cells[7];

    if (r.status === 'success' || r.status === 'pending_approval') {
      statusCell.innerHTML = r.status === 'pending_approval'
        ? '<span class="row-status pending">待审核</span>'
        : '<span class="row-status success">成功</span>';
      const parts = [];
      if (r.approval_id) {
        parts.push(`待审核(${r.approval_id})`);
      } else if (r.auto_accepted) {
        parts.push('评审通过，已自动收下');
      } else if (r.status === 'success') {
        parts.push('已完成');
      }
      if (r.image_urls && r.image_urls.length) parts.push(`${r.image_urls.length}张晒图`);
      const groups = r.image_attempts || [];
      noteCell.innerHTML = escapeHtml(parts.join(', ') || '完成')
        + (groups.length
          ? ` <button class="btn-link" onclick="toggleRowDetail('${r.record_id}')">${retrySummary(groups)}</button>`
          : '');
      const oldDetail = document.getElementById(`detail-${r.record_id}`);
      if (oldDetail) oldDetail.remove();
      if (groups.length) {
        const detail = document.createElement('tr');
        detail.id = `detail-${r.record_id}`;
        detail.className = 'detail-row hidden';
        detail.innerHTML = `<td colspan="8">${renderAttemptGroups(groups, `row-${r.record_id}`)}</td>`;
        tr.after(detail);
      }
    } else {
      statusCell.innerHTML = '<span class="row-status error">失败</span>';
      noteCell.innerHTML = `<span class="cell-error" title="${escapeHtml(r.error || '')}">${escapeHtml(r.error || '未知错误')}</span>`;
    }
  });
}

function setBadge(type, text) {
  statusBadge.className = `status-badge ${type}`;
  statusBadge.textContent = text;
  statusBadge.classList.remove('hidden');
}

async function loadPendingApprovals() {
  try {
    const resp = await fetch('/delivery-approvals?status=pending&limit=200');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    requireConfirmation = !!data.require_confirmation;
    const approvals = data.approvals || [];
    renderApprovalArea(approvals);
  } catch (e) {
    approvalArea.classList.remove('hidden');
    approvalArea.innerHTML = `<div class="approval-header">待审核图片</div><div class="approval-empty">读取待审核列表失败: ${escapeHtml(e.message)}</div>`;
  }
}

function renderApprovalArea(approvals) {
  if (!requireConfirmation) {
    approvalArea.classList.add('hidden');
    approvalArea.innerHTML = '';
    return;
  }

  approvalArea.classList.remove('hidden');
  if (!approvals.length) {
    approvalArea.innerHTML = '<div class="approval-header">待审核图片</div><div class="approval-empty">暂无待审核任务</div>';
    return;
  }

  let html = `<div class="approval-header">待审核图片（${approvals.length}）</div>`;
  html += '<div class="approval-list">';
  approvals.forEach((item) => {
    const contacts = (item.target_contacts || []).join(', ');
    const reviewHtml = formatReviewText(item.review_text || '');
    const imagePaths = item.image_paths || [];
    const approvalType = (item.extra || {}).approval_type;
    const isImageReview = approvalType === 'image_review';
    const regenerateCount = (item.extra || {}).regenerate_count || 0;

    const attemptGroups = (item.extra || {}).image_attempts || [];
    let imagesHtml = '';
    if (attemptGroups.length > 0) {
      imagesHtml = renderAttemptGroups(attemptGroups, `approval-${item.approval_id}`, isImageReview, item.approval_id, (item.extra || {}).phase === 'await_anchor');
    } else if (imagePaths.length > 0) {
      imagesHtml = '<div class="approval-images">';
      imagePaths.forEach((p, index) => {
        const filename = p.replace(/\\/g, '/').split('/').pop();
        const url = '/generated-images/' + encodeURIComponent(filename);
        const redo = isImageReview
          ? `<label class="redo-toggle"><input type="checkbox" class="redo-check" value="${index}">重做这张</label>`
          : '';
        imagesHtml += `<div class="approval-shot">${redo}<a href="${url}" target="_blank" class="approval-img-link"><img src="${url}" alt="晒图 ${index + 1}" class="approval-img" loading="lazy"></a></div>`;
      });
      imagesHtml += '</div>';
    }

    // 判断是否有生成失败的标记
    const hasError = (item.extra || {}).image_generation_failed;
    const errorBadge = hasError ? '<span class="error-badge">生成失败</span>' : '';

    html += `<div class="approval-card" id="approval-${item.approval_id}" data-image-review="${isImageReview ? '1' : '0'}" data-phase="${escapeHtml((item.extra || {}).phase || '')}">
      <div class="approval-card-header">
        <div>
          <div class="approval-title">${escapeHtml(item.product_title || '未命名商品')}${errorBadge}</div>
          <div class="approval-meta">联系人：${escapeHtml(contacts || '-')} ｜ 图片：${imagePaths.length} 张${regenerateCount ? ` ｜ 已重生成 ${regenerateCount} 次` : ''}</div>
        </div>
        <div class="approval-actions">
          ${(item.extra || {}).phase === 'await_anchor' ? '' : `<button class="btn btn-primary btn-sm" onclick="confirmApproval('${item.approval_id}')" ${hasError ? 'disabled' : ''}>审核通过</button>`}
          <button class="btn btn-secondary btn-sm" onclick="rejectApproval('${item.approval_id}')">${isImageReview ? '重做选中' : '驳回'}</button>
        </div>
      </div>
      ${imagesHtml}
      <div class="approval-reviews">${reviewHtml || '<span class="approval-empty-hint">（无评价内容）</span>'}</div>
      ${hasError ? '<div class="approval-error-hint">⚠️ 晒图生成失败，建议驳回后重新批量生成</div>' : ''}
    </div>`;
  });
  html += '</div>';
  approvalArea.innerHTML = html;
}

function formatReviewText(text) {
  if (!text) return '';
  const blocks = text.split(/\n{2,}/);
  let html = '';
  blocks.forEach((block) => {
    const trimmed = block.trim();
    if (!trimmed) return;
    const isHeading = /^(评价\s*\d|第?\d+[条）\)]|---)/i.test(trimmed);
    const lines = trimmed.split('\n').map(l => escapeHtml(l.trim())).filter(Boolean);
    if (isHeading && lines.length === 1) {
      html += `<div class="review-heading">${lines[0]}</div>`;
    } else {
      html += `<div class="review-block">${lines.join('<br>')}</div>`;
    }
  });
  return html;
}

async function confirmApproval(approvalId) {
  const note = window.prompt('审核通过备注（可选）', '');
  if (note === null) return;
  setCardBusy(approvalId, '正在提交审核并写回飞书，请稍等');
  await approveAction(approvalId, 'confirm', note);
}

async function confirmScene(approvalId, sceneIndex) {
  const note = window.prompt('备注（可选，取消则不生成）', '');
  if (note === null) return;
  setCardBusy(approvalId, `正在生成场景 ${sceneIndex + 1} 的另外两张，其他场景不动`);
  await approveAction(approvalId, 'confirm', note, [], sceneIndex);
}

function selectedImageIndexes(approvalId) {
  return Array.from(document.querySelectorAll(`#approval-${approvalId} .redo-check:checked`))
    .map((el) => Number(el.value))
    .filter((n) => Number.isInteger(n));
}

async function rejectApproval(approvalId) {
  const card = document.getElementById(`approval-${approvalId}`);
  const isImageReview = card && card.dataset.imageReview === '1';
  const imageIndexes = isImageReview ? selectedImageIndexes(approvalId) : [];
  if (isImageReview && imageIndexes.length === 0) {
    alert('请先勾选要重做的图片。没勾选的会保留。');
    return;
  }
  const note = window.prompt(isImageReview ? '重做备注（可选，取消则不重做）' : '驳回原因（可选）', '');
  if (note === null) return;
  const labels = imageIndexes.map((index) => `第 ${index + 1} 张`).join('、');
  setCardBusy(
    approvalId,
    isImageReview ? `正在重做${labels}，生成和评审需要一两分钟，页面会停在这里` : '正在驳回，请稍等',
    imageIndexes,
  );
  await approveAction(approvalId, 'reject', note, imageIndexes);
}

function setCardBusy(approvalId, message, imageIndexes) {
  const card = document.getElementById(`approval-${approvalId}`);
  if (!card) return;
  card.classList.add('is-busy');
  let bar = card.querySelector('.card-busy');
  if (!bar) {
    bar = document.createElement('div');
    bar.className = 'card-busy';
    const header = card.querySelector('.approval-card-header');
    header.insertAdjacentElement('afterend', bar);
  }
  bar.textContent = message;
  card.querySelectorAll('button, input').forEach((el) => { el.disabled = true; });
  const redoBtn = card.querySelector('.approval-actions .btn-secondary');
  if (redoBtn && (imageIndexes || []).length) redoBtn.textContent = '正在重做...';
  (imageIndexes || []).forEach((index) => {
    const box = card.querySelector(`.redo-check[value="${index}"]`);
    const shot = box && box.closest('.shot-card, .approval-shot');
    if (shot) shot.classList.add('is-redoing');
  });
  progressArea.classList.remove('hidden');
  progressBar.style.width = '45%';
  progressText.textContent = message;
  setBadge('running', '处理中...');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function approveAction(approvalId, action, note, imageIndexes, sceneIndex) {
  try {
    const body = { note, image_indexes: imageIndexes || [] };
    if (Number.isInteger(sceneIndex)) body.scene_index = sceneIndex;
    const resp = await fetch(`/delivery-approvals/${approvalId}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    progressBar.style.width = '100%';
    progressText.textContent = Number.isInteger(sceneIndex)
      ? `场景 ${sceneIndex + 1} 已补上，其他场景没有改`
      : (action === 'reject' ? '重做完成，请再看新图' : '审核已提交');
    setBadge('completed', Number.isInteger(sceneIndex) ? '已补这一组' : (action === 'reject' ? '重做完成' : '已提交'));
    await loadPendingApprovals();
  } catch (e) {
    alert(`操作失败: ${e.message}`);
    progressText.textContent = `操作失败: ${e.message}`;
    setBadge('error', '失败');
    await loadPendingApprovals();
  }
}

const COLLAPSE_EQUAL_OVER = 80;
const COLLAPSE_KEEP = 24;

function imageSrcOf(attempt) {
  const path = (attempt && attempt.image_path) || '';
  if (path) {
    const filename = path.replace(/\\/g, '/').split('/').pop();
    return '/generated-images/' + encodeURIComponent(filename);
  }
  return (attempt && attempt.image_url) || '';
}

function reviewOf(attempt) {
  return (attempt && attempt.review) || {};
}

function retrySummary(groups) {
  const retried = groups.filter((g) => (g.attempts || []).filter((a) => !a.from_previous_round).length > 1).length;
  return retried ? `查看生成过程（${retried} 张重做过）` : '查看生成过程';
}

function groupStatus(group) {
  const attempts = group.attempts || [];
  const chosen = attempts[group.chosen] || {};
  const review = reviewOf(chosen);
  const newRounds = attempts.filter((a) => !a.from_previous_round).length;
  const redoText = newRounds > 1 ? `重做 ${newRounds - 1} 次` : '一次生成';
  if (review.skipped) return { cls: 'warn', text: `${redoText} · 未完成评审` };
  if (review.pass) return { cls: 'ok', text: `${redoText} · 已通过` };
  return { cls: 'bad', text: `${redoText} · 仍需人工` };
}

function renderAttemptGroups(groups, scope, redo, approvalId, allowFill) {
  const hasScene = groups.some((group) => Number.isInteger(group.scene_index));
  if (!hasScene) return renderShotGrid(groups, scope, redo);
  const blocks = [];
  groups.forEach((group) => {
    let block = blocks.find((item) => item.scene === group.scene_index);
    if (!block) {
      block = { scene: group.scene_index, groups: [] };
      blocks.push(block);
    }
    block.groups.push(group);
  });
  return blocks.map((block) => {
    const onlyAnchor = allowFill && block.groups.length === 1 && block.groups[0].shot_index === 0;
    const hint = onlyAnchor
      ? `<div class="scene-hint">只根据场景 ${block.scene + 1} 的这张图，生成同组另外两张。其他场景不会动。</div>
         <button type="button" class="btn btn-primary btn-sm scene-fill" onclick="confirmScene('${approvalId}', ${block.scene})">生成场景 ${block.scene + 1} 的另外两张</button>`
      : '';
    return `<div class="scene-block"><div class="scene-title">场景 ${block.scene + 1}</div>${hint}${renderShotGrid(block.groups, `${scope}-s${block.scene}`, redo)}</div>`;
  }).join('');
}

function renderShotGrid(groups, scope, redo) {
  let grid = '<div class="shot-grid">';
  let panels = '';
  groups.forEach((group, gi) => {
    const attempts = group.attempts || [];
    const chosen = attempts[group.chosen] || attempts[attempts.length - 1] || {};
    const src = imageSrcOf(chosen);
    const status = groupStatus(group);
    const panelId = `cmp-${scope}-${gi}`;
    const canCompare = attempts.length > 1;
    const imageIndex = Number.isInteger(group.index) ? group.index : gi;
    const redoHtml = redo
      ? `<label class="redo-toggle"><input type="checkbox" class="redo-check" value="${imageIndex}">重做这张</label>`
      : '';
    grid += `<div class="shot-card">
      ${redoHtml}
      <a href="${escapeHtml(src)}" target="_blank" class="approval-img-link">
        <img src="${escapeHtml(src)}" alt="第 ${gi + 1} 张" class="approval-img" loading="lazy">
        <span class="shot-badge ${status.cls}">${escapeHtml(status.text)}</span>
      </a>
      <div class="shot-foot">
        <span>第 ${gi + 1} 张 · ${reviewOf(chosen).score ?? 0} 分</span>
        ${canCompare ? `<button class="btn-link" onclick="toggleCompare('${panelId}', this)">对比两次</button>` : ''}
      </div>
    </div>`;
    if (canCompare) panels += renderComparePanel(attempts, panelId, gi);
  });
  grid += '</div>';
  return grid + panels;
}

function renderComparePanel(attempts, panelId, gi) {
  const lastPair = attempts.length - 1;
  let tabs = '';
  if (attempts.length > 2) {
    tabs = '<div class="cmp-tabs">';
    for (let i = 1; i < attempts.length; i += 1) {
      tabs += `<button type="button" class="cmp-tab ${i === lastPair ? 'active' : ''}" onclick="switchPair('${panelId}', ${i}, this)">${attemptLabel(attempts[i - 1], i - 1)} → ${attemptLabel(attempts[i], i)}</button>`;
    }
    tabs += '</div>';
  }
  let pairs = '';
  for (let i = 1; i < attempts.length; i += 1) {
    pairs += `<div class="cmp-pair ${i === lastPair ? '' : 'hidden'}" data-pair="${i}">
      ${renderPair(attempts[i - 1], attempts[i], i - 1, i)}
    </div>`;
  }
  return `<div id="${panelId}" class="compare-panel hidden">
    <div class="cmp-head">
      <span class="cmp-title">第 ${gi + 1} 张的生成过程</span>
      <label class="cmp-toggle"><input type="checkbox" onchange="toggleFullPrompt('${panelId}', this.checked)">看全文</label>
    </div>
    ${tabs}
    ${pairs}
  </div>`;
}

function attemptLabel(attempt, index) {
  return attempt.from_previous_round ? '上一轮' : `第 ${index + 1} 次`;
}

function renderPair(before, after, bi, ai) {
  const added = after.added_constraints || [];
  const diff = (after.diff && after.diff.length)
    ? after.diff
    : [{ op: 'equal', text: before.prompt || '' }, { op: 'insert', text: (after.prompt || '').slice((before.prompt || '').length) }];
  return `<div class="cmp-shots">
      ${renderAttemptColumn(before, bi)}
      <div class="cmp-arrow">→</div>
      ${renderAttemptColumn(after, ai)}
    </div>
    <div class="cmp-rules">
      <div class="cmp-section-title">${attemptLabel(after, ai)}新增的约束（${added.length} 条）</div>
      ${added.length
        ? `<ol>${added.map((c) => `<li>${escapeHtml(c)}</li>`).join('')}</ol>`
        : '<div class="cmp-empty">没有追加约束</div>'}
    </div>
    <div class="cmp-section-title">提示词变化 <span class="cmp-legend"><ins>新增</ins><del>删除</del></span></div>
    <div class="diff-view compact">${renderDiff(diff, true)}</div>
    <div class="diff-view full">${renderDiff(diff, false)}</div>`;
}

function renderAttemptColumn(attempt, index) {
  const review = reviewOf(attempt);
  const src = imageSrcOf(attempt);
  const problems = review.problems || [];
  const verdict = review.skipped
    ? `<span class="verdict warn">未评审</span>`
    : (review.pass ? '<span class="verdict ok">通过</span>' : '<span class="verdict bad">未通过</span>');
  return `<div class="cmp-col">
    <div class="cmp-col-head">${escapeHtml(attemptLabel(attempt, index))} ${verdict} <span class="cmp-score">${review.score ?? 0} 分</span></div>
    ${src
      ? `<a href="${escapeHtml(src)}" target="_blank" class="cmp-img-link"><img src="${escapeHtml(src)}" alt="${escapeHtml(attemptLabel(attempt, index))}" loading="lazy"></a>`
      : '<div class="cmp-img-missing">图片未生成</div>'}
    ${problems.length
      ? `<ul class="cmp-problems">${problems.map((p) => `<li>${escapeHtml(p)}</li>`).join('')}</ul>`
      : `<div class="cmp-empty">${escapeHtml(review.error || '评审没有指出问题')}</div>`}
  </div>`;
}

function renderDiff(segments, compact) {
  return segments.map((seg) => {
    const text = seg.text || '';
    if (seg.op === 'insert') return `<ins>${escapeHtml(text)}</ins>`;
    if (seg.op === 'delete') return `<del>${escapeHtml(text)}</del>`;
    if (compact && text.length > COLLAPSE_EQUAL_OVER) {
      const hidden = text.length - COLLAPSE_KEEP * 2;
      return `<span>${escapeHtml(text.slice(0, COLLAPSE_KEEP))}</span>`
        + `<span class="diff-fold">…未改动 ${hidden} 字…</span>`
        + `<span>${escapeHtml(text.slice(-COLLAPSE_KEEP))}</span>`;
    }
    return `<span>${escapeHtml(text)}</span>`;
  }).join('');
}

function toggleCompare(panelId, btn) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  const opening = panel.classList.contains('hidden');
  panel.classList.toggle('hidden', !opening);
  if (btn) btn.textContent = opening ? '收起对比' : '对比两次';
  if (opening) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function toggleFullPrompt(panelId, full) {
  document.querySelectorAll(`#${panelId} .diff-view`).forEach((el) => {
    el.classList.toggle('show-full', full);
  });
}

function switchPair(panelId, pairIndex, btn) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.querySelectorAll('.cmp-pair').forEach((el) => {
    el.classList.toggle('hidden', Number(el.dataset.pair) !== pairIndex);
  });
  panel.querySelectorAll('.cmp-tab').forEach((el) => el.classList.toggle('active', el === btn));
}

function toggleRowDetail(recordId) {
  const row = document.getElementById(`detail-${recordId}`);
  if (row) row.classList.toggle('hidden');
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
