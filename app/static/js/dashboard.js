(() => {
  const $ = (id) => document.getElementById(id);

  const fileInput = $('cycle-wav-input');
  const chooseButton = $('choose-wav-btn');
  const runButton = $('run-inference-btn');
  const subjectInput = $('subject-id-input');
  const fileNameLabel = $('selected-file-name');
  const statusLabel = $('analysis-status');
  const historyBody = $('measurement-history-body');
  const longitudinalChart = $('longitudinal-chart');
  const componentChart = $('component-chart');
  const exportCsvTop = $('export-csv-btn-top');
  const exportCsvHistory = $('export-csv-btn-history');
  const exportJsonHistory = $('export-json-btn-history');
  const subjectSelect = $('subject-select');
  const loadSubjectBtn = $('load-subject-btn');
  const clearSubjectBtn = $('clear-subject-btn');

  let monitoringSeries = [];
  try {
    const initialData = $('initial-monitoring-data');
    monitoringSeries = initialData ? JSON.parse(initialData.textContent || '[]') : [];
  } catch (error) {
    console.warn('Initial monitoring history parse failed:', error);
    monitoringSeries = [];
  }

  if (!fileInput || !chooseButton || !runButton) return;

  const fmt3 = (value) => Number(value ?? 0).toFixed(3);
  const fmt4Signed = (value) => {
    const n = Number(value ?? 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(4)}`;
  };
  const fmt3Signed = (value) => {
    const n = Number(value ?? 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(3)}`;
  };
  const pct1 = (value) => `${(Number(value ?? 0) * 100).toFixed(1)}%`;

  function setStatus(text, isError = false) {
    if (!statusLabel) return;
    statusLabel.textContent = text;
    statusLabel.classList.toggle('text-error', isError);
    statusLabel.classList.toggle('text-on-tertiary-container', !isError);
  }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
  }

  const svgNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs = {}, text = null) {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    if (text !== null) el.textContent = text;
    return el;
  }

  function chartPoints(values, width, height, left, right, top, bottom, yMax = 1.0) {
    if (!values.length) return [];
    const innerW = width - left - right;
    const innerH = height - top - bottom;
    return values.map((v, i) => {
      const x = values.length === 1 ? left + innerW / 2 : left + (i * innerW) / (values.length - 1);
      const y = top + (1 - Math.max(0, Math.min(yMax, Number(v ?? 0))) / yMax) * innerH;
      return [x, y];
    });
  }

  function pathFromPoints(points) {
    if (!points.length) return '';
    return points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  }


  function melColor(v) {
    const x = Math.max(0, Math.min(1, Number(v || 0)));
    // Dark navy -> blue -> cyan -> pale yellow.
    if (x < 0.33) {
      const t = x / 0.33;
      return [Math.round(5 + 8*t), Math.round(13 + 45*t), Math.round(28 + 95*t)];
    }
    if (x < 0.66) {
      const t = (x - 0.33) / 0.33;
      return [Math.round(13 + 25*t), Math.round(58 + 120*t), Math.round(123 + 105*t)];
    }
    const t = (x - 0.66) / 0.34;
    return [Math.round(38 + 217*t), Math.round(178 + 70*t), Math.round(228 - 120*t)];
  }

  function camColor(v) {
    const x = Math.max(0, Math.min(1, Number(v || 0)));
    if (x < 0.5) {
      const t = x / 0.5;
      return [Math.round(245), Math.round(158 + 55*t), Math.round(11 - 11*t)];
    }
    const t = (x - 0.5) / 0.5;
    return [Math.round(245 - 6*t), Math.round(213 - 145*t), Math.round(0 + 68*t)];
  }

  function drawMatrix(canvas, matrix, overlay = null) {
    if (!canvas || !Array.isArray(matrix) || !matrix.length || !matrix[0]?.length) return;
    const rows = matrix.length;
    const cols = matrix[0].length;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d', { alpha: false });
    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r += 1) {
      const srcR = rows - 1 - r; // low frequencies at the bottom
      for (let c = 0; c < cols; c += 1) {
        const idx = (r * cols + c) * 4;
        let [rr, gg, bb] = melColor(matrix[srcR][c]);
        if (overlay && overlay[srcR] && overlay[srcR][c] != null) {
          const cam = Math.max(0, Math.min(1, Number(overlay[srcR][c])));
          const [cr, cg, cb] = camColor(cam);
          // Suppress low-level CAM noise and make truly active regions easier to distinguish
          // from the underlying Mel-Spectrogram. This is visualization-only.
          const a = cam < 0.12 ? 0 : Math.pow((cam - 0.12) / 0.88, 0.62) * 0.88;
          rr = Math.round(rr * (1 - a) + cr * a);
          gg = Math.round(gg * (1 - a) + cg * a);
          bb = Math.round(bb * (1 - a) + cb * a);
        }
        img.data[idx] = rr;
        img.data[idx + 1] = gg;
        img.data[idx + 2] = bb;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    canvas.style.imageRendering = 'auto';
  }

  function updateXAI(data) {
    const xai = data.xai;
    if (!xai) return;
    const melCanvas = $('mel-spectrogram-canvas');
    const camCanvas = $('gradcam-canvas');
    drawMatrix(melCanvas, xai.mel);
    drawMatrix(camCanvas, xai.mel, xai.cam);
    const melEmpty = $('mel-empty-state');
    const camEmpty = $('cam-empty-state');
    if (melEmpty) melEmpty.style.display = 'none';
    if (camEmpty) camEmpty.style.display = 'none';

    setText('active-cam-record-id', data.measurement_id);
    setText('xai-layer-badge', `기여 레이어: ${xai.target_layer}`);
    setText('xai-layer-footer', `${xai.target_layer} Grad-CAM`);
    setText('xai-config-badge', `${Number(xai.sample_rate).toLocaleString()} Hz · ${Number(xai.duration_sec).toFixed(1)} sec · N_FFT ${xai.n_fft} · Hop ${xai.hop_length}`);
    setText('mel-axis-meta', `${xai.fmin}–${xai.fmax} Hz · 0–${Number(xai.duration_sec).toFixed(1)} sec`);
    setText('mel-params', `N_MELS ${xai.n_mels} · N_FFT ${xai.n_fft} · Hop ${xai.hop_length}`);
    setText('xai-target-class', `목표 클래스: ${xai.target_class} (p=${Number(xai.target_probability).toFixed(3)})`);
    setText('xai-target-footer', `예측 클래스 ${xai.target_class} 기준 역전파`);
  }

  function renderLongitudinalChart() {
    if (!longitudinalChart) return;
    longitudinalChart.replaceChildren();
    const W = 920, H = 280, L = 55, R = 30, T = 25, B = 48;
    const series = monitoringSeries;

    [1.0, 0.75, 0.5, 0.25, 0.0].forEach((v) => {
      const y = T + (1 - v) * (H - T - B);
      longitudinalChart.appendChild(svgEl('line', {x1:L, x2:W-R, y1:y, y2:y, stroke:'#CBD5E1', 'stroke-width':0.75, 'stroke-dasharray': v===0 ? '0' : '3 3'}));
      longitudinalChart.appendChild(svgEl('text', {x:22, y:y+4, fill:'#64748B', 'font-family':'JetBrains Mono', 'font-size':10}, v.toFixed(2)));
    });

    if (!series.length) {
      longitudinalChart.appendChild(svgEl('text', {x:W/2, y:H/2, fill:'#64748B', 'text-anchor':'middle', 'font-family':'Inter', 'font-size':14}, '측정 데이터가 아직 없습니다.'));
      return;
    }

    const vals = series.map(d => Number(d.abnormal_probability));
    const pts = chartPoints(vals, W, H, L, R, T, B, 1.0);

    const calibrationCount = Math.min(3, series.length);
    if (calibrationCount > 0) {
      const firstX = pts[0][0];
      const lastX = pts[calibrationCount - 1][0];
      const halfStep = series.length > 1 ? (pts[1][0] - pts[0][0]) / 2 : 30;
      const x = Math.max(L, firstX - halfStep);
      const width = Math.max(20, Math.min(W-R, lastX + halfStep) - x);
      longitudinalChart.appendChild(svgEl('rect', {x, y:T, width, height:H-T-B, rx:4, fill:'#CBD5E1', 'fill-opacity':0.28}));
      longitudinalChart.appendChild(svgEl('text', {x:x+8, y:T+16, fill:'#475569', 'font-family':'Inter', 'font-size':10, 'font-weight':600}, `기준선 설정 구간 (${calibrationCount}/3)`));
    }

    const baseline = series.length >= 3 ? Number(series[series.length - 1].baseline) : null;
    if (Number.isFinite(baseline)) {
      const y = T + (1 - baseline) * (H - T - B);
      longitudinalChart.appendChild(svgEl('line', {x1:L, x2:W-R, y1:y, y2:y, stroke:'#475569', 'stroke-width':1.5, 'stroke-dasharray':'4 4'}));
      longitudinalChart.appendChild(svgEl('text', {x:W-R-4, y:y-5, fill:'#475569', 'text-anchor':'end', 'font-family':'JetBrains Mono', 'font-size':10}, `기준선 ${baseline.toFixed(3)}`));
    }

    longitudinalChart.appendChild(svgEl('path', {d:pathFromPoints(pts), fill:'none', stroke:'#0F172A', 'stroke-width':2.5}));

    const rollingVals = series.map(d => d.rolling_mean == null ? null : Number(d.rolling_mean));
    const validRolling = rollingVals.map((v, i) => v == null ? null : [i, v]).filter(Boolean);
    if (validRolling.length >= 2) {
      const innerW = W-L-R, innerH=H-T-B;
      const rPts = validRolling.map(([i,v]) => {
        const x = series.length===1 ? L+innerW/2 : L + (i*innerW)/(series.length-1);
        const y = T + (1-Math.max(0,Math.min(1,v))) * innerH;
        return [x,y];
      });
      longitudinalChart.appendChild(svgEl('path', {d:pathFromPoints(rPts), fill:'none', stroke:'#2563EB', 'stroke-width':2, 'stroke-dasharray':'6 4'}));
    }

    pts.forEach((p, i) => {
      const active = i === pts.length - 1;
      longitudinalChart.appendChild(svgEl('circle', {cx:p[0], cy:p[1], r:active?6:4, fill:active?'#2563EB':'#0F172A', stroke:'#FFFFFF', 'stroke-width':1.5}));
      longitudinalChart.appendChild(svgEl('text', {x:p[0], y:H-25, fill:active?'#0F172A':'#64748B', 'text-anchor':'middle', 'font-family':'JetBrains Mono', 'font-size':9, 'font-weight':active?700:400}, `M${String(i+1).padStart(2,'0')}`));
    });

    const last = series[series.length - 1];
    const lastP = pts[pts.length - 1];
    const labelX = Math.max(L+8, Math.min(W-220, lastP[0]-155));
    const labelY = Math.max(8, lastP[1]-58);
    longitudinalChart.appendChild(svgEl('rect', {x:labelX, y:labelY, width:205, height:43, rx:6, fill:'#0F172A'}));
    longitudinalChart.appendChild(svgEl('text', {x:labelX+10, y:labelY+17, fill:'#FFFFFF', 'font-family':'JetBrains Mono', 'font-size':10, 'font-weight':600}, `${last.measurement_id}: ${Number(last.abnormal_probability).toFixed(3)}`));
    longitudinalChart.appendChild(svgEl('text', {x:labelX+10, y:labelY+33, fill:'#93C5FD', 'font-family':'Inter', 'font-size':9}, last.deviation == null ? '기준선 형성 중' : `기준선 대비 Δ ${Number(last.deviation)>=0?'+':''}${Number(last.deviation).toFixed(3)}`));
  }

  function renderComponentChart() {
    if (!componentChart) return;
    componentChart.replaceChildren();
    const W=920,H=140,L=55,R=30,T=12,B=25;
    const series=monitoringSeries;
    [1.0,0.75,0.5,0.25,0.0].forEach(v=>{
      const y=T+(1-v)*(H-T-B);
      componentChart.appendChild(svgEl('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:'#CBD5E1','stroke-width':0.65,'stroke-dasharray':v===0?'0':'3 3'}));
      componentChart.appendChild(svgEl('text',{x:22,y:y+3,fill:'#64748B','font-family':'JetBrains Mono','font-size':8},v.toFixed(2)));
    });
    if(!series.length){
      componentChart.appendChild(svgEl('text',{x:W/2,y:H/2,fill:'#64748B','text-anchor':'middle','font-family':'Inter','font-size':12},'측정 데이터가 아직 없습니다.'));
      return;
    }
    const configs=[
      ['crackle','#F97316',2.4],['wheeze','#2563EB',2.1],['both','#374151',1.9]
    ];
    configs.forEach(([key,color,width])=>{
      const vals=series.map(d=>Number(d[key] ?? 0));
      const pts=chartPoints(vals,W,H,L,R,T,B,1.0);
      componentChart.appendChild(svgEl('path',{d:pathFromPoints(pts),fill:'none',stroke:color,'stroke-width':width}));
      const last=pts[pts.length-1];
      componentChart.appendChild(svgEl('circle',{cx:last[0],cy:last[1],r:3,fill:color,stroke:'#FFFFFF','stroke-width':1}));
    });
  }

  function updateChartMetricsFromMonitoring(m) {
    setText('chart-baseline-value', m.baseline_ready ? fmt3(m.baseline) : '—');
    setText('chart-rolling-value', m.rolling_mean == null ? '—' : fmt3(m.rolling_mean));
    setText('chart-current-value', fmt3(m.abnormal_probability));
    setText('chart-deviation-value', m.deviation == null ? '—' : fmt3Signed(m.deviation));
    setText('chart-trend-value', m.baseline_ready ? m.trend_direction_ko : '산출 대기');
    setText('baseline-legend-text', m.baseline_ready ? `개인 기준선 (Subject Baseline: ${fmt3(m.baseline)})` : '개인 기준선 (형성 중)');
    const note = $('chart-baseline-note');
    if (note) {
      note.innerHTML = `<span class="font-semibold block text-on-surface mb-0.5">개인 기준선: 초기 3회 측정값의 평균</span>${m.baseline_ready ? '초기 3회 기준선이 형성되었습니다. 이후 측정치는 해당 기준선과 비교하여 표시합니다.' : `현재 ${m.measurement_count}/3회 측정 완료. 기준선 형성 전에는 편차와 추세를 해석하지 않습니다.`}`;
    }
  }

  function appendMonitoringSeries(data) {
    const p=data.prediction, m=data.monitoring, probs=p.probabilities;
    monitoringSeries.push({
      measurement_id:data.measurement_id,
      timestamp:m.timestamp,
      predicted_class:p.predicted_class,
      confidence:p.confidence,
      normal:probs.Normal,
      crackle:probs.Crackle,
      wheeze:probs.Wheeze,
      both:probs.Both,
      abnormal_probability:p.abnormal_probability,
      baseline:m.baseline,
      rolling_mean:m.rolling_mean,
      deviation:m.deviation,
    });
    renderLongitudinalChart();
    renderComponentChart();
    updateChartMetricsFromMonitoring(m);
  }

  function updateMetricCards(data) {
    const p = data.prediction;
    const m = data.monitoring;

    setText('predicted-class', p.predicted_class);
    setText('confidence-badge', `신뢰도: ${pct1(p.confidence)}`);
    setText('confidence-prob', `(p=${fmt3(p.confidence)})`);
    setText('measurement-id-card', data.measurement_id);
    setText('measurement-time-card', m.timestamp);
    setText('active-measurement-badge', `활성: ${data.measurement_id}`);

    setText('abnormal-probability', fmt3(p.abnormal_probability));
    const bar = $('abnormal-probability-bar');
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, p.abnormal_probability * 100))}%`;

    if (m.baseline_ready) {
      setText('baseline-status-badge', '기준선 형성 완료');
      setText('baseline-deviation', fmt3Signed(m.deviation));
      setText('baseline-deviation-label', `개인 기준선 대비 (${fmt3Signed(m.deviation)})`);
      setText('baseline-summary', `개인 기준선: ${fmt3(m.baseline)} · 최근 3회 이동평균: ${fmt3(m.rolling_mean)}`);
      setText('trend-direction', `${m.trend_direction_ko} (${m.trend_direction})`);
      const slopeUnit = m.recent_slope_unit === 'day' ? 'day' : 'measurement';
      setText('trend-summary', `최근 slope: ${fmt4Signed(m.recent_slope)}/${slopeUnit} · 이동평균: ${fmt3(m.rolling_mean)}`);
    } else {
      setText('baseline-status-badge', `기준선 형성 중 (${m.measurement_count}/3)`);
      setText('baseline-deviation', '—');
      setText('baseline-deviation-label', '초기 3회 측정 후 계산');
      setText('baseline-summary', `기준선 형성 중: ${m.measurement_count}/3회 완료 · ${m.measurements_to_baseline}회 추가 측정 필요`);
      setText('trend-direction', '산출 대기');
      setText('trend-summary', '초기 3회 측정 완료 후 최근 변화 방향을 계산합니다.');
    }
  }

  function addHistoryRow(data) {
    if (!historyBody) return;
    const p = data.prediction;
    const m = data.monitoring;
    const probs = p.probabilities;

    const tr = document.createElement('tr');
    tr.className = 'measurement-row hover:bg-surface-container-low cursor-pointer transition-colors';
    tr.dataset.measurementId = data.measurement_id;

    const cells = [
      { html: '<span class="row-indicator w-3.5 h-3.5 rounded-full bg-secondary inline-block"></span>', cls: 'py-space-sm px-space-md text-center' },
      { text: data.measurement_id, cls: 'py-space-sm px-space-md font-code-badge text-code-badge text-on-surface' },
      { text: m.timestamp, cls: 'py-space-sm px-space-md text-on-surface-variant' },
      { text: p.predicted_class, cls: 'py-space-sm px-space-md' },
      { text: pct1(p.confidence), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right text-on-surface' },
      { text: fmt3(probs.Normal), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right' },
      { text: fmt3(probs.Crackle), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right' },
      { text: fmt3(probs.Wheeze), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right' },
      { text: fmt3(probs.Both), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right' },
      { text: fmt3(p.abnormal_probability), cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right font-bold' },
      { text: m.baseline_ready ? fmt3Signed(m.deviation) : '—', cls: 'py-space-sm px-space-md font-metric-sm text-metric-sm text-right text-secondary font-semibold' },
      { html: `<div class="flex items-center justify-center gap-1"><button class="view-measurement-btn p-1 rounded hover:bg-surface-container" type="button" data-measurement-id="${data.measurement_id}" title="저장된 측정 정보 확인"><span class="material-symbols-outlined text-on-surface-variant text-[18px]">visibility</span></button><button class="delete-measurement-btn p-1 rounded hover:bg-error-container" type="button" data-measurement-id="${data.measurement_id}" title="이 측정 삭제"><span class="material-symbols-outlined text-error text-[18px]">delete</span></button></div>`, cls: 'py-space-sm px-space-md text-center' },
    ];

    cells.forEach((cell, index) => {
      const td = document.createElement('td');
      td.className = cell.cls;
      if (cell.html) {
        td.innerHTML = cell.html;
      } else if (index === 3) {
        const span = document.createElement('span');
        span.className = 'bg-secondary-fixed text-on-secondary-fixed font-code-badge text-code-badge px-space-xs py-0.5 rounded';
        span.textContent = cell.text;
        td.appendChild(span);
      } else {
        td.textContent = cell.text;
      }
      tr.appendChild(td);
    });

    historyBody.prepend(tr);
    const count = historyBody.querySelectorAll('tr').length;
    setText('observation-count', `${count}회 관측`);
    setText('history-count', `${count}개 관측 기록 등록`);
    setText('history-total-count', `총 ${count}개 측정 기록 · 개인 기준선은 초기 3회 평균`);
  }

  renderLongitudinalChart();
  renderComponentChart();


  function exportSubjectCsv() {
    const subjectId = (subjectInput?.value || '').trim();
    if (!subjectId) {
      setStatus('CSV로 내보낼 대상자 ID를 입력해 주세요.', true);
      return;
    }
    window.location.href = `/api/export/${encodeURIComponent(subjectId)}.csv`;
  }

  function exportSubjectJson() {
    const subjectId = (subjectInput?.value || '').trim();
    if (!subjectId) {
      setStatus('JSON으로 내보낼 대상자 ID를 입력해 주세요.', true);
      return;
    }
    window.location.href = `/api/export/${encodeURIComponent(subjectId)}.json`;
  }

  function loadSelectedSubject() {
    const subjectId = (subjectInput?.value || '').trim();
    if (!subjectId) {
      setStatus('조회할 대상자 ID를 입력하거나 선택해 주세요.', true);
      return;
    }
    window.location.href = `/?subject_id=${encodeURIComponent(subjectId)}`;
  }

  async function deleteMeasurement(measurementId) {
    const subjectId = (subjectInput?.value || '').trim();
    if (!subjectId || !measurementId) return;
    const ok = window.confirm(`${measurementId} 측정 기록을 삭제하시겠습니까?\n삭제 후 baseline과 trend가 다시 계산됩니다.`);
    if (!ok) return;
    setStatus(`${measurementId} 삭제 중...`);
    try {
      const response = await fetch(`/api/subjects/${encodeURIComponent(subjectId)}/measurements/${encodeURIComponent(measurementId)}`, { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      window.location.href = `/?subject_id=${encodeURIComponent(subjectId)}`;
    } catch (error) {
      setStatus(`삭제 실패: ${error.message}`, true);
    }
  }

  async function clearSubjectRecords() {
    const subjectId = (subjectInput?.value || '').trim();
    if (!subjectId) {
      setStatus('초기화할 대상자 ID를 입력해 주세요.', true);
      return;
    }
    const ok = window.confirm(`${subjectId}의 모든 측정 기록을 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.`);
    if (!ok) return;
    setStatus(`${subjectId} 기록 초기화 중...`);
    try {
      const response = await fetch(`/api/subjects/${encodeURIComponent(subjectId)}`, { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      window.location.href = `/?subject_id=${encodeURIComponent(subjectId)}`;
    } catch (error) {
      setStatus(`초기화 실패: ${error.message}`, true);
    }
  }

  exportCsvTop?.addEventListener('click', exportSubjectCsv);
  exportCsvHistory?.addEventListener('click', exportSubjectCsv);
  exportJsonHistory?.addEventListener('click', exportSubjectJson);
  loadSubjectBtn?.addEventListener('click', loadSelectedSubject);
  clearSubjectBtn?.addEventListener('click', clearSubjectRecords);

  subjectSelect?.addEventListener('change', () => {
    if (subjectSelect.value && subjectInput) subjectInput.value = subjectSelect.value;
  });

  historyBody?.addEventListener('click', (event) => {
    const deleteBtn = event.target.closest('.delete-measurement-btn');
    if (deleteBtn) {
      event.preventDefault();
      event.stopPropagation();
      deleteMeasurement(deleteBtn.dataset.measurementId);
      return;
    }
    const viewBtn = event.target.closest('.view-measurement-btn');
    if (viewBtn) {
      event.preventDefault();
      event.stopPropagation();
      const mid = viewBtn.dataset.measurementId;
      const row = viewBtn.closest('tr');
      historyBody.querySelectorAll('tr').forEach((r) => r.classList.remove('bg-surface-container-low'));
      row?.classList.add('bg-surface-container-low');
      setStatus(`저장된 측정 ${mid} 선택됨 · XAI는 현재 추론 측정 기준으로 표시됩니다.`);
    }
  });

  chooseButton.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileNameLabel.textContent = file ? file.name : '선택된 WAV 없음';
    setStatus(file ? '분석 준비 완료' : '');
  });

  runButton.addEventListener('click', async () => {
    const file = fileInput.files?.[0];
    const subjectId = (subjectInput?.value || '').trim();

    if (!file) {
      setStatus('WAV 파일을 먼저 선택해 주세요.', true);
      return;
    }
    if (!subjectId) {
      setStatus('대상자 ID를 입력해 주세요.', true);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('subject_id', subjectId);

    runButton.disabled = true;
    setStatus('AlexNet 추론 중...');

    try {
      const response = await fetch('/api/analyze-cycle', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `HTTP ${response.status}`);
      }

      updateMetricCards(payload);
      addHistoryRow(payload);
      appendMonitoringSeries(payload);
      updateXAI(payload);
      setStatus(`분석 완료: ${payload.prediction.predicted_class}`);
    } catch (error) {
      console.error(error);
      setStatus(`분석 실패: ${error.message}`, true);
    } finally {
      runButton.disabled = false;
    }
  });


  // ---------------- Research batch evaluation (v15) ----------------
  let batchResults = [];
  let lastBatchPayload = null;
  const batchInput = $('batch-wav-input');
  const chooseBatchBtn = $('choose-batch-btn');
  const runBatchBtn = $('run-batch-btn');
  const exportBatchBtn = $('export-batch-csv-btn');
  const batchInferLabels = $('batch-infer-labels');

  function fmtPctNullable(v) {
    return v == null || Number.isNaN(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(1)}%`;
  }

  function renderBatchRows(rows) {
    const body = $('batch-result-body');
    if (!body) return;
    body.replaceChildren();
    if (!rows.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="10" class="px-space-sm py-space-md text-center text-on-surface-variant">성공한 배치 결과가 없습니다.</td>';
      body.appendChild(tr);
      return;
    }
    rows.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'border-t border-surface-container';
      const verdict = r.correct == null ? '—' : (r.correct ? '일치' : '불일치');
      const verdictClass = r.correct == null ? 'text-on-surface-variant' : (r.correct ? 'text-on-tertiary-container font-semibold' : 'text-error font-semibold');
      tr.innerHTML = `
        <td class="px-space-sm py-space-xs max-w-[300px] truncate" title="${escapeHtml(r.filename)}">${escapeHtml(r.filename)}</td>
        <td class="px-space-sm py-space-xs">${escapeHtml(r.true_label || '—')}</td>
        <td class="px-space-sm py-space-xs font-semibold">${escapeHtml(r.predicted_class)}</td>
        <td class="px-space-sm py-space-xs">${(Number(r.confidence)*100).toFixed(1)}%</td>
        <td class="px-space-sm py-space-xs">${Number(r.Normal).toFixed(3)}</td>
        <td class="px-space-sm py-space-xs">${Number(r.Crackle).toFixed(3)}</td>
        <td class="px-space-sm py-space-xs">${Number(r.Wheeze).toFixed(3)}</td>
        <td class="px-space-sm py-space-xs">${Number(r.Both).toFixed(3)}</td>
        <td class="px-space-sm py-space-xs">${Number(r.Abnormal_Probability).toFixed(3)}</td>
        <td class="px-space-sm py-space-xs ${verdictClass}">${verdict}</td>`;
      body.appendChild(tr);
    });
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
  }

  function setBatchSummary(payload) {
    const m = payload.metrics || {};
    setText('batch-success-count', String(payload.success_count ?? 0));
    setText('batch-labeled-count', String(m.labeled_count ?? 0));
    setText('batch-accuracy', fmtPctNullable(m.accuracy));
    setText('batch-balanced', fmtPctNullable(m.balanced_accuracy));
    setText('batch-macro-f1', fmtPctNullable(m.macro_f1));
  }


  function sourceSubjectFromFilename(filename) {
    const m = String(filename || '').match(/^([0-9]+)_/);
    return m ? m[1] : null;
  }

  function renderCohortStatistics(payload) {
    const rows = payload?.results || [];
    const metrics = payload?.metrics || {};
    const classOrder = payload?.class_order || ['Both','Crackle','Normal','Wheeze'];
    if (!rows.length) return;

    const subjects = new Set(rows.map(r => sourceSubjectFromFilename(r.filename)).filter(Boolean));
    const meanAbn = rows.reduce((s,r) => s + Number(r.Abnormal_Probability || 0), 0) / rows.length;
    setText('cohort-file-count', String(rows.length));
    setText('cohort-subject-count', String(subjects.size));
    setText('cohort-labeled-count', String(metrics.labeled_count ?? 0));
    setText('cohort-mean-abn', Number.isFinite(meanAbn) ? meanAbn.toFixed(3) : '—');
    setText('cohort-status', `현재 배치 ${rows.length}개 · source subject ${subjects.size}명`);

    const trueCounts = Object.fromEntries(classOrder.map(c => [c,0]));
    const predCounts = Object.fromEntries(classOrder.map(c => [c,0]));
    rows.forEach(r => {
      if (r.true_label in trueCounts) trueCounts[r.true_label] += 1;
      if (r.predicted_class in predCounts) predCounts[r.predicted_class] += 1;
    });
    const labeledN = Number(metrics.labeled_count || 0);
    const classBody = $('cohort-class-body');
    if (classBody) {
      classBody.replaceChildren();
      classOrder.forEach(c => {
        const tr = document.createElement('tr');
        tr.className = 'border-t border-surface-container';
        const truePct = labeledN ? `${(trueCounts[c]/labeledN*100).toFixed(1)}%` : '—';
        const predPct = rows.length ? `${(predCounts[c]/rows.length*100).toFixed(1)}%` : '—';
        tr.innerHTML = `<td class="px-space-sm py-space-xs font-semibold">${escapeHtml(c)}</td><td class="px-space-sm py-space-xs">${trueCounts[c]}</td><td class="px-space-sm py-space-xs">${truePct}</td><td class="px-space-sm py-space-xs">${predCounts[c]}</td><td class="px-space-sm py-space-xs">${predPct}</td>`;
        classBody.appendChild(tr);
      });
    }

    const perfBody = $('cohort-performance-body');
    if (perfBody) {
      perfBody.replaceChildren();
      classOrder.forEach(c => {
        const p = metrics.per_class?.[c] || {};
        const tr = document.createElement('tr');
        tr.className = 'border-t border-surface-container';
        tr.innerHTML = `<td class="px-space-sm py-space-xs font-semibold">${escapeHtml(c)}</td><td class="px-space-sm py-space-xs">${p.support ?? 0}</td><td class="px-space-sm py-space-xs">${fmtPctNullable(p.precision)}</td><td class="px-space-sm py-space-xs">${fmtPctNullable(p.recall)}</td><td class="px-space-sm py-space-xs">${fmtPctNullable(p.f1)}</td>`;
        perfBody.appendChild(tr);
      });
    }

    const confBody = $('cohort-confusion-body');
    if (confBody) {
      confBody.replaceChildren();
      classOrder.forEach(t => {
        const tr = document.createElement('tr');
        tr.className = 'border-t border-surface-container';
        const cells = classOrder.map(p => `<td class="px-space-sm py-space-xs ${t===p ? 'font-bold bg-secondary-fixed/40' : ''}">${metrics.confusion_matrix?.[t]?.[p] ?? 0}</td>`).join('');
        tr.innerHTML = `<td class="px-space-sm py-space-xs text-left font-semibold">${escapeHtml(t)}</td>${cells}`;
        confBody.appendChild(tr);
      });
    }
  }

  if (chooseBatchBtn && batchInput) {
    chooseBatchBtn.addEventListener('click', () => batchInput.click());
    batchInput.addEventListener('change', () => {
      const n = batchInput.files?.length || 0;
      setText('batch-selected-count', `선택 ${n}개`);
      setText('batch-status', n ? '실행 준비' : '대기');
      if (runBatchBtn) runBatchBtn.disabled = n === 0;
    });
  }

  if (runBatchBtn && batchInput) {
    runBatchBtn.addEventListener('click', async () => {
      const files = Array.from(batchInput.files || []);
      if (!files.length) return;
      runBatchBtn.disabled = true;
      if (exportBatchBtn) exportBatchBtn.disabled = true;
      setText('batch-status', `배치 추론 중… 0/${files.length}`);
      const errorBox = $('batch-error-box');
      if (errorBox) { errorBox.classList.add('hidden'); errorBox.textContent = ''; }
      try {
        const form = new FormData();
        files.forEach((f) => form.append('files', f));
        form.append('infer_labels_from_filename', batchInferLabels?.checked ? 'true' : 'false');
        const started = performance.now();
        const res = await fetch('/api/batch-evaluate', { method: 'POST', body: form });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
        batchResults = payload.results || [];
        lastBatchPayload = payload;
        renderBatchRows(batchResults);
        setBatchSummary(payload);
        renderCohortStatistics(payload);
        setText('batch-status', `완료 · ${payload.success_count}/${payload.file_count} 성공 · ${payload.elapsed_seconds}s`);
        if (exportBatchBtn) exportBatchBtn.disabled = batchResults.length === 0;
        if (payload.errors?.length && errorBox) {
          errorBox.classList.remove('hidden');
          errorBox.textContent = `오류 ${payload.errors.length}건: ` + payload.errors.map(e => `${e.filename}: ${e.error}`).join(' | ');
        }
      } catch (err) {
        setText('batch-status', `실패: ${err.message}`);
        if (errorBox) { errorBox.classList.remove('hidden'); errorBox.textContent = err.message; }
      } finally {
        runBatchBtn.disabled = false;
      }
    });
  }

  if (exportBatchBtn) {
    exportBatchBtn.addEventListener('click', () => {
      if (!batchResults.length) return;
      const cols = ['filename','true_label','predicted_class','confidence','Normal','Crackle','Wheeze','Both','Abnormal_Probability','correct'];
      const q = (v) => `"${String(v ?? '').replaceAll('"','""')}"`;
      const lines = [cols.join(','), ...batchResults.map(r => cols.map(c => q(r[c])).join(','))];
      const blob = new Blob(['\ufeff' + lines.join('\r\n')], {type:'text/csv;charset=utf-8'});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `batch_evaluation_${new Date().toISOString().replaceAll(':','-').slice(0,19)}.csv`;
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
    });
  }

})();
