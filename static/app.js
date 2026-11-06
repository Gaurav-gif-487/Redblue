/**
 * RedBlue AI — Adversarial Execution Intelligence Dashboard
 * Frontend Application Logic (WebSocket + Chart.js)
 */

// Chart.js Default Config (Dark Theme)
Chart.defaults.color = '#8E99A8';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(20, 25, 35, 0.9)';
Chart.defaults.plugins.tooltip.titleFont = { family: 'Inter', size: 13, weight: 'bold' };
Chart.defaults.plugins.tooltip.bodyFont = { family: 'JetBrains Mono', size: 12 };
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;

const CHART_COLORS = {
  red: 'rgb(255, 42, 85)',
  redBg: 'rgba(255, 42, 85, 0.2)',
  cyan: 'rgb(0, 229, 255)',
  cyanBg: 'rgba(0, 229, 255, 0.2)',
  green: 'rgb(0, 230, 118)',
  purple: 'rgb(179, 136, 255)',
  amber: 'rgb(255, 179, 0)',
  grid: 'rgba(255, 255, 255, 0.05)',
};

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 }, // For real-time updates
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: CHART_COLORS.grid }, ticks: { maxTicksLimit: 10 } },
    y: { grid: { color: CHART_COLORS.grid } }
  }
};

// Global State
let ws;
let isConnected = false;
let maxDataPoints = 50;

// Chart Instances
const charts = {};

// DOM Elements
const DOM = {
  // Navigation
  navItems: document.querySelectorAll('.nav-item'),
  sections: document.querySelectorAll('.section'),
  
  // Controls
  btnStart: document.getElementById('btn-start'),
  btnStop: document.getElementById('btn-stop'),
  btnReset: document.getElementById('btn-reset'),
  speedSlider: document.getElementById('speed-slider'),
  speedLabel: document.getElementById('speed-label'),
  aggSlider: document.getElementById('aggression-slider'),
  aggLabel: document.getElementById('aggression-label'),
  intSlider: document.getElementById('intensity-slider'),
  intLabel: document.getElementById('intensity-label'),
  advToggle: document.getElementById('adversarial-toggle'),
  advLabel: document.getElementById('adversarial-label'),
  simStatus: document.getElementById('sim-status'),
  
  // Alert Banner
  alertBanner: document.getElementById('alert-banner'),
  alertText: document.getElementById('alert-text'),
  alertDismiss: document.getElementById('alert-dismiss'),
  
  // Hero Metrics
  heroDetBefore: document.getElementById('hero-detection-before'),
  heroDetAfter: document.getElementById('hero-detection-after'),
  heroReduction: document.getElementById('hero-reduction'),
  heroLatency: document.getElementById('hero-latency'),
  heroStrategy: document.getElementById('hero-strategy'),
  
  // Intelligence Panel (New)
  leakageToday: document.getElementById('leakage-today'),
  leakageTotal: document.getElementById('leakage-total'),
  leakagePerOrder: document.getElementById('leakage-per-order'),
  vixValue: document.getElementById('vix-value'),
  regimeValue: document.getElementById('regime-value'),
  intelVix: document.getElementById('intel-vix'),
  intelVixRegime: document.getElementById('intel-vix-regime'),
  intelHft: document.getElementById('intel-hft'),
  intelAdverse: document.getElementById('intel-adverse'),
  intelIntensity: document.getElementById('intel-intensity'),
  intelSession: document.getElementById('intel-session'),
  marketBody: document.getElementById('market-body'),
  marketSource: document.getElementById('market-source'),
  
  // Benchmarks
  benchLatencyMean: document.getElementById('bench-latency-mean'),
  benchLatencyP50: document.getElementById('bench-latency-p50'),
  benchLatencyP99: document.getElementById('bench-latency-p99'),
  benchLatencyBar: document.getElementById('bench-latency-bar'),
  benchVwapMean: document.getElementById('bench-vwap-mean'),
  benchVwapMax: document.getElementById('bench-vwap-max'),
  benchVwapPct: document.getElementById('bench-vwap-pct'),
  benchVwapBar: document.getElementById('bench-vwap-bar'),
  benchDetReduction: document.getElementById('bench-det-reduction'),
  benchDetBefore: document.getElementById('bench-det-before'),
  benchDetAfter: document.getElementById('bench-det-after'),
  benchDetBar: document.getElementById('bench-det-bar'),
  benchCompliance: document.getElementById('bench-compliance'),
  benchLpaCount: document.getElementById('bench-lpa-count'),
  benchAllCompliant: document.getElementById('bench-all-compliant'),
  benchComplianceBar: document.getElementById('bench-compliance-bar'),
  
  // VWAP display in command center
  vwapDevValue: document.getElementById('vwap-dev-value'),
  vwapBadge: document.getElementById('vwap-badge'),
  
  // Tables & Lists
  liveFeedBody: document.getElementById('live-feed-body'),
  ordersCount: document.getElementById('orders-count'),
  patternsList: document.getElementById('patterns-list'),
  patternsCount: document.getElementById('patterns-count'),
  modsLog: document.getElementById('modifications-log'),
  auditBody: document.getElementById('audit-body'),
  
  // Compliance
  compTotal: document.getElementById('comp-total'),
  compCompliant: document.getElementById('comp-compliant'),
  compLatency: document.getElementById('comp-latency'),
  compMaxDev: document.getElementById('comp-max-dev'),
  
  // Gauges
  gaugeUnprotected: document.getElementById('gauge-text-unprotected'),
  gaugeArcUnprotected: document.getElementById('gauge-arc-unprotected'),
  gaugeProtected: document.getElementById('gauge-text-protected'),
  gaugeArcProtected: document.getElementById('gauge-arc-protected'),
  
  // Investor Features
  mlBadge: document.getElementById('ml-badge'),
  mlAccuracyVal: document.getElementById('ml-accuracy-val'),
  compBeforeVal: document.getElementById('comp-before-val'),
  compAfterVal: document.getElementById('comp-after-val'),
  
  // Audit Tool
  auditFileInput: document.getElementById('audit-file-input'),
  auditResults: document.getElementById('audit-results'),
  auditLoading: document.getElementById('audit-loading'),
  auditUploadBox: document.getElementById('audit-upload-box'),
  auditScore: document.getElementById('audit-score'),
  auditRisk: document.getElementById('audit-risk'),
  auditLeakage: document.getElementById('audit-leakage'),
  auditStrategy: document.getElementById('audit-strategy'),
  auditPatterns: document.getElementById('audit-patterns'),
  auditMessage: document.getElementById('audit-message'),
};

// Initialize Charts
function initCharts() {
  // Command Center: Detection Timeline
  charts.detectionTimeline = new Chart(document.getElementById('chart-detection-timeline').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Unprotected Score', data: [], borderColor: CHART_COLORS.red, backgroundColor: CHART_COLORS.redBg, borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
        { label: 'Protected Score', data: [], borderColor: CHART_COLORS.cyan, backgroundColor: CHART_COLORS.cyanBg, borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }
      ]
    },
    options: { ...commonOptions, scales: { y: { min: 0, max: 100 } } }
  });

  // Command Center: VWAP
  charts.vwap = new Chart(document.getElementById('chart-vwap').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{ label: 'VWAP Deviation (bps)', data: [], borderColor: CHART_COLORS.purple, backgroundColor: 'rgba(179, 136, 255, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }]
    },
    options: { ...commonOptions, scales: { y: { min: 0, max: 10 } } }
  });

  // Attacker: Feature Analysis (Radar)
  charts.features = new Chart(document.getElementById('chart-features').getContext('2d'), {
    type: 'radar',
    data: {
      labels: ['Size Autocorr', 'Time Autocorr', 'Venue Predictability', 'Size CV', 'Timing CV', 'Directional Runs', 'Bursts', 'Rounding'],
      datasets: [
        { label: 'Unprotected', data: [0,0,0,0,0,0,0,0], borderColor: CHART_COLORS.red, backgroundColor: CHART_COLORS.redBg, pointBackgroundColor: CHART_COLORS.red },
        { label: 'Protected', data: [0,0,0,0,0,0,0,0], borderColor: CHART_COLORS.cyan, backgroundColor: CHART_COLORS.cyanBg, pointBackgroundColor: CHART_COLORS.cyan }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 }, scales: { r: { min: 0, max: 1, ticks: { display: false }, grid: { color: CHART_COLORS.grid }, angleLines: { color: CHART_COLORS.grid } } } }
  });

  // Flow: Comparison
  const flowOptions = { ...commonOptions, scales: { x: { display: false }, y: { title: { display: true, text: 'Size' } } } };
  charts.flowUnprotected = new Chart(document.getElementById('chart-flow-unprotected').getContext('2d'), {
    type: 'bar', data: { labels: [], datasets: [{ label: 'Size', data: [], backgroundColor: CHART_COLORS.red }] }, options: flowOptions
  });
  charts.flowProtected = new Chart(document.getElementById('chart-flow-protected').getContext('2d'), {
    type: 'bar', data: { labels: [], datasets: [{ label: 'Size', data: [], backgroundColor: CHART_COLORS.cyan }] }, options: flowOptions
  });

  // Intelligence: Strategy Comparison
  charts.strategy = new Chart(document.getElementById('chart-strategy').getContext('2d'), {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Avg Detection Score', data: [], backgroundColor: CHART_COLORS.red }] },
    options: { ...commonOptions, scales: { y: { min: 0, max: 100 } } }
  });

  // Intelligence: Obfuscation Effectiveness Scatter
  charts.effectiveness = new Chart(document.getElementById('chart-effectiveness').getContext('2d'), {
    type: 'scatter',
    data: { datasets: [{ label: 'Effectiveness', data: [], backgroundColor: CHART_COLORS.cyan }] },
    options: { ...commonOptions, scales: { x: { title: { display: true, text: 'Modifications Count' }, min: 0, max: 6, ticks: { stepSize: 1 } }, y: { title: { display: true, text: 'Score Reduction (%)' }, min: 0, max: 100 } } }
  });

  // Intelligence: Leakage Trend
  charts.leakageTrend = new Chart(document.getElementById('chart-leakage-trend').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Cumulative Leakage ($)', data: [], borderColor: CHART_COLORS.amber, backgroundColor: 'rgba(255, 179, 0, 0.1)', fill: true, tension: 0.4 }] },
    options: { ...commonOptions }
  });

  // Other histograms
  const histOptions = { ...commonOptions, scales: { x: { grid: { display: false } }, y: { display: false } } };
  charts.sizeDist = new Chart(document.getElementById('chart-size-dist').getContext('2d'), {
    type: 'bar', data: { labels: ['Q1', 'Q2', 'Q3', 'Q4'], datasets: [{ data: [0,0,0,0], backgroundColor: CHART_COLORS.purple }] }, options: histOptions
  });
  charts.venueDist = new Chart(document.getElementById('chart-venue-dist').getContext('2d'), {
    type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: CHART_COLORS.cyan }] }, options: histOptions
  });
  charts.timingDist = new Chart(document.getElementById('chart-timing-dist').getContext('2d'), {
    type: 'bar', data: { labels: ['0-200', '200-500', '500-1000', '>1000'], datasets: [{ data: [0,0,0,0], backgroundColor: CHART_COLORS.green }] }, options: histOptions
  });
}

// Draw Heatmap (Canvas 2D)
function drawHeatmap(data) {
  const canvas = document.getElementById('chart-heatmap');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  // Set explicit dimensions for canvas rendering
  const width = canvas.parentElement.clientWidth - 48; // account for padding
  const height = 200;
  canvas.width = width;
  canvas.height = height;
  
  ctx.clearRect(0, 0, width, height);
  
  if (Object.keys(data).length === 0) {
    ctx.fillStyle = CHART_COLORS.grid;
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting data...', width/2, height/2);
    return;
  }
  
  const cellW = width / 13; // 13 time buckets
  const cellH = height / 5; // 5 size buckets
  
  for (let t = 0; t < 13; t++) {
    for (let s = 0; s < 5; s++) {
      const key = `${t}_${s}`;
      const score = data[key] || 0;
      
      // Color interpolation: blue (low) to red (high)
      let r, g, b, a;
      if (score === 0) {
        r = 30; g = 35; b = 45; a = 0.5; // empty
      } else if (score < 50) {
        r = 0; g = 229; b = 255; a = Math.max(0.2, score/50); // cyan
      } else {
        r = 255; g = 42; b = 85; a = Math.max(0.4, score/100); // red
      }
      
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
      ctx.fillRect(t * cellW + 1, s * cellH + 1, cellW - 2, cellH - 2);
      
      if (score > 0) {
        ctx.fillStyle = '#fff';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(score), t * cellW + cellW/2, s * cellH + cellH/2);
      }
    }
  }
}

// UI Navigation
DOM.navItems.forEach(item => {
  item.addEventListener('click', () => {
    DOM.navItems.forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    DOM.sections.forEach(s => s.classList.remove('active'));
    document.getElementById(`section-${item.dataset.section}`).classList.add('active');
  });
});

// WebSocket Connection
let reconnectAttempts = 0;
function connectWS() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    isConnected = true;
    reconnectAttempts = 0;
    DOM.simStatus.querySelector('.status-dot').style.animation = 'pulse 2s infinite';
    DOM.simStatus.querySelector('span').innerText = 'CONNECTED';
  };

  ws.onclose = () => {
    isConnected = false;
    DOM.simStatus.querySelector('.status-dot').style.animation = 'none';
    DOM.simStatus.querySelector('.status-dot').style.boxShadow = 'none';
    DOM.simStatus.querySelector('.status-dot').style.background = CHART_COLORS.red;
    DOM.simStatus.querySelector('span').innerText = 'DISCONNECTED';
    
    // Exponential backoff
    reconnectAttempts++;
    const delay = Math.min(10000, 1000 * Math.pow(1.5, reconnectAttempts));
    setTimeout(connectWS, delay);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'tick') {
      updateDashboard(data);
    }
  };
}

// Update UI on Tick
function updateDashboard(data) {
  const m = data.metrics;
  const ua = data.undefended_analysis;
  const da = data.defended_analysis;
  const la = data.last_action;
  
  // Hero Metrics
  const scoreBefore = ua.detection_score || 0;
  const scoreAfter = da.detection_score || 0;
  const reduction = Math.max(0, scoreBefore - scoreAfter);
  
  DOM.heroDetBefore.innerText = scoreBefore.toFixed(1) + '%';
  DOM.heroDetAfter.innerText = scoreAfter.toFixed(1) + '%';
  DOM.heroReduction.innerText = reduction.toFixed(1) + '%';
  DOM.heroLatency.innerText = (da.latency_ms || 0).toFixed(2) + 'ms';
  DOM.heroStrategy.innerText = `Detected: ${ua.strategy_classification || 'UNKNOWN'}`;

  // Investor Comparison
  DOM.compBeforeVal.innerText = scoreBefore.toFixed(0) + '%';
  DOM.compAfterVal.innerText = scoreAfter.toFixed(0) + '%';

  // ML Metrics — nested under strategy_classifier / defense_discriminator
  // (see engine/ml_models.py). held_out_auc/held_out_accuracy are null until
  // scripts/train_and_evaluate.py has been run at least once; that's
  // intentional, not a bug — there is no live-trained number to show instead.
  const disc = data.ml_metrics && data.ml_metrics.defense_discriminator;
  if (disc && disc.trained && disc.held_out_auc !== null) {
    DOM.mlBadge.style.display = 'flex';
    DOM.mlAccuracyVal.innerText = (disc.held_out_auc * 100).toFixed(1) + '% AUC';
  }

  // The adversarial defender (engine/adversarial_defender.py) can only run
  // once a DefenseDiscriminator is trained — enable the toggle only then,
  // and don't silently re-enable a checked box the backend can't honor.
  if (DOM.advToggle) {
    const available = !!(disc && disc.trained);
    DOM.advToggle.disabled = !available;
    DOM.advLabel.innerText = available
      ? 'discriminator-guided search, ~5-15ms/order — see docs/experiments.md'
      : 'unavailable — model not trained';
  }
  
  // Alert Banner
  if (scoreBefore > 65) {
    DOM.alertBanner.style.display = 'flex';
    DOM.alertText.innerText = `HIGH RISK: Attacker detection score (${scoreBefore.toFixed(1)}%) exceeds safe threshold. Strategy isolated: ${ua.strategy_classification}`;
  }
  DOM.alertDismiss.onclick = () => DOM.alertBanner.style.display = 'none';

  // Intelligence
  if (data.leakage) {
    DOM.leakageToday.innerText = '$' + data.leakage.today.toLocaleString('en-US', {minimumFractionDigits: 2});
    DOM.leakageTotal.innerText = '$' + data.leakage.total.toLocaleString('en-US', {minimumFractionDigits: 2});
    DOM.leakagePerOrder.innerText = '$' + data.leakage.per_order.toLocaleString('en-US', {minimumFractionDigits: 2});
  }
  
  if (data.market_risk) {
    DOM.vixValue.innerText = data.market_risk.vix.toFixed(2);
    DOM.regimeValue.innerText = data.market_risk.market_regime;
    DOM.intelVix.innerText = data.market_risk.vix.toFixed(2);
    DOM.intelVixRegime.innerText = data.market_risk.vix_regime;
    DOM.intelHft.innerText = data.market_risk.hft_activity_score.toFixed(1);
    DOM.intelAdverse.innerText = (data.market_risk.adverse_selection_prob * 100).toFixed(1) + '%';
    DOM.intelIntensity.innerText = 'L' + data.market_risk.recommended_intensity;
    DOM.intelSession.innerText = data.market_risk.session_phase;
    
    // Market Table
    DOM.marketBody.innerHTML = '';
    const symbols = data.market_risk.tracked_symbols;
    let anyLive = false;
    for (const [sym, info] of Object.entries(symbols)) {
      if (info.source === 'yfinance_live') anyLive = true;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${sym}</strong></td>
        <td>$${info.price.toFixed(2)}</td>
        <td>${(info.volatility*100).toFixed(1)}%</td>
        <td class="${info.momentum_1h >= 0 ? 'text-cyan' : 'text-red'}">${(info.momentum_1h*100).toFixed(2)}%</td>
        <td class="text-muted">${info.source === 'yfinance_live' ? 'Live API' : 'Synthetic'}</td>
      `;
      DOM.marketBody.appendChild(tr);
    }
    DOM.marketSource.innerText = anyLive ? "Live Data Connected" : "Synthetic Sandbox";
    DOM.marketSource.className = anyLive ? "badge badge-blue" : "badge badge-amber";
  }

  // VWAP Display
  const vwapDev = Math.abs(m.vwap_deviation_bps || 0);
  DOM.vwapDevValue.innerText = vwapDev.toFixed(1) + ' bps';
  if (vwapDev <= 8.0) {
    DOM.vwapBadge.className = 'badge badge-green';
    DOM.vwapBadge.innerText = 'Within Tolerance';
  } else {
    DOM.vwapBadge.className = 'badge badge-red';
    DOM.vwapBadge.innerText = 'Exceeds Limit';
  }

  // Update Charts Arrays
  const timeLabel = data.tick.toString();
  
  // 1. Timeline
  if (charts.detectionTimeline.data.labels.length > maxDataPoints) {
    charts.detectionTimeline.data.labels.shift();
    charts.detectionTimeline.data.datasets[0].data.shift();
    charts.detectionTimeline.data.datasets[1].data.shift();
  }
  charts.detectionTimeline.data.labels.push(timeLabel);
  charts.detectionTimeline.data.datasets[0].data.push(scoreBefore);
  charts.detectionTimeline.data.datasets[1].data.push(scoreAfter);
  charts.detectionTimeline.update();

  // 2. VWAP
  if (charts.vwap.data.labels.length > maxDataPoints) {
    charts.vwap.data.labels.shift();
    charts.vwap.data.datasets[0].data.shift();
  }
  charts.vwap.data.labels.push(timeLabel);
  charts.vwap.data.datasets[0].data.push(vwapDev);
  charts.vwap.update();

  // 3. Features Radar
  if (ua.features && da.features) {
    charts.features.data.datasets[0].data = [
      ua.features.size_autocorr_1 || 0, ua.features.timing_autocorr_1 || 0,
      1 - (ua.features.venue_entropy || 0), ua.features.size_cv || 0,
      ua.features.timing_cv || 0, ua.features.direction_runs_z ? Math.min(1, ua.features.direction_runs_z/3) : 0,
      ua.features.burst_flag || 0, ua.features.size_rounding_score || 0
    ];
    charts.features.data.datasets[1].data = [
      da.features.size_autocorr_1 || 0, da.features.timing_autocorr_1 || 0,
      1 - (da.features.venue_entropy || 0), da.features.size_cv || 0,
      da.features.timing_cv || 0, da.features.direction_runs_z ? Math.min(1, da.features.direction_runs_z/3) : 0,
      da.features.burst_flag || 0, da.features.size_rounding_score || 0
    ];
    charts.features.update();
  }

  // 4. Intelligence Charts
  if (data.strategy_scores) {
    charts.strategy.data.labels = Object.keys(data.strategy_scores);
    charts.strategy.data.datasets[0].data = Object.values(data.strategy_scores).map(v => v.avg_score);
    charts.strategy.update();
  }
  
  if (la) {
    const modCount = la.modifications.length;
    if (charts.effectiveness.data.datasets[0].data.length > 30) charts.effectiveness.data.datasets[0].data.shift();
    charts.effectiveness.data.datasets[0].data.push({ x: modCount, y: reduction });
    charts.effectiveness.update();
  }

  // 5. Defender/Flow Charts
  if (la) {
    if (charts.flowUnprotected.data.labels.length > 30) {
      charts.flowUnprotected.data.labels.shift();
      charts.flowUnprotected.data.datasets[0].data.shift();
      charts.flowProtected.data.labels.shift();
      charts.flowProtected.data.datasets[0].data.shift();
    }
    charts.flowUnprotected.data.labels.push(timeLabel);
    charts.flowUnprotected.data.datasets[0].data.push(la.original.size);
    charts.flowUnprotected.update();
    
    charts.flowProtected.data.labels.push(timeLabel);
    charts.flowProtected.data.datasets[0].data.push(la.modified.size);
    charts.flowProtected.update();

    // Maintain local history for distributions
    if (!window._recent) window._recent = { sizes: [], venues: {}, timings: [] };
    if (window._recent.sizes.length > 50) window._recent.sizes.shift();
    if (window._recent.timings.length > 50) window._recent.timings.shift();
    
    window._recent.sizes.push(la.modified.size);
    window._recent.timings.push(la.modified.timing_ms || 500);
    window._recent.venues[la.modified.venue] = (window._recent.venues[la.modified.venue] || 0) + 1;
    
    // Size Dist Update
    let sBins = [0,0,0,0];
    window._recent.sizes.forEach(s => {
      if (s < 1000) sBins[0]++;
      else if (s < 5000) sBins[1]++;
      else if (s < 10000) sBins[2]++;
      else sBins[3]++;
    });
    charts.sizeDist.data.datasets[0].data = sBins;
    charts.sizeDist.update();
    
    // Timing Dist Update
    let tBins = [0,0,0,0];
    window._recent.timings.forEach(t => {
      if (t < 200) tBins[0]++;
      else if (t < 500) tBins[1]++;
      else if (t < 1000) tBins[2]++;
      else tBins[3]++;
    });
    charts.timingDist.data.datasets[0].data = tBins;
    charts.timingDist.update();
    
    // Venue Dist Update
    charts.venueDist.data.labels = Object.keys(window._recent.venues);
    charts.venueDist.data.datasets[0].data = Object.values(window._recent.venues);
    charts.venueDist.update();
  }

  // Update Heatmap
  if (data.heatmap_summary && document.getElementById('section-intelligence').classList.contains('active')) {
    drawHeatmap(data.heatmap_summary);
  }

  // Live Feed Table Update
  if (la) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${new Date(la.timestamp * 1000).toLocaleTimeString()}</td>
      <td><strong>${la.original.symbol}</strong></td>
      <td class="${la.original.direction === 'BUY' ? 'text-cyan' : 'text-red'}">${la.original.direction}</td>
      <td>${la.original.size}</td>
      <td class="${la.original.size !== la.modified.size ? 'text-cyan' : ''}">${la.modified.size}</td>
      <td>${la.original.venue}</td>
      <td class="${la.original.venue !== la.modified.venue ? 'text-cyan' : ''}">${la.modified.venue}</td>
      <td><span class="badge badge-amber">${la.modifications.length} mods</span></td>
      <td class="text-cyan">${la.lpa ? la.lpa.token : '—'}</td>
    `;
    DOM.liveFeedBody.prepend(tr);
    if (DOM.liveFeedBody.children.length > 20) DOM.liveFeedBody.lastChild.remove();
    DOM.ordersCount.innerText = `${data.tick} orders`;
  }

  // Gauges
  updateGauge(DOM.gaugeArcUnprotected, DOM.gaugeUnprotected, scoreBefore);
  updateGauge(DOM.gaugeArcProtected, DOM.gaugeProtected, scoreAfter);

  // Patterns
  if (ua.patterns) {
    DOM.patternsCount.innerText = `${ua.patterns.length} patterns`;
    DOM.patternsList.innerHTML = '';
    if (ua.patterns.length === 0) {
      DOM.patternsList.innerHTML = '<div class="empty-state">No detectable patterns</div>';
    } else {
      ua.patterns.forEach(p => {
        let cls = p.severity === 'HIGH' ? 'high' : (p.severity === 'MEDIUM' ? 'medium' : 'low');
        DOM.patternsList.innerHTML += `
          <div class="pattern-item ${cls}">
            <div class="pattern-header">
              <span class="pattern-name">${p.name}</span>
              <span class="pattern-score">${p.score}%</span>
            </div>
            <div class="pattern-detail">${p.detail}</div>
          </div>
        `;
      });
    }
  }

  // Modifications Log
  if (la && la.modifications) {
    DOM.modsLog.innerHTML = '';
    if (la.modifications.length === 0) {
      DOM.modsLog.innerHTML = '<div class="empty-state">No modifications needed</div>';
    } else {
      la.modifications.forEach(m => {
        DOM.modsLog.innerHTML += `
          <div class="mod-item low">
            <div class="mod-header">
              <span class="mod-type">${m.type}</span>
              <span style="font-family: monospace;">${m.from} → ${m.to}</span>
            </div>
            <div class="mod-detail">${m.rationale}</div>
          </div>
        `;
      });
    }
  }

  // Update Benchmarks Panel
  updateBenchmarks(m, data.audit_log);
  
  // Update Compliance Audit Table
  updateAuditTable(data.audit_log);
}

function updateGauge(arcPath, textEl, score) {
  // SVG Arc length is ~251.3
  const dashoffset = 251.3 - (score / 100) * 251.3;
  arcPath.style.strokeDashoffset = dashoffset;
  textEl.innerText = score.toFixed(1) + '%';
}

function updateBenchmarks(m, audit) {
  // Latency
  const lat = m.latency_stats || {};
  DOM.benchLatencyMean.innerText = (lat.mean || 0).toFixed(2) + 'ms';
  DOM.benchLatencyP50.innerText = (lat.p50 || 0).toFixed(2) + 'ms';
  DOM.benchLatencyP99.innerText = (lat.p99 || 0).toFixed(2) + 'ms';
  const latPct = Math.min(100, ((lat.mean || 0) / 1.5) * 100);
  DOM.benchLatencyBar.style.width = latPct + '%';
  DOM.benchLatencyBar.className = latPct <= 100 ? 'benchmark-fill green' : 'benchmark-fill red';

  // VWAP
  DOM.benchVwapMean.innerText = (m.vwap_deviation_bps || 0).toFixed(2) + ' bps';
  DOM.benchVwapMax.innerText = (m.max_vwap_dev_bps || 0).toFixed(2) + ' bps';
  DOM.benchVwapPct.innerText = ((m.pct_within_vwap_limit || 0) * 100).toFixed(1) + '%';
  const vwapPct = Math.min(100, (Math.abs(m.vwap_deviation_bps || 0) / 8.0) * 100);
  DOM.benchVwapBar.style.width = vwapPct + '%';
  DOM.benchVwapBar.className = vwapPct <= 100 ? 'benchmark-fill blue' : 'benchmark-fill amber';

  // Compliance
  if (audit && audit.length > 0) {
    const compliant = audit.filter(a => a.compliant).length;
    const rate = (compliant / audit.length) * 100;
    DOM.benchCompliance.innerText = rate.toFixed(1) + '%';
    DOM.benchLpaCount.innerText = m.total_lpa_tokens || audit.length;
    DOM.benchAllCompliant.innerText = rate === 100 ? 'Yes' : 'No';
    DOM.benchComplianceBar.style.width = rate + '%';
  }
}

let lastAuditLog = [];
function updateAuditTable(audit) {
  if (!audit) return;
  lastAuditLog = audit;
  DOM.auditBody.innerHTML = '';
  // Stats
  DOM.compTotal.innerText = audit.length;
  const compliant = audit.filter(a => a.compliant).length;
  DOM.compCompliant.innerText = compliant;
  if (audit.length > 0) {
    const sumLat = audit.reduce((acc, a) => acc + a.latency_ms, 0);
    DOM.compLatency.innerText = (sumLat / audit.length).toFixed(2) + 'ms';
    const maxDev = Math.max(...audit.map(a => parseFloat(a.vwap_deviation)));
    DOM.compMaxDev.innerText = maxDev.toFixed(2) + 'bps';
  }

  // Table
  audit.forEach((a, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${audit.length - idx}</td>
      <td>${new Date(a.timestamp).toLocaleTimeString()}</td>
      <td class="text-cyan">${a.token}</td>
      <td>${a.vwap_deviation}</td>
      <td>${a.modifications.join(', ') || 'NONE'}</td>
      <td>${a.latency_ms}ms</td>
      <td class="text-cyan">${a.lpa_score ? a.lpa_score.toFixed(2) : '1.00'}</td>
      <td>${a.compliant ? '<span class="badge badge-green">PASS</span>' : '<span class="badge badge-red">FAIL</span>'}</td>
    `;
    DOM.auditBody.appendChild(tr);
  });
}

// Setup Controls
function sendControl(action, payload = {}) {
  if (isConnected) {
    ws.send(JSON.stringify({ action, ...payload }));
  }
}

DOM.btnStart.onclick = () => { sendControl('start'); DOM.btnStart.disabled = true; DOM.btnStop.disabled = false; };
DOM.btnStop.onclick = () => { sendControl('stop'); DOM.btnStart.disabled = false; DOM.btnStop.disabled = true; };
DOM.btnReset.onclick = () => {
  sendControl('reset');
  DOM.liveFeedBody.innerHTML = '';
  DOM.auditBody.innerHTML = '';
  Object.values(charts).forEach(c => {
    if (c.data.labels) c.data.labels = [];
    c.data.datasets.forEach(d => d.data = []);
    c.update();
  });
};

DOM.speedSlider.oninput = (e) => {
  DOM.speedLabel.innerText = e.target.value + '×';
  sendControl('configure', { speed: parseFloat(e.target.value) });
};

DOM.aggSlider.oninput = (e) => {
  DOM.aggLabel.innerText = e.target.value + '%';
  sendControl('configure', { aggressiveness: parseInt(e.target.value) / 100 });
};

DOM.intSlider.oninput = (e) => {
  DOM.intLabel.innerText = 'L' + e.target.value;
  sendControl('configure', { intensity: parseInt(e.target.value) });
};

if (DOM.advToggle) {
  DOM.advToggle.onchange = (e) => {
    sendControl('configure', { use_adversarial: e.target.checked });
  };
}

// Shadow Diagnostics Logic
let scanInProgress = false;
let recommendedIntensity = 0;

document.getElementById('btn-run-scan').onclick = async () => {
  if (scanInProgress) return;
  scanInProgress = true;
  const term = document.getElementById('shadow-terminal');
  const content = document.getElementById('shadow-report-content');
  const badge = document.getElementById('shadow-status-badge');
  
  badge.className = 'badge badge-amber';
  badge.innerText = 'SCANNING...';
  content.style.opacity = '0.3';
  content.style.pointerEvents = 'none';
  
  term.innerHTML = '';
  
  const typeLine = async (text, delay=30, color='#00E676') => {
    const div = document.createElement('div');
    div.style.color = color;
    term.appendChild(div);
    for (let i=0; i<text.length; i++) {
      div.innerHTML += text[i];
      term.scrollTop = term.scrollHeight;
      await new Promise(r => setTimeout(r, delay));
    }
  };

  await typeLine('> INITIALIZING DEEP EXECUTION SCAN...');
  await typeLine('> CONNECTING TO SIMULATION ENGINE...', 20);
  
  try {
    const res = await fetch('/api/orders/shadow', { method: 'POST' });
    const data = await res.json();
    
    if (data.status === 'no_data') {
      await typeLine('> ERROR: No active simulation data found.', 30, '#FF2A55');
      await typeLine('> Start the simulation first to collect order flow.');
      scanInProgress = false;
      badge.className = 'badge badge-red';
      badge.innerText = 'ERROR';
      return;
    }
    
    await typeLine('> EXTRACTING RECENT ORDER BATCH: 200 orders');
    await new Promise(r => setTimeout(r, 500));
    await typeLine('> RUNNING HEURISTIC PATTERN ANALYSIS...', 15);
    await new Promise(r => setTimeout(r, 800));

    const assess = data.attacker_assessment;
    const leakage = data.estimated_leakage;
    recommendedIntensity = data.recommended_intensity || 3;

    await typeLine(`> PATTERNS DETECTED: ${assess.top_features.join(', ') || 'None'}`, 20, '#FFB300');
    await typeLine(`> STRATEGY CLASSIFICATION: ${assess.strategy_classification}`, 20, '#00E5FF');
    await typeLine(`> SCAN COMPLETE. GENERATING REPORT...`);

    // Populate Report
    document.getElementById('shadow-strategy').innerText = assess.strategy_classification;
    document.getElementById('shadow-confidence').innerText = assess.confidence.toFixed(1) + '%';
    document.getElementById('shadow-bot').innerText = '$' + leakage.per_order_usd.toLocaleString();
    document.getElementById('shadow-impact').innerText = leakage.basis;
    document.getElementById('shadow-cure').innerText = data.recommended_action;
    document.getElementById('shadow-savings-label').innerText = 'Applying this changes the Defender intensity — it does not guarantee any specific outcome.';
    
    // Reveal
    badge.className = 'badge badge-red';
    badge.innerText = 'VULNERABLE';
    content.style.opacity = '1';
    content.style.pointerEvents = 'auto';
    
  } catch (err) {
    await typeLine('> CONNECTION FAILED.', 30, '#FF2A55');
  }
  scanInProgress = false;
};

document.getElementById('btn-deploy-cure').onclick = () => {
  fetch('/api/simulation/configure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ intensity: recommendedIntensity })
  });
  
  DOM.intSlider.value = recommendedIntensity;
  DOM.intLabel.innerText = 'L' + recommendedIntensity;
  
  const btn = document.getElementById('btn-deploy-cure');
  btn.innerText = 'CURE DEPLOYED ✓';
  btn.style.background = '#00E676';
  btn.style.color = '#000';
  document.getElementById('shadow-status-badge').className = 'badge badge-green';
  document.getElementById('shadow-status-badge').innerText = 'SECURED';
  
  setTimeout(() => {
    btn.innerText = 'DEPLOY CURE';
    btn.style.background = '';
    btn.style.color = '';
  }, 3000);
};

// Audit Tool Logic
DOM.auditFileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    DOM.auditUploadBox.style.display = 'none';
    DOM.auditLoading.style.display = 'block';
    DOM.auditResults.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/audit/upload', {
            method: 'POST',
            body: formData
        });
        const result = await res.json();

        DOM.auditLoading.style.display = 'none';

        if (result.error) {
            alert(result.error);
            DOM.auditUploadBox.style.display = 'block';
            return;
        }

        renderAuditResults(result);

    } catch (err) {
        alert("Upload failed. Check console for details.");
        console.error(err);
        DOM.auditLoading.style.display = 'none';
        DOM.auditUploadBox.style.display = 'block';
    }
};

window.runSampleAnalysis = async (filename) => {
    DOM.auditUploadBox.style.display = 'none';
    DOM.auditLoading.style.display = 'block';
    DOM.auditResults.style.display = 'none';

    try {
        const res = await fetch(`/api/audit/analyze-sample/${filename}`, {
            method: 'POST'
        });
        const result = await res.json();
        
        DOM.auditLoading.style.display = 'none';
        
        if (result.error) {
            alert(result.error);
            DOM.auditUploadBox.style.display = 'block';
            return;
        }
        
        renderAuditResults(result);
    } catch (err) {
        alert("Sample analysis failed.");
        console.error(err);
        DOM.auditLoading.style.display = 'none';
        DOM.auditUploadBox.style.display = 'block';
    }
};

function renderAuditResults(result) {
    DOM.auditResults.style.display = 'block';
    DOM.auditScore.innerText = result.detection_score.toFixed(1) + '%';
    DOM.auditRisk.innerText = result.risk_level;
    DOM.auditLeakage.innerText = '$' + result.estimated_annual_leakage_usd.toLocaleString();
    DOM.auditStrategy.innerText = result.strategy_classification;
    DOM.auditMessage.innerText = result.message;

    // Validated (GroupKFold) model output — separate from the heuristic
    // fields above. If no model has been trained yet, say so plainly
    // instead of showing a blank or a misleading number.
    const vm = result.validated_models;
    const vmEl = document.getElementById('audit-validated-model');
    if (vmEl) {
        if (!vm || !vm.strategy_classifier || !vm.strategy_classifier.trained) {
            vmEl.innerText = 'Not trained yet (run scripts/train_and_evaluate.py) — showing heuristic score only above.';
        } else {
            const sc = vm.strategy_classifier;
            const dd = vm.defense_discriminator;
            const parts = [`Predicted strategy: ${sc.predicted_strategy} (${(sc.confidence * 100).toFixed(0)}% confidence)`];
            if (dd && dd.trained) {
                parts.push(`P(defended) = ${(dd.p_defended * 100).toFixed(0)}%`);
            }
            vmEl.innerText = parts.join(' · ');
        }
    }

    // Patterns
    DOM.auditPatterns.innerHTML = '';
    result.patterns.forEach(p => {
        const li = document.createElement('li');
        li.innerText = p.name;
        DOM.auditPatterns.appendChild(li);
    });

    // Set risk card color
    const riskCard = document.getElementById('audit-risk-card');
    riskCard.className = 'metric-card glass-card';
    if (result.risk_level === 'HIGH') riskCard.classList.add('metric-red');
    else if (result.risk_level === 'MEDIUM') riskCard.classList.add('metric-amber');
    else riskCard.classList.add('metric-green');
}

// CSV Export Logic
document.getElementById('btn-export-audit').onclick = () => {
  if (!lastAuditLog || lastAuditLog.length === 0) {
    alert("No audit logs available to export.");
    return;
  }
  
  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Timestamp,Token,Symbol,Direction,Original_Size,Modified_Size,VWAP_Deviation_bps,Latency_ms,Compliant\n";
  
  lastAuditLog.forEach(log => {
    const d = new Date(log.timestamp).toISOString();
    const token = log.token;
    // Extract info if available, otherwise generic
    const vwap = log.vwap_deviation;
    const lat = log.latency_ms;
    const comp = log.compliant ? "TRUE" : "FALSE";
    
    csvContent += `${d},${token},,,,${vwap},${lat},${comp}\n`;
  });
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `redblue_lpa_audit_${new Date().getTime()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Heatmap resize observer
window.addEventListener('resize', () => {
  const canvas = document.getElementById('chart-heatmap');
  if (canvas && document.getElementById('section-intelligence').classList.contains('active')) {
    // Will redraw on next tick
  }
});

// Boot
window.onload = () => {
  initCharts();
  connectWS();
};
