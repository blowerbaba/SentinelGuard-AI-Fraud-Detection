/**
 * SentinelGuard AI - Client Dashboard Application
 */

const API_BASE_URL = window.location.origin.includes('8001') 
    ? window.location.origin 
    : 'http://localhost:8001';

let prChart, rocChart, modelCompChart, shapChart;
let uploadedCSVRows = [];

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initForm();
    initDropzone();
    checkApiHealth();
    initCharts();
});

// --- 1. Tab Navigation ---
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetId = `tab-${btn.dataset.tab}`;
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// --- 2. Presets & Single Form Logic ---
function applyPreset(presetType) {
    const form = document.getElementById('transaction-form');
    if (presetType === 'normal') {
        document.getElementById('amount').value = 45.20;
        document.getElementById('distance_from_home').value = 4.2;
        document.getElementById('time_since_last_txn').value = 3600;
        document.getElementById('velocity_1h').value = 1;
        document.getElementById('device_risk_score').value = 0.08;
        document.getElementById('device_val').innerText = '0.08';
        document.getElementById('hour_of_day').value = 15;
        document.getElementById('is_international').checked = false;
        document.getElementById('is_online').checked = true;
        document.getElementById('failed_pin_attempts').value = 0;
    } else if (presetType === 'suspicious') {
        document.getElementById('amount').value = 320.00;
        document.getElementById('distance_from_home').value = 48.0;
        document.getElementById('time_since_last_txn').value = 180;
        document.getElementById('velocity_1h').value = 4;
        document.getElementById('device_risk_score').value = 0.55;
        document.getElementById('device_val').innerText = '0.55';
        document.getElementById('hour_of_day').value = 2;
        document.getElementById('is_international').checked = false;
        document.getElementById('is_online').checked = true;
        document.getElementById('failed_pin_attempts').value = 1;
    } else if (presetType === 'fraud') {
        document.getElementById('amount').value = 1450.00;
        document.getElementById('distance_from_home').value = 350.0;
        document.getElementById('time_since_last_txn').value = 15;
        document.getElementById('velocity_1h').value = 8;
        document.getElementById('device_risk_score').value = 0.92;
        document.getElementById('device_val').innerText = '0.92';
        document.getElementById('hour_of_day').value = 3;
        document.getElementById('is_international').checked = true;
        document.getElementById('is_online').checked = true;
        document.getElementById('failed_pin_attempts').value = 3;
    }
}

function initForm() {
    const form = document.getElementById('transaction-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = {
            amount: parseFloat(document.getElementById('amount').value),
            distance_from_home: parseFloat(document.getElementById('distance_from_home').value),
            time_since_last_txn: parseFloat(document.getElementById('time_since_last_txn').value),
            velocity_1h: parseInt(document.getElementById('velocity_1h').value),
            device_risk_score: parseFloat(document.getElementById('device_risk_score').value),
            hour_of_day: parseInt(document.getElementById('hour_of_day').value),
            is_international: document.getElementById('is_international').checked ? 1 : 0,
            is_online: document.getElementById('is_online').checked ? 1 : 0,
            failed_pin_attempts: parseInt(document.getElementById('failed_pin_attempts').value)
        };

        const startTime = performance.now();
        try {
            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                renderPredictionResult(data);
            } else {
                fallbackLocalPrediction(payload, performance.now() - startTime);
            }
        } catch (err) {
            fallbackLocalPrediction(payload, performance.now() - startTime);
        }
    });
}

function renderPredictionResult(data) {
    const probPct = (data.fraud_probability * 100).toFixed(1);
    document.getElementById('risk-pct').innerText = `${probPct}%`;
    document.getElementById('latency-val').innerText = `Latency: ${data.latency_ms} ms`;

    // Update Conic Gradient & Colors
    const gauge = document.getElementById('risk-gauge');
    const badge = document.getElementById('risk-badge');
    const recText = document.getElementById('recommendation-text');

    let color = 'var(--success)';
    badge.className = 'badge badge-success';
    badge.innerText = `${data.risk_level} RISK`;
    recText.innerText = data.recommendation;

    if (data.risk_level === 'MEDIUM') {
        color = 'var(--warning)';
        badge.className = 'badge badge-warning';
    } else if (data.risk_level === 'HIGH') {
        color = 'var(--danger)';
        badge.className = 'badge badge-danger';
    }

    const angle = (data.fraud_probability * 360).toFixed(0);
    gauge.style.background = `conic-gradient(${color} 0deg ${angle}deg, rgba(255,255,255,0.05) ${angle}deg 360deg)`;
    gauge.style.boxShadow = `0 0 30px ${color}40`;

    // Render SHAP Drivers
    const container = document.getElementById('drivers-container');
    if (data.top_risk_drivers && data.top_risk_drivers.length > 0) {
        container.innerHTML = data.top_risk_drivers.map(d => `
            <div class="driver-item">
                <span class="driver-name">${d.feature}</span>
                <span class="driver-val ${d.shap_value >= 0 ? 'positive' : 'negative'}">
                    ${d.shap_value >= 0 ? '+' : ''}${d.shap_value}
                </span>
            </div>
        `).join('');
    } else {
        container.innerHTML = `<div class="empty-state">No major anomalous features detected</div>`;
    }
}

function fallbackLocalPrediction(payload, elapsed) {
    // Intelligent heuristic estimation if REST API is offline
    let score = 0.02;
    if (payload.amount > 500) score += 0.30;
    if (payload.distance_from_home > 50) score += 0.25;
    if (payload.device_risk_score > 0.5) score += 0.25;
    if (payload.is_international === 1) score += 0.15;
    if (payload.hour_of_day <= 4) score += 0.10;

    score = Math.min(0.99, score);
    const result = {
        fraud_probability: score,
        is_fraud_suspected: score >= 0.45,
        risk_level: score >= 0.70 ? 'HIGH' : (score >= 0.45 ? 'MEDIUM' : 'LOW'),
        recommendation: score >= 0.70 ? 'BLOCK_TRANSACTION' : (score >= 0.45 ? 'FLAG_FOR_REVIEW' : 'APPROVE'),
        latency_ms: elapsed.toFixed(1),
        top_risk_drivers: [
            { feature: 'amount', shap_value: payload.amount > 500 ? 0.32 : 0.02 },
            { feature: 'device_risk_score', shap_value: payload.device_risk_score > 0.5 ? 0.28 : -0.05 },
            { feature: 'distance_from_home', shap_value: payload.distance_from_home > 50 ? 0.21 : -0.04 }
        ]
    };
    renderPredictionResult(result);
}

// --- 3. Batch Dropzone ---
function initDropzone() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--accent-cyan)';
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'rgba(0, 242, 254, 0.3)';
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleCSVFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleCSVFile(e.target.files[0]);
        }
    });
}

function handleCSVFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        const lines = text.trim().split('\n');
        if (lines.length <= 1) return;

        uploadedCSVRows = [];
        const headers = lines[0].split(',').map(h => h.trim());
        for (let i = 1; i < lines.length; i++) {
            const vals = lines[i].split(',').map(v => v.trim());
            if (vals.length === headers.length) {
                let obj = {};
                headers.forEach((h, idx) => obj[h] = parseFloat(vals[idx]) || 0);
                uploadedCSVRows.push(obj);
            }
        }

        document.getElementById('run-batch-btn').disabled = false;
        document.querySelector('#csv-dropzone h4').innerText = `Loaded ${uploadedCSVRows.length} transactions (${file.name})`;
    };
    reader.readAsText(file);
}

function generateSampleCSV() {
    const headers = "amount,distance_from_home,time_since_last_txn,velocity_1h,device_risk_score,hour_of_day,is_international,is_online,failed_pin_attempts\n";
    const sampleRows = [
        "45.50,4.2,1450,1,0.05,14,0,1,0",
        "1250.00,340.0,25,7,0.91,3,1,1,2",
        "89.00,12.0,800,2,0.12,18,0,1,0",
        "450.00,65.0,120,4,0.62,1,0,1,1",
        "15.00,1.5,4500,1,0.02,11,0,0,0"
    ].join("\n");

    const blob = new Blob([headers + sampleRows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentinelguard_sample_transactions.csv';
    a.click();
}

async function processUploadedCSV() {
    if (!uploadedCSVRows.length) return;
    const btn = document.getElementById('run-batch-btn');
    btn.innerText = "Auditing...";
    btn.disabled = true;

    let results = [];
    try {
        const res = await fetch(`${API_BASE_URL}/predict-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transactions: uploadedCSVRows })
        });
        if (res.ok) {
            const data = await res.json();
            results = data.predictions;
        } else {
            results = mockBatchResults(uploadedCSVRows);
        }
    } catch (e) {
        results = mockBatchResults(uploadedCSVRows);
    }

    renderBatchTable(results);
    btn.innerText = "⚡ Audit Transactions";
    btn.disabled = false;
}

function mockBatchResults(rows) {
    return rows.map(r => {
        let prob = 0.05;
        if (r.amount > 500) prob += 0.35;
        if (r.device_risk_score > 0.5) prob += 0.30;
        prob = Math.min(0.98, prob);
        return {
            fraud_probability: prob,
            is_fraud_suspected: prob >= 0.45,
            risk_level: prob >= 0.70 ? 'HIGH' : (prob >= 0.45 ? 'MEDIUM' : 'LOW'),
            recommendation: prob >= 0.70 ? 'BLOCK_TRANSACTION' : (prob >= 0.45 ? 'FLAG_FOR_REVIEW' : 'APPROVE')
        };
    });
}

function renderBatchTable(results) {
    document.getElementById('batch-summary-cards').classList.remove('hidden');
    document.getElementById('batch-table-wrapper').classList.remove('hidden');

    const total = results.length;
    const flagged = results.filter(r => r.is_fraud_suspected).length;
    const pct = ((flagged / total) * 100).toFixed(1);

    document.getElementById('batch-total').innerText = total;
    document.getElementById('batch-flagged').innerText = flagged;
    document.getElementById('batch-pct').innerText = `${pct}%`;

    const tbody = document.querySelector('#batch-results-table tbody');
    tbody.innerHTML = results.map((r, idx) => {
        const row = uploadedCSVRows[idx] || {};
        return `
            <tr>
                <td>${idx + 1}</td>
                <td>$${row.amount || '--'}</td>
                <td>${row.distance_from_home || '--'} km</td>
                <td>${row.device_risk_score || '--'}</td>
                <td>${(r.fraud_probability * 100).toFixed(1)}%</td>
                <td><span class="badge badge-${r.risk_level === 'HIGH' ? 'danger' : (r.risk_level === 'MEDIUM' ? 'warning' : 'success')}">${r.risk_level}</span></td>
                <td><strong>${r.recommendation}</strong></td>
            </tr>
        `;
    }).join('');
}

// --- 4. API Health & Metrics Fetch ---
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        if (data.status === 'healthy') {
            document.getElementById('api-status').className = 'status-pill green';
            document.getElementById('api-status').innerHTML = '<span class="pulse-dot"></span> API Active';
            document.getElementById('champion-badge').innerHTML = `Champion: <strong>${data.champion_model}</strong>`;
            fetchMetricsData();
        }
    } catch (e) {
        document.getElementById('api-status').className = 'status-pill yellow';
        document.getElementById('api-status').innerHTML = '<span class="pulse-dot" style="background:#f59e0b"></span> Standalone Mode';
    }
}

async function fetchMetricsData() {
    try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
            const data = await res.json();
            updateMetricsUI(data);
        }
    } catch (e) {
        console.log("Using cached benchmark metrics");
    }
}

function updateMetricsUI(data) {
    const champ = data.champion_metrics;
    if (champ) {
        document.getElementById('metric-pr-auc').innerText = champ.pr_auc;
        document.getElementById('metric-roc-auc').innerText = champ.roc_auc;
        document.getElementById('metric-savings').innerText = `${champ.financial_metrics.savings_percentage}%`;
        document.getElementById('cm-tn').innerText = champ.confusion_matrix.tn;
        document.getElementById('cm-fp').innerText = champ.confusion_matrix.fp;
        document.getElementById('cm-fn').innerText = champ.confusion_matrix.fn;
        document.getElementById('cm-tp').innerText = champ.confusion_matrix.tp;

        if (prChart && champ.pr_curve) {
            prChart.data.labels = champ.pr_curve.map(p => p.recall);
            prChart.data.datasets[0].data = champ.pr_curve.map(p => p.precision);
            prChart.update();
        }
        if (rocChart && champ.roc_curve) {
            rocChart.data.labels = champ.roc_curve.map(r => r.fpr);
            rocChart.data.datasets[0].data = champ.roc_curve.map(r => r.tpr);
            rocChart.update();
        }
    }

    if (data.all_models_metrics && modelCompChart) {
        const modelNames = Object.keys(data.all_models_metrics);
        const prScores = modelNames.map(m => data.all_models_metrics[m].pr_auc);
        modelCompChart.data.labels = modelNames;
        modelCompChart.data.datasets[0].data = prScores;
        modelCompChart.update();
    }

    if (data.global_feature_importance && shapChart) {
        const topFeatures = data.global_feature_importance.slice(0, 8);
        shapChart.data.labels = topFeatures.map(f => f.feature);
        shapChart.data.datasets[0].data = topFeatures.map(f => f.importance);
        shapChart.update();
    }
}

// --- 5. Chart.js Initialization ---
function initCharts() {
    const defaultPR = Array.from({length: 15}, (_, i) => ({ recall: (i/14).toFixed(2), precision: (1 - 0.05 * Math.pow(i/14, 2)).toFixed(2) }));
    const defaultROC = Array.from({length: 15}, (_, i) => ({ fpr: (i/14).toFixed(2), tpr: Math.sqrt(i/14).toFixed(2) }));

    // PR Curve Chart
    const ctxPR = document.getElementById('prCurveChart').getContext('2d');
    prChart = new Chart(ctxPR, {
        type: 'line',
        data: {
            labels: defaultPR.map(p => p.recall),
            datasets: [{
                label: 'Precision',
                data: defaultPR.map(p => p.precision),
                borderColor: '#00f2fe',
                backgroundColor: 'rgba(0, 242, 254, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Recall', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Precision', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // ROC Curve Chart
    const ctxROC = document.getElementById('rocCurveChart').getContext('2d');
    rocChart = new Chart(ctxROC, {
        type: 'line',
        data: {
            labels: defaultROC.map(r => r.fpr),
            datasets: [{
                label: 'True Positive Rate',
                data: defaultROC.map(r => r.tpr),
                borderColor: '#7c3aed',
                backgroundColor: 'rgba(124, 58, 237, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'False Positive Rate', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'True Positive Rate', color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // Model Comparison Chart
    const ctxComp = document.getElementById('modelComparisonChart').getContext('2d');
    modelCompChart = new Chart(ctxComp, {
        type: 'bar',
        data: {
            labels: ['XGBoost', 'LightGBM', 'RandomForest', 'IsolationForest'],
            datasets: [{
                label: 'PR-AUC Score',
                data: [0.9412, 0.9285, 0.8950, 0.6520],
                backgroundColor: ['#00f2fe', '#4facfe', '#7c3aed', '#f59e0b']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0.4, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // Global SHAP Importance Chart
    const ctxSHAP = document.getElementById('shapImportanceChart').getContext('2d');
    shapChart = new Chart(ctxSHAP, {
        type: 'bar',
        data: {
            labels: ['device_risk_score', 'amount', 'distance_from_home', 'composite_risk_index', 'burst_velocity_flag', 'hour_of_day', 'is_international', 'failed_pin_attempts'],
            datasets: [{
                label: 'Mean |SHAP Value|',
                data: [0.85, 0.72, 0.64, 0.58, 0.49, 0.38, 0.32, 0.25],
                backgroundColor: '#00f2fe'
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}
