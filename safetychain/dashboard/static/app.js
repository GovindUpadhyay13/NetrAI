/**
 * SafetyChain — Dashboard Application
 * 
 * WebSocket client + DOM rendering for real-time alert display.
 * Handles: WebSocket connection, alert rendering, pipeline status,
 * feedback submission, and all UI interactions.
 */

// ═══ State ═══
let ws = null;
let selectedAlertId = null;
let alerts = {};
let currentFilter = 'all';
const WS_RECONNECT_DELAY = 3000;

// ═══ WebSocket Connection ═══

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/alerts`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[WS] Connected');
        updateConnectionStatus(true);
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected — reconnecting...');
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, WS_RECONNECT_DELAY);
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        updateConnectionStatus(false);
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
}

function handleMessage(message) {
    switch (message.type) {
        case 'new_alert':
            handleNewAlert(message.data);
            break;
        case 'pipeline_status':
            handlePipelineStatus(message.data);
            break;
        case 'alert_updated':
            handleAlertUpdated(message.data);
            break;
        case 'pong':
            break;
        default:
            console.log('[WS] Unknown message type:', message.type);
    }
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    const label = document.getElementById('connectionLabel');

    if (connected) {
        el.classList.add('connected');
        el.classList.remove('disconnected');
        label.textContent = 'LIVE';
    } else {
        el.classList.remove('connected');
        el.classList.add('disconnected');
        label.textContent = 'OFFLINE';
    }
}

// ═══ Alert Handling ═══

function handleNewAlert(alertData) {
    // Store the alert
    alerts[alertData.alert_id] = alertData;

    // Update counter
    document.getElementById('alertsCount').textContent =
        Object.keys(alerts).length;

    // Remove empty state
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.remove();

    // Add to alert list
    renderAlertCard(alertData, true);

    // Play sound/visual notification for high severity
    if (alertData.severity === 'EMERGENCY') {
        flashHeader('emergency');
    } else if (alertData.severity === 'ALERT') {
        flashHeader('alert');
    }
}

function renderAlertCard(alert, isNew = false) {
    const list = document.getElementById('alertList');
    const card = document.createElement('div');
    card.className = `alert-card severity-${alert.severity}${isNew ? ' new-alert' : ''}`;
    card.id = `alert-${alert.alert_id}`;
    card.onclick = () => selectAlert(alert.alert_id);

    // Check filter
    if (currentFilter !== 'all' && alert.severity !== currentFilter) {
        card.style.display = 'none';
    }

    const time = new Date(alert.timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    });

    const confidencePct = Math.round(alert.confidence * 100);
    const confidenceColor = getConfidenceColor(alert.confidence);

    card.innerHTML = `
        <div class="alert-card-header">
            <span class="alert-title">${escapeHtml(alert.title)}</span>
            <span class="alert-severity-badge ${alert.severity}">${alert.severity}</span>
        </div>
        <div class="alert-card-body">
            <span class="alert-zone">${escapeHtml(alert.zone_name)}</span>
            <span class="alert-time">${time}</span>
        </div>
        <div class="alert-confidence">
            <div class="confidence-mini-bar">
                <div class="confidence-mini-fill" style="width: ${confidencePct}%; background: ${confidenceColor};"></div>
            </div>
            <span class="confidence-mini-value">${confidencePct}%</span>
            <span class="alert-strategy">${alert.reasoning_strategy}</span>
        </div>
    `;

    // Insert at top
    list.insertBefore(card, list.firstChild);
}

function selectAlert(alertId) {
    selectedAlertId = alertId;
    const alert = alerts[alertId];
    if (!alert) return;

    // Update selected state in list
    document.querySelectorAll('.alert-card').forEach(c =>
        c.classList.remove('selected')
    );
    const card = document.getElementById(`alert-${alertId}`);
    if (card) card.classList.add('selected');

    // Update evidence panel
    renderEvidence(alert);

    // Update reasoning panel
    renderReasoning(alert);

    // Update action bar
    updateActionBar(alert);
}

// ═══ Evidence Panel ═══

function renderEvidence(alert) {
    const placeholder = document.getElementById('evidencePlaceholder');
    const frame = document.getElementById('evidenceFrame');
    const details = document.getElementById('evidenceDetails');
    const meta = document.getElementById('evidenceMeta');

    // Show frame
    if (alert.frame_b64) {
        placeholder.style.display = 'none';
        frame.style.display = 'block';
        frame.src = `data:image/jpeg;base64,${alert.frame_b64}`;
    } else {
        placeholder.style.display = 'flex';
        frame.style.display = 'none';
    }

    // Show details
    details.style.display = 'grid';

    document.getElementById('detailClassification').textContent =
        alert.classification;
    document.getElementById('detailClassification').style.color =
        getClassificationColor(alert.classification);

    const confPct = Math.round(alert.confidence * 100);
    document.getElementById('detailConfidence').textContent = `${confPct}%`;
    document.getElementById('detailConfidenceBar').style.setProperty(
        '--confidence', `${confPct}%`
    );

    document.getElementById('detailStrategy').textContent =
        alert.reasoning_strategy;
    document.getElementById('detailLatency').textContent =
        `${alert.reasoning_latency_ms}ms`;

    // Meta
    const time = new Date(alert.timestamp).toLocaleString();
    meta.textContent = `${alert.alert_id} · ${time}`;
}

// ═══ Reasoning Panel ═══

function renderReasoning(alert) {
    const content = document.getElementById('reasoningContent');
    const badge = document.getElementById('strategyBadge');

    // Update strategy badge
    badge.textContent = alert.reasoning_strategy;
    badge.className = `strategy-badge ${alert.reasoning_strategy}`;

    // Clear content
    content.innerHTML = '';

    if (alert.reasoning_strategy === 'ZeroThink') {
        content.innerHTML = `
            <div class="zerothink-message">
                <div class="icon">⚡</div>
                <div class="title">ZeroThink — Instant Verdict</div>
                <div class="desc">Critical threat detected. Reasoning bypassed for immediate escalation.<br>
                Classification: CONFIRMED_ANOMALY at 99% confidence.</div>
            </div>
        `;
    } else {
        // Render reasoning steps
        for (const step of alert.reasoning_chain) {
            const stepEl = document.createElement('div');
            stepEl.className = `reasoning-step ${step.passed ? 'passed' : 'failed'}`;

            stepEl.innerHTML = `
                <div class="step-header">
                    <span class="step-title">Step ${step.step}: ${escapeHtml(step.title)}</span>
                    <span class="step-icon">${step.passed ? '✅' : '❌'}</span>
                </div>
                <div class="step-content">${escapeHtml(step.content)}</div>
            `;

            content.appendChild(stepEl);
        }
    }

    // Hypotheses
    const hypoSection = document.getElementById('hypothesesSection');
    const hypoList = document.getElementById('hypothesesList');
    if (alert.alternative_hypotheses && alert.alternative_hypotheses.length > 0) {
        hypoSection.style.display = 'block';
        hypoList.innerHTML = alert.alternative_hypotheses
            .map(h => `<li>${escapeHtml(h)}</li>`)
            .join('');
    } else {
        hypoSection.style.display = 'none';
    }

    // SOP
    const sopSection = document.getElementById('sopSection');
    const sopContent = document.getElementById('sopContent');
    if (alert.sop) {
        sopSection.style.display = 'block';
        sopContent.textContent = alert.sop;
    } else {
        sopSection.style.display = 'none';
    }

    // Contacts
    const contactsSection = document.getElementById('contactsSection');
    const contactsGrid = document.getElementById('contactsGrid');
    if (alert.contacts && Object.keys(alert.contacts).length > 0) {
        contactsSection.style.display = 'block';
        contactsGrid.innerHTML = Object.entries(alert.contacts)
            .map(([role, number]) => `
                <div class="contact-item">
                    <span class="contact-role">${escapeHtml(role.replace(/_/g, ' '))}</span>
                    <span class="contact-number">${escapeHtml(number)}</span>
                </div>
            `)
            .join('');
    } else {
        contactsSection.style.display = 'none';
    }
}

// ═══ Pipeline Status ═══

function handlePipelineStatus(status) {
    // Update FPS
    document.getElementById('fpsValue').textContent = status.fps || '0.0';

    // Update active stage
    const stages = ['PERCEIVE', 'DESCRIBE', 'CONTEXTUALIZE', 'VERIFY', 'ACT'];
    const stageIds = ['stagePerceive', 'stageDescribe', 'stageContext', 'stageVerify', 'stageAct'];

    stageIds.forEach((id, idx) => {
        const el = document.getElementById(id);
        if (stages[idx] === status.active_stage) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

// ═══ Alert Updates ═══

function handleAlertUpdated(data) {
    const alert = alerts[data.alert_id];
    if (!alert) return;

    alert.status = data.status;
    if (data.feedback) {
        alert.operator_feedback = data.feedback;
    }

    // Update card visuals if needed
    const card = document.getElementById(`alert-${data.alert_id}`);
    if (card) {
        card.style.opacity = data.status === 'dismissed' ? '0.5' : '1';
    }
}

// ═══ Action Bar ═══

function updateActionBar(alert) {
    document.getElementById('actionAlertTitle').textContent = alert.title;

    const isActive = alert.status === 'active';
    document.getElementById('btnAcknowledge').disabled = !isActive;
    document.getElementById('btnInvestigate').disabled = !isActive;
    document.getElementById('btnDismiss').disabled = !isActive;
    document.getElementById('btnTP').disabled = false;
    document.getElementById('btnFP').disabled = false;
}

// ═══ Actions ═══

async function acknowledgeAlert() {
    if (!selectedAlertId) return;
    try {
        await fetch(`/api/alerts/${selectedAlertId}/acknowledge`, { method: 'POST' });
    } catch (e) {
        console.error('Acknowledge failed:', e);
    }
}

async function investigateAlert() {
    if (!selectedAlertId) return;
    // Open evidence in new tab (or highlight on map in future)
    const alert = alerts[selectedAlertId];
    if (alert) {
        console.log('Investigating:', alert.title);
    }
}

async function dismissAlert() {
    if (!selectedAlertId) return;
    try {
        await fetch(`/api/alerts/${selectedAlertId}/dismiss`, { method: 'POST' });
    } catch (e) {
        console.error('Dismiss failed:', e);
    }
}

async function submitFeedback(feedback) {
    if (!selectedAlertId) return;
    try {
        await fetch(`/api/alerts/${selectedAlertId}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback: feedback }),
        });

        // Update local state
        const alert = alerts[selectedAlertId];
        if (alert) {
            alert.operator_feedback = feedback;
            alert.status = 'resolved';
        }

        // Disable buttons
        document.getElementById('btnTP').disabled = true;
        document.getElementById('btnFP').disabled = true;
    } catch (e) {
        console.error('Feedback failed:', e);
    }
}

// ═══ Filters ═══

function filterAlerts(severity) {
    currentFilter = severity;

    // Update buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === severity);
    });

    // Show/hide cards
    document.querySelectorAll('.alert-card').forEach(card => {
        if (severity === 'all') {
            card.style.display = '';
        } else {
            const cardSeverity = Array.from(card.classList)
                .find(c => c.startsWith('severity-'))
                ?.replace('severity-', '');
            card.style.display = cardSeverity === severity ? '' : 'none';
        }
    });
}

// ═══ Utility ═══

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return '#ff3b5c';
    if (confidence >= 0.6) return '#fd7e14';
    if (confidence >= 0.4) return '#ffc107';
    return '#6c757d';
}

function getClassificationColor(classification) {
    switch (classification) {
        case 'CONFIRMED_ANOMALY': return '#ff3b5c';
        case 'SUSPICIOUS': return '#fd7e14';
        case 'FALSE_POSITIVE': return '#6c757d';
        default: return '#e8e8f0';
    }
}

function flashHeader(type) {
    const header = document.getElementById('header');
    const color = type === 'emergency' ? 'rgba(255,59,92,0.15)' : 'rgba(253,126,20,0.1)';
    header.style.background = color;
    setTimeout(() => {
        header.style.background = '';
    }, 2000);
}

// ═══ Initial Load ═══

async function loadExistingAlerts() {
    try {
        const response = await fetch('/api/alerts?limit=50');
        const alertsList = await response.json();

        if (alertsList.length > 0) {
            const emptyState = document.getElementById('emptyState');
            if (emptyState) emptyState.remove();
        }

        // Render in reverse order (oldest first, so newest is at top)
        alertsList.reverse().forEach(alert => {
            alerts[alert.alert_id] = alert;
            renderAlertCard(alert, false);
        });

        document.getElementById('alertsCount').textContent = alertsList.length;
    } catch (e) {
        console.error('Failed to load alerts:', e);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        document.getElementById('alertsCount').textContent = stats.total_alerts || 0;
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

// ═══ Init ═══

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadExistingAlerts();

    // Periodic status poll (backup for when WebSocket status updates aren't flowing)
    setInterval(async () => {
        try {
            const response = await fetch('/api/pipeline/status');
            const status = await response.json();
            handlePipelineStatus(status);
        } catch (e) {
            // Ignore — server may be busy
        }
    }, 5000);
});
