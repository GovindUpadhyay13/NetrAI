// NetrAI Delhi Police Operations Center Dashboard Logic

let map;
let cameraMarkers = [];
let activeIncidentCircle = null;
let camerasData = [];

document.addEventListener('DOMContentLoaded', async () => {
    await initDelhiMap();
    setupEventListeners();
    fetchLiveTraces();
    setInterval(fetchLiveTraces, 3000);
});

async function initDelhiMap() {
    // Center of South Delhi (Hauz Khas / Saket region)
    const southDelhiCoords = [28.5450, 77.2000];

    // Initialize Leaflet map with CartoDB Positron clean light tiles (matching reference UI)
    map = L.map('delhi-map', {
        zoomControl: false
    }).setView(southDelhiCoords, 13);

    L.control.zoom({ position: 'topright' }).addTo(map);

    // Base tile layer: ESRI World Light Gray (clean modern aesthetic, zero watermarks, 100% free, no API key needed)
    const esriBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri & OpenStreetMap contributors',
        maxZoom: 16
    }).addTo(map);

    const esriLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 16
    }).addTo(map);

    // Standard OpenStreetMap layer as an alternative option
    const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    });

    // Layer control for instant toggling
    L.control.layers({
        "Clean Minimalist (No Key)": L.layerGroup([esriBase, esriLabels]),
        "OpenStreetMap Street View": osmLayer
    }, null, { position: 'bottomright' }).addTo(map);


        let res = await fetch('/dashboard/delhi_cams.json').catch(() => null);
        if (!res || !res.ok) res = await fetch('delhi_cams.json');
        camerasData = await res.json();
    } catch (e) {
        camerasData = [
            { id: "CAM-SD-01", name: "Hauz Khas Village - Main Gate", lat: 28.5535, lng: 77.1945, threat_level: "high", last_anomaly: "Distress SOS Signaling" },
            { id: "CAM-SD-04", name: "Green Park North Lane", lat: 28.5586, lng: 77.2045, threat_level: "high", last_anomaly: "Stalking Pattern" },
            { id: "CAM-SD-08", name: "Deer Park Lake Walking Trail", lat: 28.5562, lng: 77.1889, threat_level: "high", last_anomaly: "Distress Gesture Flagged" }
        ];
    }

    renderCrossCameraTrajectory();
}

let trajectoryPolylineGlow = null;
let trajectoryPolylineFlow = null;
let projectedVectorLine = null;
let projectedTargetMarker = null;

function renderCrossCameraTrajectory() {
    // 1. Remove any previous markers / layers
    cameraMarkers.forEach(m => map.removeLayer(m));
    cameraMarkers = [];
    if (trajectoryPolylineGlow) map.removeLayer(trajectoryPolylineGlow);
    if (trajectoryPolylineFlow) map.removeLayer(trajectoryPolylineFlow);
    if (projectedVectorLine) map.removeLayer(projectedVectorLine);
    if (projectedTargetMarker) map.removeLayer(projectedTargetMarker);
    if (activeIncidentCircle) map.removeLayer(activeIncidentCircle);

    // 2. DEFINE THE CHRONOLOGICAL CROSS-CAMERA TRAJECTORY
    // Point 1: Distress Origin (Hauz Khas Village)
    // Point 2: First Re-ID Sighting (Deer Park Lake Trail)
    // Point 3: Active Cross-Camera Sighting (Green Park North)
    const trajectoryWaypoints = [
        {
            step: 1,
            id: "CAM-SD-01",
            name: "Hauz Khas Village - Main Gate",
            lat: 28.5535,
            lng: 77.1945,
            time: "20:28:14",
            type: "origin",
            label: "1. SOS ORIGIN",
            details: "Distress gesture raised by victim. Flagged by MediaPipe & AnomalyCLIP."
        },
        {
            step: 2,
            id: "CAM-SD-08",
            name: "Deer Park Lake Walking Trail",
            lat: 28.5562,
            lng: 77.1889,
            time: "20:30:22",
            type: "match",
            label: "2. RE-ID SIGHTED",
            similarity: "99.8%",
            details: "Subject tracked exiting lake trail corridor. Qdrant cosine similarity: 99.8%."
        },
        {
            step: 3,
            id: "CAM-SD-04",
            name: "Green Park Market - North Lane",
            lat: 28.5586,
            lng: 77.2045,
            time: "20:32:05",
            type: "active-vector",
            label: "3. CURRENT POSITION",
            similarity: "99.4%",
            details: "Live sighting. Subject moving towards Green Park Metro corridor."
        }
    ];

    const coords = trajectoryWaypoints.map(w => [w.lat, w.lng]);

    // 3. DRAW GLOWING OUTER TRAJECTORY LINE
    trajectoryPolylineGlow = L.polyline(coords, {
        color: '#ef4444',
        weight: 9,
        opacity: 0.35,
        lineCap: 'round',
        lineJoin: 'round'
    }).addTo(map);

    // 4. DRAW ANIMATED INNER FLOWING DASHED LINE
    trajectoryPolylineFlow = L.polyline(coords, {
        color: '#dc2626',
        weight: 3.5,
        opacity: 0.95,
        className: 'trajectory-flow',
        dashArray: '10, 10'
    }).addTo(map);

    // 5. DRAW PROJECTED ESCAPE VECTOR TOWARDS GREEN PARK METRO
    const activePoint = trajectoryWaypoints[2];
    const metroCoords = [28.5615, 77.2085];
    projectedVectorLine = L.polyline([[activePoint.lat, activePoint.lng], metroCoords], {
        color: '#f59e0b',
        weight: 2.5,
        dashArray: '5, 6',
        opacity: 0.8
    }).addTo(map);

    projectedTargetMarker = L.circleMarker(metroCoords, {
        radius: 7,
        color: '#f59e0b',
        fillColor: '#fef3c7',
        fillOpacity: 0.9,
        weight: 2
    }).addTo(map).bindPopup(`
        <div style="font-family: 'Inter', sans-serif; font-size: 11px;">
            <strong style="color: #b45309;">⚡ RECOMMENDED INTERCEPT POINT</strong><br>
            <span>Green Park Metro Gate #2 / Outer Ring Rd</span><br>
            <span style="color: #16a34a; font-weight: 700;">PCR Echo-14 En Route (ETA: 1.5 min)</span>
        </div>
    `);

    // 6. RENDER NUMBERED WAYPOINT MARKERS (WITHOUT ANY NORMAL CAMERAS)
    trajectoryWaypoints.forEach(wp => {
        const isOrigin = wp.type === 'origin';
        const isMatch = wp.type === 'match';
        const isActive = wp.type === 'active-vector';

        const bubbleClass = isOrigin ? 'origin' : isMatch ? 'match' : 'active-vector';
        const iconChar = isOrigin ? 'SOS' : isMatch ? wp.step : '🎯';

        const customHtml = `
            <div class="waypoint-pin-container">
                <div class="waypoint-pin-bubble ${bubbleClass}">
                    ${iconChar}
                </div>
                <div class="waypoint-pin-tag">
                    ${wp.label}
                </div>
            </div>
        `;

        const waypointIcon = L.divIcon({
            html: customHtml,
            className: 'custom-waypoint-pin',
            iconSize: [80, 56],
            iconAnchor: [40, 24]
        });

        const marker = L.marker([wp.lat, wp.lng], { icon: waypointIcon }).addTo(map);

        marker.bindPopup(`
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; min-width: 200px; padding: 2px;">
                <div style="font-size: 10px; color: #64748b; font-weight: 700; text-transform: uppercase;">
                    Checkpoint ${wp.step} &bull; ${wp.time}
                </div>
                <strong style="color: #0f172a; font-size: 13px;">${wp.name}</strong><br>
                <span style="font-size: 11px; color: #334155;">Camera: <code>${wp.id}</code></span>
                ${wp.similarity ? `<div style="margin: 4px 0; color: #b45309; font-weight: 700; font-size: 11px;">Cross-Camera Match: ${wp.similarity}</div>` : ''}
                <div style="margin-top: 6px; padding: 6px 8px; border-radius: 4px; background: #f8fafc; border: 1px solid #e2e8f0; font-size: 11px; color: #475569;">
                    ${wp.details}
                </div>
            </div>
        `);

        marker.on('click', () => {
            document.getElementById('current-sublocation-label').innerText = `${wp.name} (${wp.id}) [Step ${wp.step}]`;
            map.flyTo([wp.lat, wp.lng], 15, { duration: 1.0 });
        });

        cameraMarkers.push(marker);
    });

    // 7. RADAR RIPPLE ON ACTIVE/CURRENT SIGHTING (GREEN PARK)
    activeIncidentCircle = L.circle([activePoint.lat, activePoint.lng], {
        radius: 300,
        color: '#ef4444',
        fillColor: '#fca5a5',
        fillOpacity: 0.25,
        weight: 1.5,
        dashArray: '4, 4'
    }).addTo(map);

    // 8. AUTO FIT BOUNDS TO ENTIRE PURSUIT TRAJECTORY
    const bounds = L.latLngBounds(coords.concat([metroCoords]));
    map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
}

function focusTrajectoryPath() {
    if (cameraMarkers.length > 0) {
        const coords = [
            [28.5535, 77.1945],
            [28.5562, 77.1889],
            [28.5586, 77.2045],
            [28.5615, 77.2085]
        ];
        map.fitBounds(L.latLngBounds(coords), { padding: [50, 50], maxZoom: 15 });
    }
}




async function fetchLiveTraces() {
    try {
        const res = await fetch('/api/traces?limit=15');
        if (!res.ok) return;
        const traces = await res.json();

        const vlmAnalyzed = traces.find(t => t.stage === 'vlm_analyzed' || t.vlm_report);
        if (vlmAnalyzed) {
            document.getElementById('live-vlm-report').innerText = vlmAnalyzed.vlm_report;
            document.getElementById('incident-event-id').innerText = `Event ID: ${vlmAnalyzed.incident_id.slice(0, 8).toUpperCase()}`;
            document.getElementById('chk-sos-flag').innerText = vlmAnalyzed.distress_gesture ? 'Yes (SOS Confirmed)' : 'Monitoring';
            document.getElementById('chk-anomaly-type').innerText = vlmAnalyzed.anomaly_type || 'Stalking / Harassment';
            document.getElementById('chk-score').innerText = `${Math.round(vlmAnalyzed.anomaly_score * 100)}%`;
        }

        const reidMatches = traces.filter(t => t.stage === 'reid_match').length;
        document.getElementById('stat-reid-matches').innerText = `${reidMatches} Matches`;

    } catch (e) {
        // Standalone browser mode without active backend
    }
}

function setupEventListeners() {
    document.getElementById('btn-refresh-live').addEventListener('click', () => {
        fetchLiveTraces();
        const btn = document.getElementById('btn-refresh-live');
        btn.innerHTML = '<span>⚡ Updating...</span>';
        setTimeout(() => {
            btn.innerHTML = '<span>Rest Change</span>';
        }, 800);
    });

    const focusBtn = document.getElementById('btn-focus-path');
    if (focusBtn) {
        focusBtn.addEventListener('click', () => {
            focusTrajectoryPath();
        });
    }
}

