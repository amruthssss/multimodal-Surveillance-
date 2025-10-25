const socket = io();

// 📊 Detection statistics
let stats = {
  total: 0,
  accidents: 0,
  fires: 0,
  explosions: 0,
  fighting: 0,
  smoke: 0
};

function riskBadge(risk){
  const map = {high:'badge-risk-high', medium:'badge-risk-medium', low:'badge-risk-low'};
  return `<span class="badge ${map[risk]||'bg-secondary'}">${risk}</span>`;
}

function updateStats(event) {
// 🚀 Enhanced detection system integration (80% Agent + 20% YOLO)
socket.on('detection_update', data => {
  console.log('🚨 Enhanced Detection:', data);
  
  // Update current detection display
  updateCurrentDetection(data);
  
  // Only alert for actual events (not 'normal')
  if (data.event !== 'normal' && data.confidence >= 65) {
    updateStats(data.event);
    addEnhancedAlert(data);
    playAlertSound();
  }
});
  // Update UI
  const update = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  
  update('stat-total', stats.total);
  update('stat-accidents', stats.accidents);
  update('stat-fires', stats.fires);
  update('stat-explosions', stats.explosions);
  update('stat-fighting', stats.fighting);
  update('stat-smoke', stats.smoke);
}

function updateCurrentDetection(data) {
  const card = document.getElementById('current-detection-card');
  const content = document.getElementById('current-detection-content');
  
  if (!card || !content) return;
  
  if (data.event !== 'normal' && data.confidence >= 65) {
    card.style.display = 'block';
    
    const riskColors = {
      'CRITICAL': 'danger',
      'HIGH': 'danger',
      'MEDIUM': 'warning',
      'LOW': 'success'
    };
    
    content.innerHTML = `
      <div class="alert alert-${riskColors[data.risk] || 'secondary'} mb-0">
        <div class="d-flex justify-content-between align-items-center">
          <div>
            <h5 class="mb-2">🚨 ${data.event}</h5>
            <div class="row">
              <div class="col-md-3">
                <strong>Confidence:</strong> ${data.confidence.toFixed(1)}%
              </div>
              <div class="col-md-3">
                <strong>Risk:</strong> ${data.risk}
              </div>
              ${data.objects.vehicles > 0 ? `
              <div class="col-md-3">
                <strong>🚗 Vehicles:</strong> ${data.objects.vehicles}
              </div>` : ''}
              ${data.objects.people > 0 ? `
              <div class="col-md-3">
                <strong>👥 People:</strong> ${data.objects.people}
              </div>` : ''}
            </div>
            ${data.motion_spike ? `<div class="mt-2"><strong>📊 Motion Spike:</strong> ${data.motion_spike.toFixed(1)}</div>` : ''}
          </div>
        </div>
      </div>
    `;
  } else {
    // Hide if normal
    setTimeout(() => card.style.display = 'none', 2000);
  }
}

socket.on('status', msg => console.log('Status:', msg));

socket.on('inference', msg => {
  // generic inference event from app.py (objects)
});

socket.on('detection', payload => {
  addLiveAlert(payload);
  playAlertSound();
});

// 🚀 Enhanced detection system integration (80% Agent + 20% YOLO)
socket.on('detection_update', data => {
  console.log('🚨 Enhanced Detection:', data);
  
  // Only alert for actual events (not 'normal')
  if (data.event !== 'normal' && data.confidence >= 65) {
    addEnhancedAlert(data);
    playAlertSound();
  }
});

function addLiveAlert(p){
  const container = document.getElementById('live-alerts');
  if(!container) return;
  const div = document.createElement('div');
  div.className = 'col-md-3';
  div.innerHTML = `<div class="card alert-popup border-${p.risk==='high'?'danger':(p.risk==='medium'?'warning':'success')}">
    <div class="card-body p-2">
      <h6 class="card-title mb-1">${p.event} ${riskBadge(p.risk)}</h6>
      <small class="text-muted">Score: ${p.score.toFixed(2)}</small>
    </div>
  </div>`;
  container.prepend(div);
  setTimeout(()=> div.remove(), 15000);
}

// 🚀 Enhanced alert display (for detection_update events)
function addEnhancedAlert(data){
  const container = document.getElementById('live-alerts');
  if(!container) return;
  
// 🔔 Show popup notification for critical alerts
function showPopupNotification(data){
  const popup = document.getElementById('alert-popups');
  if(!popup) return;
  
  const div = document.createElement('div');
  div.className = 'alert alert-danger alert-dismissible fade show shadow-lg';
  div.setAttribute('role', 'alert');
  div.innerHTML = `
    <strong>⚠️ ${data.risk} ALERT!</strong>
    <div>${data.event} detected with ${data.confidence.toFixed(1)}% confidence</div>
    ${data.objects.vehicles > 0 ? `<small>🚗 ${data.objects.vehicles} vehicle(s)</small>` : ''}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  
  popup.appendChild(div);
  
  // Auto-dismiss after 10 seconds
  setTimeout(() => {
    div.classList.remove('show');
    setTimeout(() => div.remove(), 150);
  }, 10000);
}

// Camera grid demo placeholder (since cameras stored server-side in memory)
window.addEventListener('DOMContentLoaded', () => {
  const grid = document.getElementById('camera-grid');
  if(grid){
    // Could fetch camera list via API; placeholder creates example boxes
  }
});
  const borderColor = riskColors[data.risk] || 'secondary';
  const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
  
  const div = document.createElement('div');
  div.className = 'col-md-4 col-lg-3';
  div.innerHTML = `<div class="card alert-popup border-${borderColor} mb-2">
    <div class="card-body p-3">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="card-title mb-0">🚨 ${data.event}</h6>
        <span class="badge bg-${borderColor}">${data.risk}</span>
      </div>
      <div class="small text-muted">
        <div class="mb-1">
          <strong>Confidence:</strong> ${data.confidence.toFixed(1)}%
        </div>
        ${data.objects.vehicles > 0 ? `<div class="mb-1">🚗 Vehicles: ${data.objects.vehicles}</div>` : ''}
        ${data.objects.people > 0 ? `<div class="mb-1">👥 People: ${data.objects.people}</div>` : ''}
        ${data.motion_spike ? `<div class="mb-1">📊 Motion: ${data.motion_spike.toFixed(1)}</div>` : ''}
        <div class="mt-2 text-truncate" style="font-size: 0.75rem;">
          ${data.reasoning || ''}
        </div>
        <div class="mt-1" style="font-size: 0.7rem;">
          ⏰ ${timestamp}
        </div>
      </div>
    </div>
  </div>`;
  
  container.prepend(div);
  
  // Auto-remove after 20 seconds
  setTimeout(()=> div.remove(), 20000);
  
  // Show popup notification for HIGH/CRITICAL
  if (data.risk === 'HIGH' || data.risk === 'CRITICAL') {
    showPopupNotification(data);
  }
}

function playAlertSound(){
  const audio = new Audio('/static/assets/alert_sound.mp3');
  audio.play().catch(()=>{});
}

// Camera grid demo placeholder (since cameras stored server-side in memory)
window.addEventListener('DOMContentLoaded', () => {
  const grid = document.getElementById('camera-grid');
  if(grid){
    // Could fetch camera list via API; placeholder creates example boxes
  }
});
