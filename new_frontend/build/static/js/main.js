// AI Surveillance System - Main JavaScript
const socket = io();

// State Management
let stats = {
    total: 0,
    accidents: 0,
    fires: 0,
    explosions: 0,
    fighting: 0,
    smoke: 0
};

let recentAlerts = [];
let timelineEvents = [];
let currentDetection = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    startClock();
});

function initializeApp() {
    console.log('🚀 AI Surveillance System initialized');
    
    // Connect to WebSocket
    socket.on('connect', () => {
        console.log('✅ Connected to server');
        showNotification('System Connected', 'Connected to AI Surveillance System', 'LOW');
    });
    
    socket.on('disconnect', () => {
        console.log('❌ Disconnected from server');
        showNotification('System Disconnected', 'Connection lost', 'HIGH');
    });
    
    // Listen for enhanced detection updates
    socket.on('detection_update', handleDetectionUpdate);
    
    // Listen for legacy detection events
    socket.on('detection', handleLegacyDetection);
    
    // Status updates
    socket.on('status', (msg) => {
        console.log('📊 Status:', msg);
    });
}

// Handle Enhanced Detection Updates
function handleDetectionUpdate(data) {
    console.log('🚨 Detection Update:', data);
    
    // Update current detection
    currentDetection = data;
    
    // Only process significant events
    if (data.event !== 'normal' && data.confidence >= 65) {
        updateStats(data.event);
        addToRecentAlerts(data);
        addToTimeline(data);
        showCurrentDetection(data);
        showDetectionOverlay(data);
        
        // Show notification for HIGH/CRITICAL
        if (data.risk === 'HIGH' || data.risk === 'CRITICAL') {
            showNotification(
                `${data.event} DETECTED!`,
                `Confidence: ${data.confidence.toFixed(1)}% | Risk: ${data.risk}`,
                data.risk,
                data
            );
        }
    } else {
        hideCurrentDetection();
    }
}

// Handle Legacy Detection
function handleLegacyDetection(payload) {
    console.log('📡 Legacy Detection:', payload);
    // Convert to new format if needed
    const data = {
        event: payload.event || 'UNKNOWN',
        confidence: payload.score ? payload.score * 100 : 0,
        risk: payload.risk || 'MEDIUM',
        timestamp: Date.now() / 1000,
        objects: { vehicles: 0, people: 0 }
    };
    handleDetectionUpdate(data);
}

// Update Statistics
function updateStats(event) {
    stats.total++;
    
    switch(event) {
        case 'ACCIDENT': stats.accidents++; break;
        case 'FIRE': stats.fires++; break;
        case 'EXPLOSION': stats.explosions++; break;
        case 'FIGHTING': stats.fighting++; break;
        case 'SMOKE': stats.smoke++; break;
    }
    
    // Update UI
    document.getElementById('stat-total').textContent = stats.total;
    document.getElementById('stat-accidents').textContent = stats.accidents;
    document.getElementById('stat-fires').textContent = stats.fires;
    document.getElementById('stat-explosions').textContent = stats.explosions;
    document.getElementById('stat-fighting').textContent = stats.fighting;
    document.getElementById('stat-smoke').textContent = stats.smoke;
    
    // Animate stat update
    animateStatCard(event.toLowerCase());
}

function animateStatCard(type) {
    const card = document.querySelector(`.stat-card.${type}`);
    if (card) {
        card.style.transform = 'scale(1.1)';
        setTimeout(() => {
            card.style.transform = 'scale(1)';
        }, 300);
    }
}

// Show Current Detection
function showCurrentDetection(data) {
    const section = document.getElementById('current-detection-section');
    const container = document.getElementById('current-detection');
    
    section.style.display = 'block';
    
    const icons = {
        'ACCIDENT': '🚗',
        'FIRE': '🔥',
        'EXPLOSION': '💥',
        'FIGHTING': '🥊',
        'SMOKE': '💨'
    };
    
    container.innerHTML = `
        <div class="detection-badge ${data.risk}">${data.risk}</div>
        <h4 style="margin: 0.5rem 0;">${icons[data.event] || '⚠️'} ${data.event}</h4>
        <div style="margin-bottom: 0.5rem;">
            <strong>Confidence:</strong> ${data.confidence.toFixed(1)}%
        </div>
        ${data.objects && data.objects.vehicles > 0 ? `
            <div style="margin-bottom: 0.5rem;">
                <strong>🚗 Vehicles:</strong> ${data.objects.vehicles}
            </div>
        ` : ''}
        ${data.objects && data.objects.people > 0 ? `
            <div style="margin-bottom: 0.5rem;">
                <strong>👥 People:</strong> ${data.objects.people}
            </div>
        ` : ''}
        ${data.motion_spike ? `
            <div style="margin-bottom: 0.5rem;">
                <strong>📊 Motion:</strong> ${data.motion_spike.toFixed(1)}
            </div>
        ` : ''}
    `;
}

function hideCurrentDetection() {
    setTimeout(() => {
        const section = document.getElementById('current-detection-section');
        if (currentDetection && currentDetection.event === 'normal') {
            section.style.display = 'none';
        }
    }, 3000);
}

// Add to Recent Alerts
function addToRecentAlerts(data) {
    recentAlerts.unshift(data);
    if (recentAlerts.length > 10) {
        recentAlerts.pop();
    }
    
    renderRecentAlerts();
}

function renderRecentAlerts() {
    const container = document.getElementById('recent-alerts');
    
    if (recentAlerts.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 1rem;">No alerts yet</div>';
        return;
    }
    
    container.innerHTML = recentAlerts.map(alert => {
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
        const icons = {
            'ACCIDENT': '🚗',
            'FIRE': '🔥',
            'EXPLOSION': '💥',
            'FIGHTING': '🥊',
            'SMOKE': '💨'
        };
        
        return `
            <div class="alert-item ${alert.event}">
                <div class="alert-header">
                    <span class="alert-type">${icons[alert.event] || '⚠️'} ${alert.event}</span>
                    <span class="alert-confidence">${alert.confidence.toFixed(1)}%</span>
                </div>
                <div class="alert-details">
                    Risk: ${alert.risk}
                    ${alert.objects && alert.objects.vehicles > 0 ? `| 🚗 ${alert.objects.vehicles}` : ''}
                    ${alert.objects && alert.objects.people > 0 ? `| 👥 ${alert.objects.people}` : ''}
                </div>
                <div class="alert-time">${time}</div>
            </div>
        `;
    }).join('');
}

// Add to Timeline
function addToTimeline(data) {
    timelineEvents.unshift(data);
    if (timelineEvents.length > 20) {
        timelineEvents.pop();
    }
    
    renderTimeline();
}

function renderTimeline() {
    const container = document.getElementById('timeline');
    
    if (timelineEvents.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 1rem; width: 100%;">No events yet</div>';
        return;
    }
    
    container.innerHTML = timelineEvents.map(event => {
        const time = new Date(event.timestamp * 1000).toLocaleTimeString();
        const icons = {
            'ACCIDENT': '🚗',
            'FIRE': '🔥',
            'EXPLOSION': '💥',
            'FIGHTING': '🥊',
            'SMOKE': '💨'
        };
        
        return `
            <div class="timeline-item ${event.event}">
                <div class="timeline-event">${icons[event.event] || '⚠️'} ${event.event}</div>
                <div class="timeline-time">${time}</div>
                <div class="timeline-conf">${event.confidence.toFixed(1)}%</div>
            </div>
        `;
    }).join('');
    
    // Auto-scroll to latest
    container.scrollLeft = 0;
}

// Show Detection Overlay on Video
function showDetectionOverlay(data) {
    const overlay = document.getElementById('detection-overlay');
    const icons = {
        'ACCIDENT': '🚗',
        'FIRE': '🔥',
        'EXPLOSION': '💥',
        'FIGHTING': '🥊',
        'SMOKE': '💨'
    };
    
    overlay.textContent = `${icons[data.event] || '⚠️'} ${data.event} DETECTED!`;
    overlay.className = `detection-overlay show`;
    
    // Change color based on risk
    if (data.risk === 'CRITICAL' || data.risk === 'HIGH') {
        overlay.style.background = 'rgba(255, 71, 87, 0.95)';
    } else if (data.risk === 'MEDIUM') {
        overlay.style.background = 'rgba(255, 165, 2, 0.95)';
    } else {
        overlay.style.background = 'rgba(38, 222, 129, 0.95)';
    }
    
    // Hide after 3 seconds
    setTimeout(() => {
        overlay.classList.remove('show');
    }, 3000);
}

// Show Notification
function showNotification(title, message, risk, data = null) {
    const container = document.getElementById('notifications');
    const notification = document.createElement('div');
    notification.className = `notification ${risk}`;
    
    const id = Date.now();
    notification.id = `notification-${id}`;
    
    notification.innerHTML = `
        <div class="notification-header">
            <div class="notification-title">${title}</div>
            <button class="notification-close" onclick="closeNotification('${id}')">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="notification-body">${message}</div>
        ${data && data.objects ? `
            <div class="notification-body" style="margin-top: 0.5rem;">
                ${data.objects.vehicles > 0 ? `🚗 Vehicles: ${data.objects.vehicles} | ` : ''}
                ${data.objects.people > 0 ? `👥 People: ${data.objects.people}` : ''}
            </div>
        ` : ''}
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        closeNotification(id);
    }, 10000);
    
    // Play alert sound
    playAlertSound();
}

function closeNotification(id) {
    const notification = document.getElementById(`notification-${id}`);
    if (notification) {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }
}

// Play Alert Sound
function playAlertSound() {
    // Create a simple beep sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

// Setup Event Listeners
function setupEventListeners() {
    // Fullscreen button
    document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);
    
    // Snapshot button
    document.getElementById('btn-snapshot').addEventListener('click', takeSnapshot);
    
    // Clear timeline
    document.getElementById('btn-clear-timeline').addEventListener('click', clearTimeline);
    
    // Modal close
    document.getElementById('modal-close').addEventListener('click', closeModal);
    
    // Close modal on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

// Fullscreen
function toggleFullscreen() {
    const modal = document.getElementById('fullscreen-modal');
    modal.classList.toggle('show');
}

function closeModal() {
    const modal = document.getElementById('fullscreen-modal');
    modal.classList.remove('show');
}

// Take Snapshot
function takeSnapshot() {
    const video = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    canvas.width = video.naturalWidth;
    canvas.height = video.naturalHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Download image
    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `snapshot-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
        
        showNotification('Snapshot Saved', 'Screenshot saved successfully', 'LOW');
    });
}

// Clear Timeline
function clearTimeline() {
    if (confirm('Are you sure you want to clear the timeline?')) {
        timelineEvents = [];
        renderTimeline();
        showNotification('Timeline Cleared', 'All events removed', 'LOW');
    }
}

// Start Clock
function startClock() {
    function updateClock() {
        const now = new Date();
        const timeString = now.toLocaleString('en-US', {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        document.getElementById('current-time').textContent = timeString;
    }
    
    updateClock();
    setInterval(updateClock, 1000);
}

// FPS Counter (optional)
let lastFrameTime = Date.now();
let fps = 0;

function updateFPS() {
    const now = Date.now();
    fps = Math.round(1000 / (now - lastFrameTime));
    lastFrameTime = now;
    document.getElementById('fps-counter').textContent = `FPS: ${fps}`;
}

setInterval(updateFPS, 1000);

// Export for debugging
window.surveillanceSystem = {
    stats,
    recentAlerts,
    timelineEvents,
    currentDetection,
    socket
};

console.log('✅ AI Surveillance System Ready');
console.log('📊 Access system data via: window.surveillanceSystem');
