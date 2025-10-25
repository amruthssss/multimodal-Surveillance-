import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../utils/api';
import './Dashboard.css';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('live');
  const [cameraConfigured, setCameraConfigured] = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [config, setConfig] = useState(null);
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);
  
  // Camera configuration state
  const [cameraSource, setCameraSource] = useState({
    type: 'builtin',
    url: '',
    username: '',
    password: ''
  });

  // Alert configuration state
  const [alertConfig, setAlertConfig] = useState({
    emailAlerts: {
      enabled: true,
      highRiskThreshold: 0.7
    },
    smsAlerts: {
      enabled: false,
      mediumRiskThreshold: 0.3
    }
  });

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await api.get('/config');
      setConfig(response.data.data);
      if (response.data.data.cameraSource) {
        setCameraSource(response.data.data.cameraSource);
        setCameraConfigured(true);
      }
      setAlertConfig({
        emailAlerts: response.data.data.emailAlerts,
        smsAlerts: response.data.data.smsAlerts
      });
    } catch (error) {
      console.error('Failed to fetch config:', error);
    }
  };

  const handleSaveConfiguration = async () => {
    try {
      await api.post('/config', alertConfig);
      alert('Configuration saved successfully!');
    } catch (error) {
      console.error('Failed to save config:', error);
      alert('Failed to save configuration');
    }
  };

  const handleConfigureCamera = async () => {
    try {
      // Prepare configuration data for Flask backend
      const configData = {
        source: cameraSource.type,
        rtspUrl: cameraSource.type === 'rtsp' ? cameraSource.url : '',
        ipUrl: cameraSource.type === 'ip' ? cameraSource.url : '',
        youtubeUrl: cameraSource.type === 'youtube' ? cameraSource.url : '',
        videoFilePath: cameraSource.type === 'file' ? cameraSource.url : ''
      };

      console.log('Configuring camera with:', configData);

      // Send to Flask backend (port 5001)
      const response = await fetch('http://localhost:5001/api/camera/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(configData)
      });

      const result = await response.json();
      
      if (result.success) {
        setCameraConfigured(true);
        setShowCameraModal(false);
        
        // If built-in camera selected, start it immediately
        if (cameraSource.type === 'builtin') {
          await startBuiltInCamera();
        } else {
          alert(`✅ Camera configured successfully!\n\nSource: ${cameraSource.type}\nVideo feed will load in the Live Feeds tab.`);
        }
      } else {
        alert(`❌ Configuration failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to configure camera:', error);
      alert(`❌ Failed to configure camera: ${error.message}`);
    }
  };

  const startBuiltInCamera = async () => {
    try {
      // Stop any existing stream
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
          facingMode: 'user'
        },
        audio: false
      });
      
      setVideoStream(stream);
      
      // If video element exists, attach stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      
      console.log('Built-in camera started successfully:', stream.getVideoTracks()[0].getSettings());
      alert('✅ Built-in camera activated! Stream ready for monitoring.');
    } catch (error) {
      console.error('Camera access error:', error);
      if (error.name === 'NotAllowedError') {
        alert('❌ Camera access denied. Please allow camera permissions in your browser settings.');
      } else if (error.name === 'NotFoundError') {
        alert('❌ No camera found on this device.');
      } else {
        alert('❌ Failed to access camera: ' + error.message);
      }
    }
  };

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [videoStream]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header glass">
        <div className="header-left">
          <div>
            <h1 className="header-title">Multi-Modal Surveillance System</h1>
            <p className="header-subtitle">Real-time AI-powered threat detection and monitoring</p>
          </div>
        </div>

        <div className="header-status">
          <div className="status-item">
            <span className="status-label">Threat Level</span>
            <span className="status-value">
              <span className="status-indicator status-green"></span>
              LOW
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Camera Status</span>
            <span className="status-value">
              <span className="status-indicator status-green"></span>
              Camera Online
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">FPS</span>
            <span className="status-value">30 FPS</span>
          </div>
          <div className="status-item">
            <span className="status-label">Latency</span>
            <span className="status-value">45ms</span>
          </div>
        </div>

        <div className="header-actions">
          <button className="icon-button" title="Refresh">
            🔄
          </button>
          <button className="icon-button" title="Video">
            📹
          </button>
          <button className="icon-button" title="Alerts">
            🔔
          </button>
          <button className="icon-button" onClick={handleLogout} title="Logout">
            👤
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="dashboard-nav glass">
        <button
          className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          LIVE FEEDS
        </button>
        <button
          className={`nav-tab ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
        >
          ALERTS & EVENTS
        </button>
        <button
          className={`nav-tab ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          ANALYTICS
        </button>
        <button
          className={`nav-tab ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          CONFIGURATION
        </button>
      </nav>

      {/* Main Content */}
      <main className="dashboard-content">
        {/* Live Feeds Tab */}
        {activeTab === 'live' && (
          <div className="tab-content fade-in">
            <div className="content-grid">
              <div className="main-view glass">
                <div className="view-header">
                  <h2>Live Camera Feed</h2>
                  <button 
                    className="icon-button"
                    onClick={() => setShowCameraModal(true)}
                  >
                    ⚙️
                  </button>
                </div>
                {!cameraConfigured ? (
                  <div className="empty-state">
                    <div className="empty-icon">📹</div>
                    <h3>Camera Not Configured</h3>
                    <p>Configure your camera source to start monitoring</p>
                    <button 
                      className="btn btn-primary"
                      onClick={() => setShowCameraModal(true)}
                    >
                      CONFIGURE CAMERA
                    </button>
                  </div>
                ) : (
                  <div className="video-container">
                    {cameraSource.type === 'builtin' ? (
                      <video 
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="video-feed"
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    ) : (
                      <img 
                        src="http://localhost:5001/video_feed" 
                        alt="Live Feed"
                        className="video-feed"
                      />
                    )}
                  </div>
                )}
              </div>

              <div className="sidebar">
                <div className="sidebar-card glass">
                  <h3>AI Agents</h3>
                  <div className="agent-list">
                    <div className="agent-item">
                      <div className="agent-info">
                        <span className="agent-name">Hybrid AI (Visual Analysis)</span>
                        <span className="agent-percentage">75%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: '75%' }}></div>
                      </div>
                    </div>
                    <div className="agent-item">
                      <div className="agent-info">
                        <span className="agent-name">Emotion Model</span>
                        <span className="agent-percentage">25%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: '25%' }}></div>
                      </div>
                    </div>
                    <div className="agent-item">
                      <div className="agent-info">
                        <span className="agent-name">Action Detection</span>
                        <span className="agent-percentage">25%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: '25%' }}></div>
                      </div>
                    </div>
                    <div className="agent-item">
                      <div className="agent-info">
                        <span className="agent-name">Audio Analysis</span>
                        <span className="agent-percentage">0%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: '0%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="sidebar-card glass">
                  <h3>Recent Alerts</h3>
                  <div className="empty-alerts">
                    <p>No recent alerts</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Alerts & Events Tab */}
        {activeTab === 'alerts' && (
          <div className="tab-content fade-in">
            <div className="alerts-container glass">
              <div className="alerts-header">
                <h2>Security Events</h2>
                <div className="alerts-actions">
                  <input 
                    type="text" 
                    className="input search-input" 
                    placeholder="Search events..."
                  />
                  <button className="btn btn-secondary">FILTER</button>
                </div>
              </div>
              <table className="events-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Event Type</th>
                    <th>Risk Level</th>
                    <th>Confidence</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td colSpan="5" className="empty-row">No events found</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="tab-content fade-in">
            <div className="analytics-grid">
              <div className="chart-card glass">
                <h3>Hourly Detections</h3>
                <div className="chart-placeholder">
                  <div className="chart-empty">
                    <div className="chart-icon">📊</div>
                    <p>No data available</p>
                  </div>
                </div>
              </div>
              <div className="chart-card glass">
                <h3>Threat Distribution</h3>
                <div className="chart-placeholder">
                  <div className="chart-empty">
                    <div className="chart-icon">📈</div>
                    <p>No data available</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Tab */}
        {activeTab === 'config' && (
          <div className="tab-content fade-in">
            <div className="config-container">
              <div className="config-section glass">
                <h3>Email Alerts</h3>
                <div className="config-item">
                  <label>High Risk Threshold</label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={alertConfig.emailAlerts.highRiskThreshold}
                      onChange={(e) => setAlertConfig({
                        ...alertConfig,
                        emailAlerts: {
                          ...alertConfig.emailAlerts,
                          highRiskThreshold: parseFloat(e.target.value)
                        }
                      })}
                      className="slider"
                    />
                    <span className="slider-value">
                      {alertConfig.emailAlerts.highRiskThreshold.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="config-section glass">
                <h3>SMS Alerts</h3>
                <div className="config-item">
                  <label>Medium Risk Threshold</label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={alertConfig.smsAlerts.mediumRiskThreshold}
                      onChange={(e) => setAlertConfig({
                        ...alertConfig,
                        smsAlerts: {
                          ...alertConfig.smsAlerts,
                          mediumRiskThreshold: parseFloat(e.target.value)
                        }
                      })}
                      className="slider"
                    />
                    <span className="slider-value">
                      {alertConfig.smsAlerts.mediumRiskThreshold.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>

              <button 
                className="btn btn-primary"
                onClick={handleSaveConfiguration}
              >
                SAVE CONFIGURATION
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Camera Configuration Modal */}
      {showCameraModal && (
        <div className="modal-overlay" onClick={() => setShowCameraModal(false)}>
          <div className="modal glass" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Camera Source Configuration</h2>
              <button 
                className="close-button"
                onClick={() => setShowCameraModal(false)}
              >
                ✕
              </button>
            </div>
            <div className="modal-content">
              <div className="source-options">
                {['builtin', 'rtsp', 'ip', 'youtube', 'file'].map((type) => (
                  <button
                    key={type}
                    className={`source-option ${cameraSource.type === type ? 'active' : ''}`}
                    onClick={() => setCameraSource({ ...cameraSource, type })}
                  >
                    {type === 'builtin' && '📹 Built-in Camera'}
                    {type === 'rtsp' && '📡 RTSP Stream'}
                    {type === 'ip' && '🌐 IP Camera (HTTP)'}
                    {type === 'youtube' && '▶️ YouTube Video'}
                    {type === 'file' && '📁 Video File'}
                  </button>
                ))}
              </div>

              {cameraSource.type === 'rtsp' && (
                <div className="form-group">
                  <label>RTSP URL</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="rtsp://username:password@ip:port/stream"
                    value={cameraSource.url}
                    onChange={(e) => setCameraSource({ ...cameraSource, url: e.target.value })}
                  />
                </div>
              )}

              {cameraSource.type === 'ip' && (
                <>
                  <div className="form-group">
                    <label>IP Camera URL</label>
                    <input
                      type="text"
                      className="input"
                      placeholder="http://192.168.1.100/video"
                      value={cameraSource.url}
                      onChange={(e) => setCameraSource({ ...cameraSource, url: e.target.value })}
                    />
                  </div>
                  <div className="form-row">
                    <div className="form-group">
                      <label>Username</label>
                      <input
                        type="text"
                        className="input"
                        placeholder="admin"
                        value={cameraSource.username}
                        onChange={(e) => setCameraSource({ ...cameraSource, username: e.target.value })}
                      />
                    </div>
                    <div className="form-group">
                      <label>Password</label>
                      <input
                        type="password"
                        className="input"
                        placeholder="password"
                        value={cameraSource.password}
                        onChange={(e) => setCameraSource({ ...cameraSource, password: e.target.value })}
                      />
                    </div>
                  </div>
                </>
              )}

              {cameraSource.type === 'youtube' && (
                <div className="form-group">
                  <label>YouTube Video URL</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={cameraSource.url}
                    onChange={(e) => setCameraSource({ ...cameraSource, url: e.target.value })}
                  />
                </div>
              )}

              {cameraSource.type === 'file' && (
                <div className="form-group">
                  <label>Video File Path</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="C:/path/to/video.mp4"
                    value={cameraSource.url}
                    onChange={(e) => setCameraSource({ ...cameraSource, url: e.target.value })}
                  />
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowCameraModal(false)}
              >
                CANCEL
              </button>
              <button 
                className="btn btn-primary"
                onClick={handleConfigureCamera}
              >
                SAVE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
