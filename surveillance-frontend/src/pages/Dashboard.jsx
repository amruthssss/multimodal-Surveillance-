import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { useAuth } from '../context/AuthContext';
import api from '../utils/api';
import './Dashboard.css';

const AI_API_URL = import.meta.env.VITE_AI_URL || 'http://localhost:5000';
const sources = [
  ['builtin', 'Built-in camera', 'Local device'], ['rtsp', 'RTSP stream', 'Network camera'],
  ['ip', 'IP camera', 'HTTP stream'], ['youtube', 'YouTube', 'Public stream'], ['file', 'Video file', 'Local recording'],
];
const emptyStatus = { hybridAI: { active: false }, cameraStatus: { online: false, fps: 0, latency: null } };
const formatTime = (stamp) => stamp ? new Date(stamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
const riskClass = (risk = '') => `risk-${String(risk).toLowerCase()}`;

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [tab, setTab] = useState('overview');
  const [status, setStatus] = useState(emptyStatus);
  const [events, setEvents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [evaluation, setEvaluation] = useState({ labeledReviews: 0, metrics: {} });
  const [reviews, setReviews] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [showCamera, setShowCamera] = useState(false);
  const [source, setSource] = useState('builtin');
  const [sourceUrl, setSourceUrl] = useState('');
  const [micOn, setMicOn] = useState(false);
  const [audioEvent, setAudioEvent] = useState('');
  const audioCleanup = useRef(null);
  const [query, setQuery] = useState('');
  const [alertConfig, setAlertConfig] = useState({ emailAlerts: { enabled: true, highRiskThreshold: 0.7 }, smsAlerts: { enabled: false, mediumRiskThreshold: 0.3 } });
  const cameraOnline = Boolean(status.cameraStatus?.online);
  const cameraConfigured = Boolean(status.cameraStatus?.configured ?? cameraOnline);
  const cameraState = status.cameraStatus?.state || (cameraOnline ? 'online' : 'idle');

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      fetch(`${AI_API_URL}/api/status`).then((r) => { if (!r.ok) throw Error('AI service unavailable'); return r.json(); }),
      fetch(`${AI_API_URL}/api/detections/recent`).then((r) => r.ok ? r.json() : []),
      fetch(`${AI_API_URL}/api/alerts`).then((r) => r.ok ? r.json() : []),
      fetch(`${AI_API_URL}/api/detections/evaluation`).then((r) => r.ok ? r.json() : null),
      fetch(`${AI_API_URL}/api/detections/reviews?limit=20`).then((r) => r.ok ? r.json() : []),
    ]);
    if (results[0].status === 'fulfilled') setStatus(results[0].value);
    if (results[1].status === 'fulfilled' && Array.isArray(results[1].value)) setEvents(results[1].value);
    if (results[2].status === 'fulfilled' && Array.isArray(results[2].value)) setAlerts(results[2].value);
    if (results[3].status === 'fulfilled' && results[3].value) setEvaluation(results[3].value);
    if (results[4].status === 'fulfilled' && Array.isArray(results[4].value)) setReviews(results[4].value);
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    let alive = true;
    api.get('/config').then(({ data }) => {
      const value = data?.data;
      if (alive && value?.emailAlerts && value?.smsAlerts) setAlertConfig({ emailAlerts: value.emailAlerts, smsAlerts: value.smsAlerts });
    }).catch(() => {});
    return () => { alive = false; };
  }, []);

  useEffect(() => () => { audioCleanup.current?.(); }, []);

  const toggleMicrophone = async () => {
    if (audioCleanup.current) { audioCleanup.current(); audioCleanup.current = null; setMicOn(false); setAudioEvent(''); return; }
    let stream; let context; let processor; let muted; let chunks = []; let pending = 0;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
      context = new AudioContext();
      const sourceNode = context.createMediaStreamSource(stream);
      processor = context.createScriptProcessor(4096, 1, 1);
      muted = context.createGain(); muted.gain.value = 0;
      processor.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0))); pending += chunks[chunks.length - 1].length;
        if (pending < context.sampleRate) return;
        const windowSamples = new Float32Array(pending); let offset = 0;
        chunks.forEach((chunk) => { windowSamples.set(chunk, offset); offset += chunk.length; });
        chunks = []; pending = 0;
        const bytes = new Uint8Array(windowSamples.buffer); let binary = '';
        for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + 0x8000, bytes.length)));
        fetch(`${AI_API_URL}/api/audio-analysis`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ audioData: btoa(binary), sampleRate: context.sampleRate }) })
          .then((response) => response.ok ? response.json() : Promise.reject(new Error('Audio model unavailable')))
          .then((result) => { const found = Object.entries(result.events || {}).find(([, active]) => active); setAudioEvent(found ? `${found[0]} detected` : ''); })
          .catch((error) => setNotice(error.message));
      };
      sourceNode.connect(processor); processor.connect(muted); muted.connect(context.destination);
      audioCleanup.current = () => { processor.disconnect(); muted.disconnect(); sourceNode.disconnect(); stream.getTracks().forEach((track) => track.stop()); context.close(); };
      setMicOn(true); setNotice('Microphone monitoring enabled. Allow microphone access to detect gunshots and explosions.');
    } catch (error) { stream?.getTracks().forEach((track) => track.stop()); await context?.close(); setNotice(`Microphone unavailable: ${error.message}`); }
  };

  const topRisk = useMemo(() => events.reduce((top, event) => {
    const score = { low: 1, medium: 2, high: 3, critical: 4 }[String(event.risk).toLowerCase()] || 0;
    return score > top.score ? { label: event.risk, score } : top;
  }, { label: 'LOW', score: 1 }).label, [events]);
  const filteredEvents = useMemo(() => events.filter((event) => `${event.label} ${event.risk} ${event.camera}`.toLowerCase().includes(query.toLowerCase())), [events, query]);

  const configureCamera = async (event) => {
    event.preventDefault(); setBusy(true); setNotice('Connecting to camera…');
    try {
      const config = { source: source === 'file' ? 'video' : source, rtspUrl: source === 'rtsp' ? sourceUrl : '', ipUrl: source === 'ip' ? sourceUrl : '', youtubeUrl: source === 'youtube' ? sourceUrl : '', videoFilePath: source === 'file' ? sourceUrl : '' };
      const response = await fetch(`${AI_API_URL}/api/camera/configure`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.message || result.error || 'Camera connection failed');
      const started = await fetch(`${AI_API_URL}/api/camera/start`, { method: 'POST' });
      const startResult = await started.json();
      if (!started.ok || !startResult.success) throw new Error(startResult.message || 'Camera could not start');
      setShowCamera(false); setNotice('Camera connected. Live analysis is starting.'); await refresh();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  };

  const stopCamera = async () => {
    setBusy(true);
    try { const response = await fetch(`${AI_API_URL}/api/camera/stop`, { method: 'POST' }); const result = await response.json(); if (!response.ok || !result.success) throw new Error(result.message || 'Could not stop camera'); setNotice('Camera feed stopped.'); await refresh(); }
    catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  };

  const saveConfiguration = async () => {
    setBusy(true);
    try { await api.post('/config', alertConfig); setNotice('Alert preferences saved.'); }
    catch (error) { setNotice(error.response?.data?.message || 'Could not save preferences.'); }
    finally { setBusy(false); }
  };

  const sendDetectionFeedback = async (event, actualEvent) => {
    try {
      const response = await fetch(`${AI_API_URL}/api/detections/feedback`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eventId: event.id, actualEvent }),
      });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.message || 'Could not save feedback');
      setNotice(`Review saved: predicted ${event.label}, actual ${actualEvent}. Evaluation metrics updated.`);
      await refresh();
    } catch (error) { setNotice(error.message); }
  };

  const acknowledgeAlert = async (alert) => {
    try {
      const response = await fetch(`${AI_API_URL}/api/alerts/${alert.id}/acknowledge`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.message || 'Could not acknowledge alert');
      setNotice(`${alert.type} alert acknowledged.`);
      await refresh();
    } catch (error) { setNotice(error.message); }
  };

  const signOut = () => { logout(); navigate('/login'); };
  const navItems = [['overview', 'Overview', '◫'], ['events', 'Events', '⌁'], ['settings', 'Settings', '⚙']];

  return <div className="command-shell">
    <aside className="command-rail">
      <div className="brand-mark"><span className="brand-core">◈</span><span className="brand-caption">SENTINEL<span>VISION OS</span></span></div>
      <div className="rail-divider" />
      <p className="rail-label">WORKSPACE</p>
      <nav className="rail-nav" aria-label="Main navigation">{navItems.map(([id, label, icon]) => <button key={id} className={`rail-link ${tab === id ? 'selected' : ''}`} onClick={() => setTab(id)}><span>{icon}</span>{label}</button>)}</nav>
      <div className="rail-bottom"><span className={`connection-dot ${cameraOnline ? 'online' : ''}`} /><span>{cameraOnline ? 'SYSTEM LIVE' : cameraState === 'reconnecting' ? 'RECONNECTING' : 'SYSTEM READY'}</span><button className="profile-button" title={`Sign out${user?.name ? ` (${user.name})` : ''}`} onClick={signOut}>{(user?.name || user?.email || 'U').slice(0, 1).toUpperCase()}</button></div>
    </aside>

    <main className="command-main">
      <header className="command-topbar"><div><div className="eyebrow"><span className="eyebrow-line" /> SECURITY OPERATIONS / <span>{tab.toUpperCase()}</span></div><h1>{tab === 'overview' ? 'Command center' : tab === 'events' ? 'Event intelligence' : 'System preferences'}</h1></div><div className="topbar-right"><div className="clock-label"><span className="pulse-dot" /> LIVE MONITORING <time>{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div><button className="button-quiet" onClick={toggleMicrophone}>{micOn ? '■ Stop mic' : '🎙 Monitor audio'}</button><button className="button-quiet" onClick={refresh} title="Refresh data">↻ <span>Refresh</span></button><button className="button-primary" onClick={() => setShowCamera(true)}>＋ <span>Connect source</span></button></div></header>
      {notice && <div className="notice-bar" role="status">{notice}<button onClick={() => setNotice('')} aria-label="Dismiss">×</button></div>}

      {tab === 'overview' && <>
        {audioEvent && <div className="notice-bar" role="alert">Audio alert: {audioEvent}</div>}
        <section className="metric-row" aria-label="System metrics">
          <article className="metric-card"><span className="metric-label">CAMERA STATUS</span><div className="metric-value"><span className={`metric-indicator ${cameraOnline ? 'online' : ''}`} />{cameraOnline ? 'Online' : cameraState === 'blank' ? 'No image' : cameraState === 'reconnecting' ? 'Reconnecting' : cameraState === 'connecting' ? 'Connecting' : cameraConfigured ? 'Waiting' : 'Offline'}</div><span className="metric-foot">{cameraOnline ? `Receiving video · ${status.cameraStatus?.fps || '—'} FPS` : status.cameraStatus?.error || (cameraConfigured ? 'Restoring the video connection' : 'No active source')}</span></article>
          <article className="metric-card"><span className="metric-label">CURRENT THREAT</span><div className={`metric-value ${riskClass(topRisk)}`}>{topRisk}</div><span className="metric-foot">Highest recent event severity</span></article>
          <article className="metric-card"><span className="metric-label">DETECTIONS</span><div className="metric-value">{events.length}<span className="metric-unit">events</span></div><span className="metric-foot">Recent event history</span></article>
          <article className="metric-card"><span className="metric-label">INFERENCE RATE</span><div className="metric-value">{status.cameraStatus?.fps || '—'}<span className="metric-unit">FPS</span></div><span className="metric-foot">{status.cameraStatus?.latency ? `${status.cameraStatus.latency} ms processing latency` : 'Awaiting camera telemetry'}</span></article>
        </section>
        <section className="overview-grid">
          <article className="panel feed-panel"><div className="panel-heading"><div><span className="section-index">01 / LIVE FEED</span><h2>Camera monitoring</h2></div><div className="feed-actions"><span className={`live-chip ${cameraOnline ? 'active' : ''}`}><i />{cameraOnline ? 'LIVE' : cameraState === 'blank' ? 'NO IMAGE' : cameraState === 'reconnecting' ? 'RECONNECTING' : cameraConfigured ? 'CONNECTING' : 'STANDBY'}</span>{cameraConfigured ? <button className="button-quiet danger" onClick={stopCamera} disabled={busy}>Stop feed</button> : <button className="button-quiet" onClick={() => setShowCamera(true)}>Configure</button>}</div></div>
            <div className={`feed-frame ${cameraConfigured ? 'receiving' : ''}`}>{cameraConfigured ? <><img src={`${AI_API_URL}/video_feed`} alt="AI processed live camera feed"/>{cameraState === 'blank' && <div className="camera-warning" role="status"><b>No camera image detected</b><span>Check the physical privacy shutter and Windows camera access, then reconnect.</span></div>}<div className="feed-overlay"><span>CAM-01 / PRIMARY · {cameraState.toUpperCase()}</span><span>{new Date().toLocaleString()}</span></div><div className="scanline" /></> : <div className="feed-empty"><div className="radar"><i/><i/><i/><span>◈</span></div><h3>Awaiting video source</h3><p>Connect a camera or recording to begin AI monitoring.</p><button className="button-primary" onClick={() => setShowCamera(true)}>＋ Connect a source</button></div>}</div>
            <div className="feed-footer"><span><i className={`small-dot ${cameraOnline ? 'online' : ''}`} /> {cameraOnline ? 'Stream connected' : cameraState === 'blank' ? 'Black image · check camera shutter or access' : cameraState === 'reconnecting' ? 'Reconnecting to stream' : cameraConfigured ? 'Connecting to stream' : 'No stream detected'}</span><span>VIDEO ANALYSIS <b>{status.hybridAI?.active ? 'ACTIVE' : 'IDLE'}</b></span><span>ENGINE <b>FIRE / SMOKE / ACCIDENT</b></span></div></article>
          <div className="side-stack"><article className="panel model-panel"><div className="panel-heading"><div><span className="section-index">02 / AI STACK</span><h2>Inference models</h2></div><span className="model-status"><i/> LOCAL</span></div><div className="model-entry"><div className="model-icon">◎</div><div><b>YOLO26 Nano</b><span>General object detection · pretrained</span></div><span className="model-state">{status.hybridAI?.active ? 'READY' : 'ON LOAD'}</span></div><div className="model-entry"><div className="model-icon violet">🔥</div><div><b>Fire &amp; smoke YOLO</b><span>Specialized visual detection</span></div><span className="model-state">{status.fireSmokeModel?.active ? 'READY' : 'OPTIONAL'}</span></div><div className="model-entry"><div className="model-icon violet">🚗</div><div><b>Traffic accident YOLO</b><span>Vehicle collision detection</span></div><span className="model-state">{status.accidentModel?.active ? 'READY' : 'OPTIONAL'}</span></div><div className="model-entry"><div className="model-icon">♫</div><div><b>YAMNet audio</b><span>Gunshot and explosion sounds {micOn ? '· listening' : '· microphone off'}</span></div><span className="model-state">{status.audioModel?.active ? 'READY' : 'OPTIONAL'}</span></div><div className="model-note">Explosion in video uses visual scene heuristics; enable audio monitoring to detect explosion sounds.</div></article>
            <article className="panel alert-panel"><div className="panel-heading"><div><span className="section-index">03 / PRIORITY QUEUE</span><h2>Recent alerts</h2></div><button className="text-button" onClick={() => setTab('events')}>All events ↗</button></div>{alerts.length ? <ul className="alert-list">{alerts.slice(0, 5).map((alert) => <li key={alert.id}><span className={`alert-mark ${riskClass(alert.risk)}`}>!</span><div className="alert-detail"><b>{alert.type || 'Event detected'} <span className={`risk-pill ${riskClass(alert.risk)}`}>{alert.risk}</span></b><span>{alert.message} · {alert.source} · {Math.round(Number(alert.confidence || 0) * 100)}% confidence</span><div className="alert-actions">{alert.evidenceUrl && <a href={`${AI_API_URL}${alert.evidenceUrl}`} target="_blank" rel="noreferrer">View evidence ↗</a>}{alert.acknowledged ? <small>ACKNOWLEDGED</small> : <button onClick={() => acknowledgeAlert(alert)}>Acknowledge</button>}</div></div><time>{formatTime(alert.timestamp)}</time></li>)}</ul> : <div className="quiet-empty"><span>◌</span><p>No alerts recorded</p><small>Confirmed incidents appear here with evidence.</small></div>}</article></div>
        </section>
        <section className="panel event-panel"><div className="panel-heading"><div><span className="section-index">04 / ACTIVITY</span><h2>Recent detections</h2></div><button className="text-button" onClick={() => setTab('events')}>Open event log ↗</button></div><EventTable events={events.slice(0, 5)} onFeedback={sendDetectionFeedback} /></section>
      </>}

      {tab === 'events' && <>
        <section className="panel event-panel"><div className="panel-heading"><div><span className="section-index">LABELED GROUND TRUTH</span><h2>Evaluation by event</h2></div><span className="model-status">{evaluation.labeledReviews} reviewed examples</span></div><p className="evaluation-note">Metrics use only events you have reviewed. Review both detections and missed events to measure precision and recall.</p><div className="evaluation-grid">{Object.entries(evaluation.metrics || {}).map(([label, metric]) => <article className="evaluation-card" key={label}><b>{label}</b><span>Precision <strong>{formatMetric(metric.precision)}</strong></span><span>Recall <strong>{formatMetric(metric.recall)}</strong></span><small>TP {metric.truePositives} · FP {metric.falsePositives} · FN {metric.falseNegatives}</small></article>)}</div></section>
        <section className="panel event-panel"><div className="panel-heading"><div><span className="section-index">EVENT INTELLIGENCE</span><h2>Current detections</h2></div><input className="search-field" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search type, risk, camera…" aria-label="Search events"/></div><EventTable events={filteredEvents} onFeedback={sendDetectionFeedback} /></section>
        <section className="panel event-panel"><div className="panel-heading"><div><span className="section-index">PERSISTED OPERATOR LABELS</span><h2>Reviewed examples</h2></div></div><ReviewTable reviews={reviews} /></section>
      </>}

      {tab === 'settings' && <section className="settings-grid"><div className="settings-delivery-note"><b>Outbound delivery is not connected yet.</b><span>These preferences are saved by the account API, but alerts currently appear in the local dashboard only. Email/SMS will send nothing until a delivery provider is integrated and validated.</span></div><article className="panel settings-card"><span className="section-index">NOTIFICATIONS / EMAIL</span><h2>Email alert threshold</h2><p>Choose the minimum confidence for high risk email alerts.</p><label className="toggle-row"><span>Email notifications</span><input type="checkbox" checked={Boolean(alertConfig.emailAlerts?.enabled)} onChange={(e) => setAlertConfig((v) => ({ ...v, emailAlerts: { ...v.emailAlerts, enabled: e.target.checked } }))}/></label><label className="threshold-row"><span>Confidence threshold <b>{Number(alertConfig.emailAlerts?.highRiskThreshold ?? 0.7).toFixed(1)}</b></span><input type="range" min="0" max="1" step="0.05" value={alertConfig.emailAlerts?.highRiskThreshold ?? 0.7} onChange={(e) => setAlertConfig((v) => ({ ...v, emailAlerts: { ...v.emailAlerts, highRiskThreshold: Number(e.target.value) } }))}/></label></article><article className="panel settings-card"><span className="section-index">NOTIFICATIONS / SMS</span><h2>SMS alert threshold</h2><p>Set SMS notification behavior for medium risk events.</p><label className="toggle-row"><span>SMS notifications</span><input type="checkbox" checked={Boolean(alertConfig.smsAlerts?.enabled)} onChange={(e) => setAlertConfig((v) => ({ ...v, smsAlerts: { ...v.smsAlerts, enabled: e.target.checked } }))}/></label><label className="threshold-row"><span>Confidence threshold <b>{Number(alertConfig.smsAlerts?.mediumRiskThreshold ?? 0.3).toFixed(1)}</b></span><input type="range" min="0" max="1" step="0.05" value={alertConfig.smsAlerts?.mediumRiskThreshold ?? 0.3} onChange={(e) => setAlertConfig((v) => ({ ...v, smsAlerts: { ...v.smsAlerts, mediumRiskThreshold: Number(e.target.value) } }))}/></label></article><button className="button-primary save-button" onClick={saveConfiguration} disabled={busy}>Save preferences</button></section>}
    </main>

    {showCamera && <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) setShowCamera(false); }}><form className="source-modal" onSubmit={configureCamera}><div className="modal-title"><div><span className="section-index">INPUT CONFIGURATION</span><h2>Connect a video source</h2></div><button type="button" className="close-modal" onClick={() => setShowCamera(false)} aria-label="Close">×</button></div><div className="source-grid">{sources.map(([id, title, subtitle]) => <button type="button" key={id} className={`source-choice ${source === id ? 'chosen' : ''}`} onClick={() => setSource(id)}><b>{title}</b><span>{subtitle}</span></button>)}</div>{source !== 'builtin' && <label className="source-input-label">{source === 'file' ? 'Video file path' : `${sources.find(([id]) => id === source)?.[1]} URL`}<input autoFocus required className="search-field source-input" value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)} placeholder={source === 'file' ? 'C:/videos/recording.mp4' : source === 'rtsp' ? 'rtsp://user:password@192.168.1.20:554/stream1' : source === 'ip' ? 'http://192.168.1.20:8080/video' : source === 'youtube' ? 'https://www.youtube.com/watch?v=… or https://youtu.be/…' : 'https://…'}/><small className="source-help">{source === 'youtube' ? 'Use a public video or live link. YouTube resolves through yt-dlp and needs internet access.' : source === 'rtsp' ? 'Use the camera’s RTSP stream URL. The camera must be reachable from this computer.' : source === 'ip' ? 'Use the camera’s direct HTTP(S) video stream URL, not its web dashboard page.' : source === 'file' ? 'Choose a video path that exists on the computer running the Python service.' : ''}</small></label>}{source === 'builtin' && <p className="source-help">Uses the default camera selected by VIDEO_SOURCE (usually device 0). Allow Windows camera access and close any other app using it.</p>}<div className="modal-footer"><span>Video is processed by your local AI service.</span><button type="button" className="button-quiet" onClick={() => setShowCamera(false)}>Cancel</button><button className="button-primary" disabled={busy}>{busy ? 'Connecting…' : 'Start monitoring'}</button></div></form></div>}
  </div>;
}

function EventTable({ events, onFeedback }) {
  const [missLabels, setMissLabels] = useState({});
  if (!events.length) return <div className="table-empty"><span>⌁</span><b>No detection events yet</b><p>Detected events will appear here when video analysis is running.</p></div>;
  return <div className="table-scroll"><table className="event-table"><thead><tr><th>EVENT</th><th>SEVERITY</th><th>CONFIDENCE</th><th>CAMERA</th><th>TIME</th><th>LABEL THIS SAMPLE</th></tr></thead><tbody>{events.map((event, index) => { const label = String(event.label || event.type || 'normal').toLowerCase(); return <tr key={event.id || `${event.timestamp}-${index}`}><td><span className="event-glyph">⌁</span>{label}</td><td><span className={`risk-pill ${riskClass(event.risk)}`}>{event.risk || 'LOW'}</span></td><td>{Math.round(Number(event.confidence || 0) * (Number(event.confidence || 0) <= 1 ? 100 : 1))}%</td><td>{event.camera || '—'}</td><td>{formatTime(event.timestamp)}</td><td>{onFeedback && !['normal', 'unknown'].includes(label) ? <div className="feedback-actions"><button onClick={() => onFeedback(event, label)} title="This detection is correct">Correct</button><button onClick={() => onFeedback(event, 'normal')} title="This was a false alarm">False alarm</button></div> : onFeedback ? <div className="feedback-actions"><select value={missLabels[event.id] || 'fire'} onChange={(e) => setMissLabels((v) => ({ ...v, [event.id]: e.target.value }))} aria-label="Select missed event type"><option value="fire">Fire</option><option value="smoke">Smoke</option><option value="explosion">Explosion</option><option value="accident">Accident</option><option value="gunshot">Gunshot</option></select><button onClick={() => onFeedback(event, missLabels[event.id] || 'fire')} title="Report a missed event">Missed</button></div> : null}</td></tr>; })}</tbody></table></div>;
}

const formatMetric = (value) => value == null ? '—' : `${Math.round(value * 100)}%`;

function ReviewTable({ reviews }) {
  if (!reviews.length) return <div className="table-empty"><span>⌁</span><b>No reviewed examples yet</b><p>Use the label controls in event intelligence to start a persistent evaluation set.</p></div>;
  return <div className="table-scroll"><table className="event-table"><thead><tr><th>PREDICTED</th><th>ACTUAL LABEL</th><th>CONFIDENCE</th><th>CAMERA</th><th>REVIEWED</th></tr></thead><tbody>{reviews.map((review) => <tr key={review.eventId}><td>{review.predicted}</td><td>{review.actual}</td><td>{Math.round(review.confidence * 100)}%</td><td>{review.camera}</td><td>{formatTime(review.reviewedAt)}</td></tr>)}</tbody></table></div>;
}
