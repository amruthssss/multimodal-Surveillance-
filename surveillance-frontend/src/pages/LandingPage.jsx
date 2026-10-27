import React from 'react';
import { useNavigate } from 'react-router';
import './LandingPage.css';

const features = [
  { number: '01', title: 'Connect a source', text: 'Use a camera, network stream, or recorded video.' },
  { number: '02', title: 'Review detections', text: 'See recognized objects and changes in the scene.' },
  { number: '03', title: 'Tune alerts', text: 'Set thresholds and review the event history.' },
];

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <header className="site-header">
        <a className="site-brand" href="/" aria-label="Aegis home">
          <span className="site-brand-icon" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M16 3.5 27 8v7.4c0 6.7-4.6 11.1-11 13.1C9.6 26.5 5 22.1 5 15.4V8l11-4.5Z" stroke="currentColor" strokeWidth="1.7"/><path d="m11.7 16.1 2.8 2.8 6-6.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg></span>
          <span className="site-brand-name">AEGIS<span>VISION</span></span>
        </a>
        <nav className="site-nav" aria-label="Main navigation">
          <a href="#platform">Platform</a>
          <a href="#workflow">How it works</a>
        </nav>
        <div className="site-actions">
          <button className="site-login" onClick={() => navigate('/login')}>Sign in</button>
          <button className="site-cta" onClick={() => navigate('/login')}>Open console <span aria-hidden="true">↗</span></button>
        </div>
      </header>

      <main>
        <section className="hero-section" id="platform">
          <div className="hero-copy">
            <div className="hero-kicker"><span className="kicker-dot"/> VIDEO INTELLIGENCE, MADE CLEAR</div>
            <h1>Know what’s<br/>happening <span>now.</span></h1>
            <p className="hero-description">A focused workspace for camera feeds, visual detections, and the events that need your attention.</p>
            <div className="hero-actions">
              <button className="hero-primary" onClick={() => navigate('/login')}>Launch workspace <span aria-hidden="true">→</span></button>
              <a className="hero-secondary" href="#workflow">Explore the platform <span aria-hidden="true">↓</span></a>
            </div>
            <div className="hero-footnote"><span className="mini-shield">✓</span> Local video processing · Bring your own source</div>
          </div>

          <div className="preview-wrap" aria-label="Illustrative dashboard preview">
            <div className="preview-glow"/>
            <div className="preview-card">
              <div className="preview-topbar"><div className="preview-brand"><span className="preview-mark">A</span><span>OPERATIONS <small>VIDEO INTELLIGENCE</small></span></div><span className="preview-live"><i/> SYSTEM READY</span></div>
              <div className="preview-content">
                <div className="preview-heading"><div><span>WORKSPACE PREVIEW</span><h2>Live overview</h2></div><span className="preview-menu">•••</span></div>
                <div className="preview-feed"><div className="feed-grid"/><div className="feed-corner corner-tl"/><div className="feed-corner corner-tr"/><div className="feed-corner corner-bl"/><div className="feed-corner corner-br"/><div className="feed-center"><div className="feed-rings"><i/><i/><b>◉</b></div><span>CONNECT A VIDEO SOURCE</span></div><div className="feed-label"><i/> SOURCE OFFLINE</div><div className="feed-time">CAM 01&nbsp; / &nbsp;PREVIEW</div></div>
                <div className="preview-stats"><div><span>CAMERAS</span><b>—</b><small>Awaiting source</small></div><div><span>DETECTIONS</span><b>—</b><small>Nothing to review</small></div><div><span>ALERTS</span><b>—</b><small>All quiet</small></div></div>
                <div className="preview-footer"><span><i/> READY WHEN YOU ARE</span><span>LOCAL AI</span></div>
              </div>
            </div>
            <div className="preview-orbit orbit-one"/><div className="preview-orbit orbit-two"/>
          </div>
        </section>

        <section className="workflow-section" id="workflow">
          <div className="workflow-heading"><div><span className="section-kicker">A CLEARER WAY TO MONITOR</span><h2>Everything in one view.</h2></div><p>Connect your feed, understand what the system sees, and review the moments that matter.</p></div>
          <div className="workflow-grid">{features.map((feature) => <article className="workflow-card" key={feature.number}><span className="workflow-number">{feature.number}</span><div><h3>{feature.title}</h3><p>{feature.text}</p></div><span className="workflow-arrow" aria-hidden="true">↗</span></article>)}</div>
        </section>
      </main>
      <footer className="site-footer"><span>AEGIS VISION</span><span>Video intelligence for your workspace.</span><button onClick={() => navigate('/login')}>Sign in <span aria-hidden="true">→</span></button></footer>
    </div>
  );
};

export default LandingPage;
