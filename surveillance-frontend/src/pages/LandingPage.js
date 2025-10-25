import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      {/* Background Gradient */}
      <div className="bg-gradient-radial"></div>

      {/* Main Content */}
      <div className="main-content-wrapper">
        {/* Hero Section */}
        <div className="hero-container">
          {/* Shield Icon */}
          <div className="shield-icon float-animation">
            <svg className="shield-svg" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
            </svg>
          </div>

          {/* Main Heading */}
          <h1 className="main-heading">
            Next-Gen AI Surveillance System
          </h1>

          {/* Subheading */}
          <p className="sub-heading">
            Advanced multi-modal AI analysis for comprehensive security monitoring
          </p>

          {/* Action Buttons */}
          <div className="button-group">
            <button 
              className="btn-get-started"
              onClick={() => navigate('/login')}
            >
              GET STARTED
            </button>
            <button 
              className="btn-sign-up"
              onClick={() => navigate('/login')}
            >
              SIGN UP
            </button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="features-container">
          <div className="features-grid">
            {/* Card 1 */}
            <div className="feature-card">
              <div className="card-icon">📹</div>
              <h3 className="card-title">Live Camera Feeds</h3>
              <p className="card-description">
                Monitor multiple camera feeds in real time with AI powered analysis
              </p>
            </div>

            {/* Card 2 */}
            <div className="feature-card">
              <div className="card-icon">🤖</div>
              <h3 className="card-title">AI Detection</h3>
              <p className="card-description">
                Advanced AI detects people, vehicles, and suspicious activities automatically
              </p>
            </div>

            {/* Card 3 */}
            <div className="feature-card">
              <div className="card-icon">🔔</div>
              <h3 className="card-title">Smart Alerts</h3>
              <p className="card-description">
                Instant notifications for security events with intelligent threat assessment
              </p>
            </div>

            {/* Card 4 */}
            <div className="feature-card">
              <div className="card-icon">🛡️</div>
              <h3 className="card-title">Secure & Reliable</h3>
              <p className="card-description">
                24/7 monitoring with encrypted data and robust backup systems
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer CTA */}
      <div className="footer-cta-section">
        <h2 className="footer-heading">Ready to Enhance Your Security?</h2>
        <p className="footer-subheading">
          Join thousands of users protecting their premises with AI-powered surveillance
        </p>
        <button 
          className="btn-cta-large"
          onClick={() => navigate('/login')}
        >
          START MONITORING NOW
        </button>
      </div>

      {/* Floating Background Circles */}
      <div className="floating-circles">
        <div className="floating-circle circle-1"></div>
        <div className="floating-circle circle-2"></div>
      </div>
    </div>
  );
};

export default LandingPage;
