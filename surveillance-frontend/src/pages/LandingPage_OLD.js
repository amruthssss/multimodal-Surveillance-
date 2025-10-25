import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: '📹',
      title: 'Live Camera Feeds',
      description: 'Monitor multiple camera feeds in real-time with AI-powered analysis'
    },
    {
      icon: '�',
      title: 'AI Detection',
      description: 'Advanced AI detects people, vehicles, and suspicious activities automatically'
    },
    {
      icon: '🔔',
      title: 'Smart Alerts',
      description: 'Instant notifications for security events with intelligent threat assessment'
    },
    {
      icon: '🛡️',
      title: 'Secure & Reliable',
      description: '24/7 monitoring with encrypted data and robust backup systems'
    }
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="logo-icon">🛡️</div>
          <h1 className="hero-title">
            Next-Gen AI Surveillance System
          </h1>
          <p className="hero-subtitle">
            Advanced multi-modal AI analysis for comprehensive security monitoring
          </p>
          <div className="hero-buttons">
            <button 
              className="btn btn-primary glow"
              onClick={() => navigate('/login')}
            >
              GET STARTED
            </button>
            <button 
              className="btn btn-secondary"
              onClick={() => navigate('/login')}
            >
              SIGN UP
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-grid">
          {features.map((feature, index) => (
            <div 
              key={index} 
              className="feature-card glass"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <h2 className="cta-title">Ready to Enhance Your Security?</h2>
        <p className="cta-subtitle">
          Join thousands of users protecting their premises with AI-powered surveillance
        </p>
        <button 
          className="btn btn-primary btn-large glow"
          onClick={() => navigate('/login')}
        >
          START MONITORING NOW
        </button>
      </section>

      {/* Animated Background */}
      <div className="animated-bg">
        <div className="bg-circle bg-circle-1"></div>
        <div className="bg-circle bg-circle-2"></div>
        <div className="bg-circle bg-circle-3"></div>
        <div className="grid-pattern"></div>
      </div>
    </div>
  );
};

export default LandingPage;
