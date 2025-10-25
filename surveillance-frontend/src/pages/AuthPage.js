import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../utils/api';
import './AuthPage.css';

const AuthPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [activeTab, setActiveTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Login State
  const [loginData, setLoginData] = useState({
    username: '',
    password: ''
  });
  
  // Register State
  const [registerStep, setRegisterStep] = useState(1);
  const [registerData, setRegisterData] = useState({
    username: '',
    email: '',
    mobile: '',
    password: '',
    confirmPassword: ''
  });
  const [userId, setUserId] = useState('');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);

  // Handle Login
  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await api.post('/auth/login', loginData);
      login(response.data.token, response.data.user);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  // Handle Register - Step 1
  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');

    if (registerData.password !== registerData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const response = await api.post('/auth/register', registerData);
      setUserId(response.data.data.userId);
      setRegisterStep(2);
      setError('');
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  // Handle OTP Input
  const handleOtpChange = (index, value) => {
    if (value.length > 1) value = value[0];
    if (!/^\d*$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-focus next input
    if (value && index < 5) {
      document.getElementById(`otp-${index + 1}`).focus();
    }
  };

  // Handle OTP Verification
  const handleVerifyOtp = async (e) => {
    e.preventDefault();
    setError('');

    const otpString = otp.join('');
    if (otpString.length !== 6) {
      setError('Please enter complete OTP');
      return;
    }

    setLoading(true);

    try {
      const response = await api.post('/auth/verify-otp', {
        userId,
        otp: otpString
      });
      login(response.data.token, response.data.user);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.message || 'OTP verification failed');
    } finally {
      setLoading(false);
    }
  };

  // Resend OTP
  const handleResendOtp = async () => {
    setLoading(true);
    try {
      await api.post('/auth/resend-otp', { userId });
      setError('');
      alert('OTP resent successfully!');
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to resend OTP');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <Link to="/" className="back-link">
        ← BACK TO HOME
      </Link>

      <div className="auth-container glass fade-in">
        <div className="auth-header">
          <h1 className="auth-title">SECURE ACCESS</h1>
          <p className="auth-subtitle">AI-Powered Surveillance</p>
        </div>

        {/* Tabs */}
        <div className="auth-tabs">
          <button
            className={`tab ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('login');
              setError('');
            }}
          >
            LOGIN
          </button>
          <button
            className={`tab ${activeTab === 'register' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('register');
              setError('');
              setRegisterStep(1);
            }}
          >
            REGISTER
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Login Form */}
        {activeTab === 'login' && (
          <form onSubmit={handleLogin} className="auth-form">
            <div className="form-group">
              <input
                type="text"
                className="input"
                placeholder="Username"
                value={loginData.username}
                onChange={(e) => setLoginData({ ...loginData, username: e.target.value })}
                required
              />
            </div>
            <div className="form-group">
              <input
                type="password"
                className="input"
                placeholder="Password"
                value={loginData.password}
                onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                required
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'LOGGING IN...' : 'LOGIN'}
            </button>
          </form>
        )}

        {/* Register Form */}
        {activeTab === 'register' && (
          <>
            {/* Progress Indicator */}
            <div className="progress-indicator">
              <div className={`step ${registerStep >= 1 ? 'active' : ''}`}>
                <div className="step-number">1</div>
                <div className="step-label">Enter Details</div>
              </div>
              <div className={`step-line ${registerStep >= 2 ? 'active' : ''}`}></div>
              <div className={`step ${registerStep >= 2 ? 'active' : ''}`}>
                <div className="step-number">2</div>
                <div className="step-label">Verify OTP</div>
              </div>
              <div className={`step-line ${registerStep >= 3 ? 'active' : ''}`}></div>
              <div className={`step ${registerStep >= 3 ? 'active' : ''}`}>
                <div className="step-number">3</div>
                <div className="step-label">Complete</div>
              </div>
            </div>

            {/* Step 1: Enter Details */}
            {registerStep === 1 && (
              <form onSubmit={handleRegister} className="auth-form">
                <div className="form-group">
                  <input
                    type="text"
                    className="input"
                    placeholder="Username"
                    value={registerData.username}
                    onChange={(e) => setRegisterData({ ...registerData, username: e.target.value })}
                    required
                  />
                </div>
                <div className="form-group">
                  <input
                    type="email"
                    className="input"
                    placeholder="Email"
                    value={registerData.email}
                    onChange={(e) => setRegisterData({ ...registerData, email: e.target.value })}
                    required
                  />
                </div>
                <div className="form-group">
                  <input
                    type="tel"
                    className="input"
                    placeholder="Mobile (10 digits)"
                    value={registerData.mobile}
                    onChange={(e) => setRegisterData({ ...registerData, mobile: e.target.value })}
                    pattern="[0-9]{10}"
                    required
                  />
                </div>
                <div className="form-group">
                  <input
                    type="password"
                    className="input"
                    placeholder="Password"
                    value={registerData.password}
                    onChange={(e) => setRegisterData({ ...registerData, password: e.target.value })}
                    required
                  />
                </div>
                <div className="form-group">
                  <input
                    type="password"
                    className="input"
                    placeholder="Confirm Password"
                    value={registerData.confirmPassword}
                    onChange={(e) => setRegisterData({ ...registerData, confirmPassword: e.target.value })}
                    required
                  />
                </div>
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? 'CREATING...' : 'CREATE ACCOUNT'}
                </button>
              </form>
            )}

            {/* Step 2: Verify OTP */}
            {registerStep === 2 && (
              <div className="otp-section">
                <h3 className="otp-title">Verify Your Identity</h3>
                <p className="otp-subtitle">
                  6 digit code sent to {registerData.email} and +{registerData.mobile}
                </p>
                <form onSubmit={handleVerifyOtp} className="otp-form">
                  <div className="otp-inputs">
                    {otp.map((digit, index) => (
                      <input
                        key={index}
                        id={`otp-${index}`}
                        type="text"
                        maxLength="1"
                        className="otp-input"
                        value={digit}
                        onChange={(e) => handleOtpChange(index, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Backspace' && !digit && index > 0) {
                            document.getElementById(`otp-${index - 1}`).focus();
                          }
                        }}
                      />
                    ))}
                  </div>
                  <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? 'VERIFYING...' : 'VERIFY'}
                  </button>
                  <div className="otp-actions">
                    <button type="button" className="link-button" onClick={handleResendOtp} disabled={loading}>
                      Resend
                    </button>
                    <button type="button" className="link-button" onClick={() => setRegisterStep(1)}>
                      Back
                    </button>
                  </div>
                </form>
              </div>
            )}
          </>
        )}

        <div className="auth-footer">
          <p>Your data is secured</p>
        </div>
      </div>

      {/* Background Elements */}
      <div className="auth-bg-elements">
        <div className="auth-bg-circle auth-bg-circle-1"></div>
        <div className="auth-bg-circle auth-bg-circle-2"></div>
      </div>
    </div>
  );
};

export default AuthPage;
