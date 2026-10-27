import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { useAuth } from '../context/AuthContext';
import api from '../utils/api';
import './AuthPage.css';

const getApiMessage = (error, fallback) => error.response?.data?.message || (error.response ? fallback : 'Account service is unreachable. Check that the API is running and try again.');

const AuthPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [activeTab, setActiveTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [service, setService] = useState({ state: 'checking', message: 'Checking account service…' });
  const [showPassword, setShowPassword] = useState(false);
  const [loginData, setLoginData] = useState({ identifier: '', password: '' });
  const [registerData, setRegisterData] = useState({ username: '', email: '', mobile: '', password: '', confirmPassword: '' });
  const [registerStep, setRegisterStep] = useState(1);
  const [userId, setUserId] = useState('');
  const [otp, setOtp] = useState('');
  const [devOtp, setDevOtp] = useState('');
  const otpRef = useRef(null);

  useEffect(() => {
    let active = true;
    api.get('/health', { timeout: 5000 })
      .then(({ data }) => {
        if (!active) return;
        const ready = data.status === 'ok';
        setService({ state: ready ? 'ready' : 'unavailable', message: ready ? 'Account service connected' : (data.message || 'Account service is not ready') });
      })
      .catch((err) => {
        if (!active) return;
        setService({ state: 'unavailable', message: getApiMessage(err, 'Account service is unavailable') });
      });
    return () => { active = false; };
  }, []);

  const switchTab = (tab) => {
    setActiveTab(tab);
    setError('');
    setNotice('');
    setShowPassword(false);
    if (tab === 'register') setRegisterStep(1);
  };

  const finishLogin = (data) => {
    login(data.token, data.user);
    navigate('/dashboard');
  };

  const handleLogin = async (event) => {
    event.preventDefault();
    setError('');
    setNotice('');
    setLoading(true);
    try {
      const { data } = await api.post('/auth/login', loginData);
      finishLogin(data);
    } catch (err) {
      if (err.response?.status === 403 && err.response.data.userId) {
        setUserId(err.response.data.userId);
        setOtp('');
        setRegisterStep(2);
        setActiveTab('register');
        setNotice('Your account needs email verification before you can sign in.');
      } else {
        setError(getApiMessage(err, 'We couldn’t sign you in. Check your details and try again.'));
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (event) => {
    event.preventDefault();
    setError('');
    setNotice('');
    if (registerData.password.length < 6) {
      setError('Use a password with at least 6 characters.');
      return;
    }
    if (registerData.password !== registerData.confirmPassword) {
      setError('Those passwords don’t match yet.');
      return;
    }
    setLoading(true);
    try {
      const { data } = await api.post('/auth/register', {
        ...registerData,
        username: registerData.username.trim(),
        email: registerData.email.trim().toLowerCase(),
        mobile: registerData.mobile.replace(/\D/g, ''),
      });
      setUserId(data.data.userId);
      setDevOtp(data.data.otp || '');
      setOtp('');
      setRegisterStep(2);
      setNotice(data.message || 'Enter the verification code to finish creating your account.');
      window.setTimeout(() => otpRef.current?.focus(), 0);
    } catch (err) {
      setError(getApiMessage(err, 'We couldn’t create your account. Review the details and try again.'));
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOtp = async (event) => {
    event.preventDefault();
    setError('');
    if (!/^\d{6}$/.test(otp)) {
      setError('Enter all 6 digits from your verification code.');
      return;
    }
    setLoading(true);
    try {
      const { data } = await api.post('/auth/verify-otp', { userId, otp });
      finishLogin(data);
    } catch (err) {
      setError(getApiMessage(err, 'That code could not be verified. Check it and try again.'));
    } finally {
      setLoading(false);
    }
  };

  const handleResendOtp = async () => {
    setError('');
    setNotice('');
    setLoading(true);
    try {
      const { data } = await api.post('/auth/resend-otp', { userId });
      setDevOtp(data.otp || '');
      setOtp('');
      setNotice(data.message || 'A new verification code has been generated.');
      otpRef.current?.focus();
    } catch (err) {
      setError(getApiMessage(err, 'We couldn’t resend the verification code.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <Link to="/" className="back-link"><span aria-hidden="true">←</span> Back to home</Link>
      <div className="auth-layout">
        <section className="auth-story">
          <Link className="auth-brand" to="/" aria-label="Aegis Vision home"><span className="auth-brand-icon">A</span><span>AEGIS <small>VISION</small></span></Link>
          <div className="auth-story-copy">
            <span className="auth-kicker"><i/> PRIVATE VIDEO INTELLIGENCE</span>
            <h1>See the<br/>moments that<br/><em>matter.</em></h1>
            <p>One secure workspace for your camera feeds, detections, and review history.</p>
          </div>
          <div className="auth-story-foot"><span>01</span><div><b>Your workspace stays yours.</b><small>Sign in to continue or create an account to get started.</small></div><span className="story-shield">◇</span></div>
        </section>

        <section className="auth-container" aria-labelledby="auth-title">
          <div className="auth-header">
            <span className="auth-eyebrow">WORKSPACE ACCESS</span>
            <h2 id="auth-title">{registerStep === 2 && activeTab === 'register' ? 'Verify your email' : activeTab === 'login' ? 'Welcome back' : 'Create your account'}</h2>
            <p>{registerStep === 2 && activeTab === 'register' ? 'One last step before you enter your workspace.' : activeTab === 'login' ? 'Sign in to your Aegis Vision workspace.' : 'Set up your secure video workspace in a few steps.'}</p>
          </div>

          <div className={`service-status ${service.state}`} role="status"><i/>{service.message}</div>
          <div className="auth-tabs" role="tablist" aria-label="Account access">
            <button type="button" role="tab" aria-selected={activeTab === 'login'} className={`tab ${activeTab === 'login' ? 'active' : ''}`} onClick={() => switchTab('login')}>Sign in</button>
            <button type="button" role="tab" aria-selected={activeTab === 'register'} className={`tab ${activeTab === 'register' ? 'active' : ''}`} onClick={() => switchTab('register')}>Create account</button>
          </div>

          {error && <div className="auth-message error-message" role="alert">{error}</div>}
          {notice && <div className="auth-message notice-message" role="status">{notice}</div>}

          {activeTab === 'login' && <form onSubmit={handleLogin} className="auth-form">
            <label className="auth-field"><span>Username or email</span><input autoComplete="username" type="text" className="input" placeholder="you@example.com" value={loginData.identifier} onChange={(event) => setLoginData({ ...loginData, identifier: event.target.value })} required/></label>
            <label className="auth-field"><span>Password</span><div className="password-wrap"><input autoComplete="current-password" type={showPassword ? 'text' : 'password'} className="input" placeholder="Enter your password" value={loginData.password} onChange={(event) => setLoginData({ ...loginData, password: event.target.value })} required/><button className="password-toggle" type="button" onClick={() => setShowPassword((value) => !value)}>{showPassword ? 'Hide' : 'Show'}</button></div></label>
            <button type="submit" className="auth-submit" disabled={loading}>{loading ? 'Signing in…' : 'Sign in'}<span aria-hidden="true">→</span></button>
          </form>}

          {activeTab === 'register' && registerStep === 1 && <form onSubmit={handleRegister} className="auth-form register-form">
            <label className="auth-field"><span>Username</span><input autoComplete="username" minLength="3" maxLength="32" type="text" className="input" placeholder="Choose a username" value={registerData.username} onChange={(event) => setRegisterData({ ...registerData, username: event.target.value })} required/></label>
            <div className="auth-field-row">
              <label className="auth-field"><span>Email address</span><input autoComplete="email" type="email" className="input" placeholder="you@example.com" value={registerData.email} onChange={(event) => setRegisterData({ ...registerData, email: event.target.value })} required/></label>
              <label className="auth-field"><span>Mobile number</span><input autoComplete="tel-national" type="tel" inputMode="numeric" maxLength="10" pattern="[0-9]{10}" className="input" placeholder="10 digits" value={registerData.mobile} onChange={(event) => setRegisterData({ ...registerData, mobile: event.target.value.replace(/\D/g, '').slice(0, 10) })} required/></label>
            </div>
            <label className="auth-field"><span>Password <small>At least 6 characters</small></span><input autoComplete="new-password" minLength="6" type="password" className="input" placeholder="Create a password" value={registerData.password} onChange={(event) => setRegisterData({ ...registerData, password: event.target.value })} required/></label>
            <label className="auth-field"><span>Confirm password</span><input autoComplete="new-password" minLength="6" type="password" className="input" placeholder="Type it again" value={registerData.confirmPassword} onChange={(event) => setRegisterData({ ...registerData, confirmPassword: event.target.value })} required/></label>
            <button type="submit" className="auth-submit" disabled={loading}>{loading ? 'Creating account…' : 'Create account'}<span aria-hidden="true">→</span></button>
            <p className="auth-terms">By continuing, you agree to keep your account credentials private.</p>
          </form>}

          {activeTab === 'register' && registerStep === 2 && <form onSubmit={handleVerifyOtp} className="auth-form otp-section">
            <div className="otp-destination">CODE SENT TO <b>{registerData.email || loginData.identifier}</b></div>
            {devOtp && <div className="dev-otp">Development code <b>{devOtp}</b></div>}
            <label className="auth-field"><span>6-digit verification code</span><input ref={otpRef} autoComplete="one-time-code" inputMode="numeric" pattern="[0-9]{6}" maxLength="6" className="input otp-field" placeholder="000000" value={otp} onChange={(event) => setOtp(event.target.value.replace(/\D/g, '').slice(0, 6))} required/></label>
            <button type="submit" className="auth-submit" disabled={loading}>{loading ? 'Verifying…' : 'Verify and continue'}<span aria-hidden="true">→</span></button>
            <div className="otp-actions"><button type="button" className="link-button" onClick={handleResendOtp} disabled={loading}>Resend code</button><button type="button" className="link-button" onClick={() => { setRegisterStep(1); setError(''); setNotice(''); }}>Edit details</button></div>
          </form>}

          <div className="auth-footer"><span className="footer-lock" aria-hidden="true">◇</span><span>Secure access to your video workspace</span></div>
        </section>
      </div>
      <div className="auth-bg-elements" aria-hidden="true"><div className="auth-bg-circle auth-bg-circle-1"/><div className="auth-bg-circle auth-bg-circle-2"/></div>
    </main>
  );
};

export default AuthPage;
