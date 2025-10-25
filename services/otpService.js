// 🔐 OTP Service - Unified OTP handling for Email & SMS
const emailService = require('./emailService');
const smsService = require('./smsService');
const crypto = require('crypto');
const speakeasy = require('speakeasy');
const QRCode = require('qrcode');

class OTPService {
  constructor() {
    this.otpStorage = new Map(); // In production, use Redis or database
    this.resendCooldowns = new Map();
    
    // Configuration from environment
    this.config = {
      otpLength: parseInt(process.env.OTP_LENGTH) || 6,
      expiryMinutes: parseInt(process.env.OTP_EXPIRY_MINUTES) || 10,
      maxAttempts: parseInt(process.env.MAX_OTP_ATTEMPTS) || 3,
      resendCooldownSeconds: parseInt(process.env.OTP_RESEND_COOLDOWN) || 60
    };
    
    console.log('🔐 OTP Service initialized with config:', this.config);
  }

  generateOTP() {
    const min = Math.pow(10, this.config.otpLength - 1);
    const max = Math.pow(10, this.config.otpLength) - 1;
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  generateSecureToken(length = 32) {
    return crypto.randomBytes(length).toString('hex');
  }

  async sendOTP(identifier, type = 'email', username = 'User', purpose = 'login') {
    // Check resend cooldown
    const cooldownKey = `${identifier}_${type}`;
    const lastSent = this.resendCooldowns.get(cooldownKey);
    
    if (lastSent && Date.now() - lastSent < this.config.resendCooldownSeconds * 1000) {
      const remainingSeconds = Math.ceil((this.config.resendCooldownSeconds * 1000 - (Date.now() - lastSent)) / 1000);
      return {
        success: false,
        error: 'resend_cooldown',
        message: `Please wait ${remainingSeconds} seconds before requesting another code`,
        remainingSeconds
      };
    }

    // Generate OTP
    const otp = this.generateOTP();
    const expiresAt = Date.now() + (this.config.expiryMinutes * 60 * 1000);
    const otpId = this.generateSecureToken(16);

    // Store OTP
    const otpData = {
      id: otpId,
      code: otp,
      identifier,
      type,
      purpose,
      expiresAt,
      attempts: 0,
      verified: false,
      createdAt: Date.now()
    };

    this.otpStorage.set(otpId, otpData);
    
    // Set resend cooldown
    this.resendCooldowns.set(cooldownKey, Date.now());

    // Send OTP via appropriate service
    let sendResult;
    try {
      if (type === 'email') {
        sendResult = await emailService.sendOTPEmail(identifier, otp, username);
      } else if (type === 'sms') {
        sendResult = await smsService.sendOTPSMS(identifier, otp, username);
      } else {
        throw new Error(`Unsupported OTP type: ${type}`);
      }

      console.log(`🔐 OTP sent via ${type} to ${identifier} (ID: ${otpId})`);
      
      return {
        success: true,
        otpId,
        expiresAt,
        type,
        service: sendResult.service,
        messageId: sendResult.messageId || sendResult.messageSid
      };
    } catch (error) {
      // Remove OTP from storage if sending failed
      this.otpStorage.delete(otpId);
      this.resendCooldowns.delete(cooldownKey);
      
      console.error(`❌ Failed to send OTP via ${type}:`, error.message);
      return {
        success: false,
        error: 'send_failed',
        message: `Failed to send verification code via ${type}`,
        details: error.message
      };
    }
  }

  async verifyOTP(otpId, code, identifier = null) {
    const otpData = this.otpStorage.get(otpId);
    
    if (!otpData) {
      return {
        success: false,
        error: 'invalid_otp_id',
        message: 'Invalid or expired verification code'
      };
    }

    // Check if already verified
    if (otpData.verified) {
      return {
        success: false,
        error: 'already_verified',
        message: 'This verification code has already been used'
      };
    }

    // Check expiration
    if (Date.now() > otpData.expiresAt) {
      this.otpStorage.delete(otpId);
      return {
        success: false,
        error: 'expired',
        message: 'Verification code has expired'
      };
    }

    // Increment attempts
    otpData.attempts++;

    // Check max attempts
    if (otpData.attempts > this.config.maxAttempts) {
      this.otpStorage.delete(otpId);
      return {
        success: false,
        error: 'max_attempts',
        message: 'Too many failed attempts. Please request a new code'
      };
    }

    // Verify identifier if provided
    if (identifier && otpData.identifier !== identifier) {
      return {
        success: false,
        error: 'identifier_mismatch',
        message: 'Verification code not valid for this account'
      };
    }

    // Verify code
    if (otpData.code.toString() !== code.toString()) {
      this.otpStorage.set(otpId, otpData); // Update attempts
      return {
        success: false,
        error: 'invalid_code',
        message: `Invalid verification code. ${this.config.maxAttempts - otpData.attempts} attempts remaining`
      };
    }

    // Success!
    otpData.verified = true;
    otpData.verifiedAt = Date.now();
    this.otpStorage.set(otpId, otpData);

    console.log(`✅ OTP verified successfully (ID: ${otpId})`);

    return {
      success: true,
      otpData: {
        identifier: otpData.identifier,
        type: otpData.type,
        purpose: otpData.purpose,
        verifiedAt: otpData.verifiedAt
      }
    };
  }

  async sendPasswordReset(identifier, type = 'email', username = 'User') {
    const resetToken = this.generateSecureToken(32);
    const expiresAt = Date.now() + (60 * 60 * 1000); // 1 hour

    let sendResult;
    try {
      if (type === 'email') {
        sendResult = await emailService.sendPasswordResetEmail(identifier, resetToken, username);
      } else if (type === 'sms') {
        const resetCode = this.generateOTP();
        sendResult = await smsService.sendPasswordResetSMS(identifier, resetCode, username);
        // For SMS, use the code as the token
        resetToken = resetCode.toString();
      } else {
        throw new Error(`Unsupported reset type: ${type}`);
      }

      // Store reset token
      const resetData = {
        token: resetToken,
        identifier,
        type,
        expiresAt,
        used: false,
        createdAt: Date.now()
      };

      this.otpStorage.set(`reset_${resetToken}`, resetData);

      console.log(`🔑 Password reset sent via ${type} to ${identifier}`);

      return {
        success: true,
        resetToken,
        expiresAt,
        type,
        service: sendResult.service
      };
    } catch (error) {
      console.error(`❌ Failed to send password reset via ${type}:`, error.message);
      return {
        success: false,
        error: 'send_failed',
        message: `Failed to send password reset via ${type}`,
        details: error.message
      };
    }
  }

  async verifyPasswordReset(resetToken, identifier = null) {
    const resetData = this.otpStorage.get(`reset_${resetToken}`);
    
    if (!resetData) {
      return {
        success: false,
        error: 'invalid_token',
        message: 'Invalid or expired reset token'
      };
    }

    if (resetData.used) {
      return {
        success: false,
        error: 'already_used',
        message: 'This reset token has already been used'
      };
    }

    if (Date.now() > resetData.expiresAt) {
      this.otpStorage.delete(`reset_${resetToken}`);
      return {
        success: false,
        error: 'expired',
        message: 'Reset token has expired'
      };
    }

    if (identifier && resetData.identifier !== identifier) {
      return {
        success: false,
        error: 'identifier_mismatch',
        message: 'Reset token not valid for this account'
      };
    }

    // Mark as used
    resetData.used = true;
    resetData.usedAt = Date.now();
    this.otpStorage.set(`reset_${resetToken}`, resetData);

    console.log(`✅ Password reset token verified: ${resetToken}`);

    return {
      success: true,
      resetData: {
        identifier: resetData.identifier,
        type: resetData.type,
        usedAt: resetData.usedAt
      }
    };
  }

  async sendSecurityAlert(identifier, type = 'email', message, username = 'User') {
    try {
      let sendResult;
      if (type === 'email') {
        // For security alerts via email, we can extend emailService or send a simple notification
        sendResult = await emailService.sendOTPEmail(identifier, 'SECURITY', username); // Placeholder
      } else if (type === 'sms') {
        sendResult = await smsService.sendSecurityAlert(identifier, message, username);
      }

      console.log(`🔒 Security alert sent via ${type} to ${identifier}`);
      return { success: true, service: sendResult.service };
    } catch (error) {
      console.error(`❌ Failed to send security alert via ${type}:`, error.message);
      return { success: false, error: error.message };
    }
  }

  // Utility methods
  getOTPStats() {
    const active = Array.from(this.otpStorage.values()).filter(otp => !otp.verified && Date.now() < otp.expiresAt);
    const expired = Array.from(this.otpStorage.values()).filter(otp => Date.now() >= otp.expiresAt);
    const verified = Array.from(this.otpStorage.values()).filter(otp => otp.verified);

    return {
      total: this.otpStorage.size,
      active: active.length,
      expired: expired.length,
      verified: verified.length
    };
  }

  cleanupExpired() {
    let cleaned = 0;
    for (const [key, otpData] of this.otpStorage.entries()) {
      if (Date.now() > otpData.expiresAt + (24 * 60 * 60 * 1000)) { // Keep for 24h after expiry
        this.otpStorage.delete(key);
        cleaned++;
      }
    }
    
    // Clean up resend cooldowns
    const cooldownExpiry = this.config.resendCooldownSeconds * 1000;
    for (const [key, timestamp] of this.resendCooldowns.entries()) {
      if (Date.now() - timestamp > cooldownExpiry) {
        this.resendCooldowns.delete(key);
      }
    }

    if (cleaned > 0) {
      console.log(`🧹 Cleaned up ${cleaned} expired OTP records`);
    }
    return cleaned;
  }

  // Validation methods
  validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  validatePhoneNumber(phoneNumber) {
    // Basic international phone number validation
    const phoneRegex = /^\+[1-9]\d{1,14}$/;
    return phoneRegex.test(phoneNumber);
  }

  // TOTP methods for authenticator apps
  generateSecret(issuer = 'Muli Modal', accountName) {
    const secret = speakeasy.generateSecret({
      name: `${issuer}:${accountName}`,
      issuer: issuer,
      length: 32
    });
    return secret;
  }

  async generateQRCode(secret, issuer = 'Muli Modal', accountName) {
    try {
      const otpAuthUrl = speakeasy.otpauthURL({
        secret: secret.base32,
        label: `${issuer}:${accountName}`,
        issuer: issuer,
        algorithm: 'sha1',
        digits: 6,
        period: 30
      });
      
      const qrCodeDataURL = await QRCode.toDataURL(otpAuthUrl);
      return qrCodeDataURL;
    } catch (error) {
      throw new Error(`Failed to generate QR code: ${error.message}`);
    }
  }

  verifyTOTP(token, secret) {
    return speakeasy.totp.verify({
      secret: secret.base32 || secret,
      encoding: 'base32',
      token: token,
      window: parseInt(process.env.TOTP_WINDOW) || 1
    });
  }

  // Backup codes
  generateBackupCodes(count = 8, length = 8) {
    const codes = [];
    for (let i = 0; i < count; i++) {
      const code = crypto.randomBytes(Math.ceil(length / 2)).toString('hex').slice(0, length).toUpperCase();
      codes.push(code);
    }
    return codes;
  }

  // Random OTP generation (for compatibility with old API)
  generateRandomOTP(length = 6) {
    return this.generateOTP();
  }

  // Email/SMS sending methods (for compatibility with old API)
  async sendOTPEmail(email, otp, username = 'User') {
    return await emailService.sendOTPEmail(email, otp, username);
  }

  async sendOTPSMS(phoneNumber, otp, username = 'User') {
    return await smsService.sendOTPSMS(phoneNumber, otp, username);
  }

  // Test services
  async testServices() {
    const emailTest = await emailService.testConnection();
    const smsTest = await smsService.testConnection();
    
    return {
      email: emailTest,
      sms: smsTest,
      otp: {
        success: true,
        message: 'OTP service operational',
        config: this.config
      }
    };
  }
}

// Auto-cleanup every hour
const otpService = new OTPService();
setInterval(() => {
  otpService.cleanupExpired();
}, 60 * 60 * 1000);

module.exports = otpService;