// 📧 Email Service - Supports both real and mock email delivery
const nodemailer = require('nodemailer');

class EmailService {
  constructor() {
    this.transporter = null;
    this.mockMode = process.env.ENABLE_MOCK_EMAIL === 'true' || !this.hasEmailCredentials();
    
    if (!this.mockMode) {
      this.initializeTransporter();
    }
    
    console.log(`📧 Email Service initialized in ${this.mockMode ? 'MOCK' : 'REAL'} mode`);
  }

  hasEmailCredentials() {
    return !!(
      process.env.SMTP_HOST &&
      process.env.SMTP_USER &&
      process.env.SMTP_PASS
    );
  }

  initializeTransporter() {
    try {
      this.transporter = nodemailer.createTransporter({
        host: process.env.SMTP_HOST,
        port: parseInt(process.env.SMTP_PORT) || 587,
        secure: process.env.SMTP_SECURE === 'true',
        auth: {
          user: process.env.SMTP_USER,
          pass: process.env.SMTP_PASS
        },
        tls: {
          rejectUnauthorized: false
        }
      });
      
      // Verify connection
      this.transporter.verify((error, success) => {
        if (error) {
          console.error('❌ Email transporter verification failed:', error.message);
          this.mockMode = true;
        } else {
          console.log('✅ Email transporter verified successfully');
        }
      });
    } catch (error) {
      console.error('❌ Email transporter initialization failed:', error.message);
      this.mockMode = true;
    }
  }

  async sendOTPEmail(email, otp, username = 'User') {
    const subject = '🔐 Your Muli Modal Verification Code';
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
          .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; }
          .header h1 { color: white; margin: 0; font-size: 24px; }
          .content { padding: 40px 30px; text-align: center; }
          .otp-code { font-size: 36px; font-weight: bold; color: #667eea; letter-spacing: 8px; margin: 30px 0; padding: 20px; background: #f8f9ff; border-radius: 8px; border: 2px dashed #667eea; }
          .info { color: #666; margin: 20px 0; line-height: 1.6; }
          .warning { background: #fff3cd; color: #856404; padding: 15px; border-radius: 6px; margin: 20px 0; }
          .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🚀 Muli Modal Authentication</h1>
          </div>
          <div class="content">
            <h2>Hello ${username}!</h2>
            <p class="info">Your verification code is:</p>
            <div class="otp-code">${otp}</div>
            <div class="warning">
              ⚠️ This code will expire in 10 minutes. Do not share this code with anyone.
            </div>
            <p class="info">
              If you didn't request this code, please ignore this email or contact our support team.
            </p>
          </div>
          <div class="footer">
            <p>© 2024 Muli Modal AI System | Secure Authentication Platform</p>
            <p>This is an automated message, please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const mailOptions = {
      from: `${process.env.FROM_NAME || 'Muli Modal'} <${process.env.FROM_EMAIL || process.env.SMTP_USER}>`,
      to: email,
      subject: subject,
      html: html,
      text: `Your Muli Modal verification code is: ${otp}\n\nThis code will expire in 10 minutes. Do not share this code with anyone.`
    };

    if (this.mockMode) {
      return this.mockSendEmail(mailOptions);
    }

    try {
      const info = await this.transporter.sendMail(mailOptions);
      console.log('✅ Email sent successfully:', info.messageId);
      return {
        success: true,
        messageId: info.messageId,
        service: 'real-email'
      };
    } catch (error) {
      console.error('❌ Email sending failed:', error.message);
      // Fallback to mock mode
      return this.mockSendEmail(mailOptions);
    }
  }

  async sendPasswordResetEmail(email, resetToken, username = 'User') {
    const resetUrl = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;
    const subject = '🔑 Reset Your Muli Modal Password';
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
          .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; }
          .header h1 { color: white; margin: 0; font-size: 24px; }
          .content { padding: 40px 30px; text-align: center; }
          .reset-button { display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 20px 0; }
          .reset-button:hover { background: #5a6fd8; }
          .info { color: #666; margin: 20px 0; line-height: 1.6; }
          .warning { background: #fff3cd; color: #856404; padding: 15px; border-radius: 6px; margin: 20px 0; }
          .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🔑 Password Reset Request</h1>
          </div>
          <div class="content">
            <h2>Hello ${username}!</h2>
            <p class="info">We received a request to reset your Muli Modal password.</p>
            <a href="${resetUrl}" class="reset-button">Reset Password</a>
            <div class="warning">
              ⚠️ This link will expire in 1 hour. If you didn't request this reset, please ignore this email.
            </div>
            <p class="info">
              If the button doesn't work, copy and paste this link into your browser:<br>
              <small>${resetUrl}</small>
            </p>
          </div>
          <div class="footer">
            <p>© 2024 Muli Modal AI System | Secure Authentication Platform</p>
            <p>This is an automated message, please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const mailOptions = {
      from: `${process.env.FROM_NAME || 'Muli Modal'} <${process.env.FROM_EMAIL || process.env.SMTP_USER}>`,
      to: email,
      subject: subject,
      html: html,
      text: `Reset your Muli Modal password by clicking this link: ${resetUrl}\n\nThis link will expire in 1 hour.`
    };

    if (this.mockMode) {
      return this.mockSendEmail(mailOptions);
    }

    try {
      const info = await this.transporter.sendMail(mailOptions);
      console.log('✅ Password reset email sent successfully:', info.messageId);
      return {
        success: true,
        messageId: info.messageId,
        service: 'real-email'
      };
    } catch (error) {
      console.error('❌ Password reset email failed:', error.message);
      return this.mockSendEmail(mailOptions);
    }
  }

  mockSendEmail(mailOptions) {
    console.log('\n📧 ===== MOCK EMAIL SERVICE =====');
    console.log('To:', mailOptions.to);
    console.log('Subject:', mailOptions.subject);
    console.log('From:', mailOptions.from);
    
    // Extract OTP from HTML if present
    const otpMatch = mailOptions.html?.match(/class="otp-code">(\d+)</);
    if (otpMatch) {
      console.log('🔐 OTP CODE:', otpMatch[1]);
    }
    
    // Extract reset link if present
    const resetMatch = mailOptions.html?.match(/href="([^"]*reset-password[^"]*)"/);
    if (resetMatch) {
      console.log('🔑 RESET LINK:', resetMatch[1]);
    }
    
    console.log('📧 ================================\n');
    
    return {
      success: true,
      messageId: 'mock_' + Date.now(),
      service: 'mock-email'
    };
  }

  async testConnection() {
    if (this.mockMode) {
      return { success: true, message: 'Mock mode - no real connection to test' };
    }

    try {
      await this.transporter.verify();
      return { success: true, message: 'Email service connection successful' };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }
}

module.exports = new EmailService();