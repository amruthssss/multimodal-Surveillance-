// 📱 SMS Service - Supports both real Twilio and mock SMS delivery
const twilio = require('twilio');

class SMSService {
  constructor() {
    this.client = null;
    this.mockMode = process.env.ENABLE_MOCK_SMS === 'true' || !this.hasSMSCredentials();
    
    if (!this.mockMode) {
      this.initializeClient();
    }
    
    console.log(`📱 SMS Service initialized in ${this.mockMode ? 'MOCK' : 'REAL'} mode`);
  }

  hasSMSCredentials() {
    return !!(
      process.env.TWILIO_ACCOUNT_SID &&
      process.env.TWILIO_AUTH_TOKEN &&
      process.env.TWILIO_PHONE_NUMBER
    );
  }

  initializeClient() {
    try {
      this.client = twilio(
        process.env.TWILIO_ACCOUNT_SID,
        process.env.TWILIO_AUTH_TOKEN
      );
      
      console.log('✅ Twilio client initialized successfully');
    } catch (error) {
      console.error('❌ Twilio client initialization failed:', error.message);
      this.mockMode = true;
    }
  }

  formatPhoneNumber(phoneNumber) {
    // Remove all non-digit characters
    let cleaned = phoneNumber.replace(/\D/g, '');
    
    // Add country code if not present
    if (cleaned.length === 10) {
      cleaned = '1' + cleaned; // Assume US/Canada
    }
    
    return '+' + cleaned;
  }

  async sendOTPSMS(phoneNumber, otp, username = 'User') {
    const formattedPhone = this.formatPhoneNumber(phoneNumber);
    const message = `🚀 Muli Modal Verification\n\nYour code: ${otp}\n\nExpires in 10 minutes. Do not share this code.\n\n- Muli Modal Security`;

    if (this.mockMode) {
      return this.mockSendSMS(formattedPhone, message);
    }

    try {
      const messageResponse = await this.client.messages.create({
        body: message,
        from: process.env.TWILIO_PHONE_NUMBER,
        to: formattedPhone
      });

      console.log('✅ SMS sent successfully:', messageResponse.sid);
      return {
        success: true,
        messageSid: messageResponse.sid,
        service: 'real-sms'
      };
    } catch (error) {
      console.error('❌ SMS sending failed:', error.message);
      // Fallback to mock mode
      return this.mockSendSMS(formattedPhone, message);
    }
  }

  async sendSecurityAlert(phoneNumber, message, username = 'User') {
    const formattedPhone = this.formatPhoneNumber(phoneNumber);
    const alertMessage = `🔒 MULI MODAL SECURITY ALERT\n\n${message}\n\nIf this wasn't you, secure your account immediately.\n\n- Muli Modal Security Team`;

    if (this.mockMode) {
      return this.mockSendSMS(formattedPhone, alertMessage);
    }

    try {
      const messageResponse = await this.client.messages.create({
        body: alertMessage,
        from: process.env.TWILIO_PHONE_NUMBER,
        to: formattedPhone
      });

      console.log('✅ Security alert SMS sent successfully:', messageResponse.sid);
      return {
        success: true,
        messageSid: messageResponse.sid,
        service: 'real-sms'
      };
    } catch (error) {
      console.error('❌ Security alert SMS failed:', error.message);
      return this.mockSendSMS(formattedPhone, alertMessage);
    }
  }

  async sendPasswordResetSMS(phoneNumber, resetCode, username = 'User') {
    const formattedPhone = this.formatPhoneNumber(phoneNumber);
    const message = `🔑 Muli Modal Password Reset\n\nYour reset code: ${resetCode}\n\nExpires in 1 hour.\n\n- Muli Modal Security`;

    if (this.mockMode) {
      return this.mockSendSMS(formattedPhone, message);
    }

    try {
      const messageResponse = await this.client.messages.create({
        body: message,
        from: process.env.TWILIO_PHONE_NUMBER,
        to: formattedPhone
      });

      console.log('✅ Password reset SMS sent successfully:', messageResponse.sid);
      return {
        success: true,
        messageSid: messageResponse.sid,
        service: 'real-sms'
      };
    } catch (error) {
      console.error('❌ Password reset SMS failed:', error.message);
      return this.mockSendSMS(formattedPhone, message);
    }
  }

  mockSendSMS(phoneNumber, message) {
    console.log('\n📱 ===== MOCK SMS SERVICE =====');
    console.log('To:', phoneNumber);
    console.log('Message:', message);
    
    // Extract OTP or reset code from message if present
    const codeMatch = message.match(/(?:code|Code):\s*(\d+)/);
    if (codeMatch) {
      console.log('🔐 CODE:', codeMatch[1]);
    }
    
    console.log('📱 ==============================\n');
    
    return {
      success: true,
      messageSid: 'mock_' + Date.now(),
      service: 'mock-sms'
    };
  }

  async testConnection() {
    if (this.mockMode) {
      return { success: true, message: 'Mock mode - no real connection to test' };
    }

    try {
      // Test by fetching account details
      const account = await this.client.api.accounts(process.env.TWILIO_ACCOUNT_SID).fetch();
      return { 
        success: true, 
        message: `Twilio connection successful - Account: ${account.friendlyName}` 
      };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }

  async getDeliveryStatus(messageSid) {
    if (this.mockMode) {
      return { 
        status: 'delivered', 
        service: 'mock-sms',
        message: 'Mock SMS - assumed delivered'
      };
    }

    try {
      const message = await this.client.messages(messageSid).fetch();
      return {
        status: message.status,
        errorCode: message.errorCode,
        errorMessage: message.errorMessage,
        service: 'real-sms'
      };
    } catch (error) {
      return { 
        status: 'unknown', 
        error: error.message,
        service: 'real-sms'
      };
    }
  }
}

module.exports = new SMSService();