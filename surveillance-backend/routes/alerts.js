const express = require('express');
const crypto = require('node:crypto');
const nodemailer = require('nodemailer');
const router = express.Router();
const Config = require('../models/Config');
const User = require('../models/User');
const AlertDelivery = require('../models/AlertDelivery');
const { protect } = require('../middleware/auth');

const smtpConfigured = () => Boolean(process.env.SMTP_HOST && process.env.SMTP_FROM && process.env.SMTP_USER && process.env.SMTP_PASSWORD);
const smsConfigured = () => Boolean(process.env.TWILIO_ACCOUNT_SID && process.env.TWILIO_AUTH_TOKEN && process.env.TWILIO_FROM);
let transporter;

function getTransporter() {
  if (!transporter) {
    transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT || 587),
      secure: String(process.env.SMTP_SECURE || 'false').toLowerCase() === 'true',
      requireTLS: String(process.env.SMTP_SECURE || 'false').toLowerCase() !== 'true',
      auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASSWORD },
      connectionTimeout: 10000,
      greetingTimeout: 10000,
      socketTimeout: 15000,
    });
  }
  return transporter;
}

function validAccuracy(metric) {
  return metric && Number(metric.labeledSamples) >= 20 && Number(metric.precision) >= 0.9;
}

async function claimDelivery(alertId, risk, channel, userId) {
  const deliveryKey = `${alertId}:${channel}:${risk}:${userId}`;
  try {
    await AlertDelivery.create({ deliveryKey, alertId, userId, channel, risk, status: 'sending' });
    return deliveryKey;
  } catch (error) {
    if (error.code === 11000) {
      const retried = await AlertDelivery.findOneAndUpdate(
        { deliveryKey, status: 'failed' },
        { $set: { status: 'sending', error: '' }, $inc: { attempts: 1 } },
        { new: true },
      );
      return retried ? deliveryKey : null;
    }
    throw error;
  }
}

async function sendEmail(user, config, event) {
  if (!smtpConfigured() || !config.emailAlerts?.enabled) return 'skipped';
  if (event.risk !== 'HIGH' && event.risk !== 'CRITICAL') return 'skipped';
  if (Number(event.confidence) < Number(config.emailAlerts.highRiskThreshold ?? 0.7)) return 'skipped';
  const key = await claimDelivery(event.alertId, event.risk, 'email', user._id);
  if (!key) return 'duplicate';
  try {
    const when = new Date(event.timestamp).toLocaleString();
    const image = event.snapshot?.data && event.snapshot?.mimeType === 'image/jpeg' && event.snapshot.data.length < 900000
      ? [{ filename: `alert-${event.alertId}.jpg`, content: Buffer.from(event.snapshot.data, 'base64'), contentType: 'image/jpeg' }]
      : [];
    const info = await getTransporter().sendMail({
      from: process.env.SMTP_FROM,
      to: user.email,
      subject: `[${event.risk}] ${event.event} · ${event.camera}`,
      text: `Confirmed ${event.event} alert\nSeverity: ${event.risk}\nCamera: ${event.camera}\nTime: ${when}\nConfidence: ${(Number(event.confidence) * 100).toFixed(1)}%\nRepeated detections: ${event.occurrences}\nAccuracy gate: ${event.accuracy.labeledSamples} reviewed samples, ${(Number(event.accuracy.precision) * 100).toFixed(1)}% precision\nDashboard: ${process.env.ALERT_DASHBOARD_URL || 'http://localhost:4300/dashboard'}\n${image.length ? 'A JPEG evidence snapshot is attached.' : 'Open the local dashboard for the evidence snapshot.'}`,
      attachments: image,
    });
    await AlertDelivery.updateOne({ deliveryKey: key }, { $set: { status: 'sent', sentAt: new Date(), providerMessageId: info.messageId } });
    return 'sent';
  } catch (error) {
    await AlertDelivery.updateOne({ deliveryKey: key }, { $set: { status: 'failed', error: String(error.message).slice(0, 500) } });
    console.error('Alert email delivery failed:', error.message);
    return 'failed';
  }
}

async function sendSms(user, config, event) {
  if (!smsConfigured() || !config.smsAlerts?.enabled || event.risk !== 'CRITICAL') return 'skipped';
  const phone = String(user.mobile || '').replace(/\D/g, '');
  if (phone.length !== 10) return 'skipped';
  const key = await claimDelivery(event.alertId, event.risk, 'sms', user._id);
  if (!key) return 'duplicate';
  try {
    const sid = process.env.TWILIO_ACCOUNT_SID;
    const auth = Buffer.from(`${sid}:${process.env.TWILIO_AUTH_TOKEN}`).toString('base64');
    const body = new URLSearchParams({
      To: `+91${phone}`,
      From: process.env.TWILIO_FROM,
      Body: `CRITICAL ${event.event.toUpperCase()} · ${event.camera} · ${(Number(event.confidence) * 100).toFixed(0)}% · ${new Date(event.timestamp).toLocaleString()}. Review: ${process.env.ALERT_DASHBOARD_URL || 'http://localhost:4300/dashboard'}`,
    });
    const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${sid}/Messages.json`, {
      method: 'POST', headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' }, body,
      signal: AbortSignal.timeout(12000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || `Twilio HTTP ${response.status}`);
    await AlertDelivery.updateOne({ deliveryKey: key }, { $set: { status: 'sent', sentAt: new Date(), providerMessageId: result.sid } });
    return 'sent';
  } catch (error) {
    await AlertDelivery.updateOne({ deliveryKey: key }, { $set: { status: 'failed', error: String(error.message).slice(0, 500) } });
    console.error('Critical SMS delivery failed:', error.message);
    return 'failed';
  }
}

router.get('/status', protect, async (_req, res) => {
  res.json({ success: true, data: { smtpConfigured: smtpConfigured(), smsProviderConfigured: smsConfigured(), smsPolicy: 'critical_only' } });
});

router.get('/deliveries', protect, async (req, res) => {
  try {
    const rows = await AlertDelivery.find({ userId: req.user._id })
      .sort({ createdAt: -1 }).limit(20).select('alertId channel risk status attempts error createdAt sentAt').lean();
    return res.json({ success: true, data: rows });
  } catch (error) {
    console.error('Alert delivery history failed:', error.message);
    return res.status(500).json({ success: false, message: 'Could not load alert delivery history.' });
  }
});

router.post('/dispatch', express.json({ limit: '1mb' }), async (req, res) => {
  const expected = process.env.ALERT_DISPATCH_SECRET || '';
  const supplied = req.get('x-alert-dispatch-secret') || '';
  const expectedBytes = Buffer.from(expected);
  const suppliedBytes = Buffer.from(supplied);
  if (!expected || suppliedBytes.length !== expectedBytes.length || !crypto.timingSafeEqual(suppliedBytes, expectedBytes)) {
    return res.status(401).json({ success: false, message: 'Invalid internal dispatch authorization.' });
  }
  const event = req.body || {};
  if (!event.alertId || !['explosion', 'fire', 'accident', 'smoke', 'gunshot'].includes(event.event)
      || !['HIGH', 'CRITICAL'].includes(event.risk) || !validAccuracy(event.accuracy)) {
    return res.status(400).json({ success: false, message: 'Alert payload or event accuracy gate is invalid.' });
  }
  try {
    const configs = await Config.find({ $or: [{ 'emailAlerts.enabled': true }, { 'smsAlerts.enabled': true }] }).select('userId emailAlerts smsAlerts').lean();
    const summaries = { emailed: 0, sms: 0, skipped: 0, failed: 0 };
    for (const config of configs) {
      const user = await User.findById(config.userId).select('email mobile').lean();
      if (!user) continue;
      const emailResult = await sendEmail(user, config, event);
      const smsResult = await sendSms(user, config, event);
      for (const [result, key] of [[emailResult, 'emailed'], [smsResult, 'sms']]) {
        if (result === 'sent') summaries[key] += 1;
        else if (result === 'failed') summaries.failed += 1;
        else if (result === 'skipped') summaries.skipped += 1;
      }
    }
    return res.json({ success: true, summary: summaries });
  } catch (error) {
    console.error('Alert dispatch failed:', error.message);
    return res.status(500).json({ success: false, message: 'Could not dispatch alert notifications.' });
  }
});

module.exports = router;
