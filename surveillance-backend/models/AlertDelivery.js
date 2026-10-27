const mongoose = require('mongoose');

const AlertDeliverySchema = new mongoose.Schema({
  deliveryKey: { type: String, required: true, unique: true },
  alertId: { type: String, required: true, index: true },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  channel: { type: String, enum: ['email', 'sms'], required: true },
  risk: { type: String, enum: ['HIGH', 'CRITICAL'], required: true },
  status: { type: String, enum: ['sending', 'sent', 'failed'], default: 'sending' },
  attempts: { type: Number, default: 1 },
  providerMessageId: String,
  error: String,
  createdAt: { type: Date, default: Date.now },
  sentAt: Date,
});

module.exports = mongoose.model('AlertDelivery', AlertDeliverySchema);
