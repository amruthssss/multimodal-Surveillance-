const mongoose = require('mongoose');

const ConfigSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true
  },
  emailAlerts: {
    enabled: {
      type: Boolean,
      default: true
    },
    highRiskThreshold: {
      type: Number,
      default: 0.7,
      min: 0,
      max: 1
    }
  },
  smsAlerts: {
    enabled: {
      type: Boolean,
      default: false
    },
    mediumRiskThreshold: {
      type: Number,
      default: 0.3,
      min: 0,
      max: 1
    }
  },
  cameraSource: {
    type: {
      type: String,
      enum: ['builtin', 'rtsp', 'ip', 'youtube', 'file'],
      default: 'builtin'
    },
    url: String,
    username: String,
    password: String
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Config', ConfigSchema);
