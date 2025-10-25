/**
 * 🚀 FUTURISTIC USER MODEL
 * Advanced user schema with comprehensive security features
 */

const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  // Basic User Information
  username: {
    type: String,
    required: [true, 'Username is required'],
    unique: true,
    trim: true,
    minlength: [3, 'Username must be at least 3 characters'],
    maxlength: [30, 'Username cannot exceed 30 characters'],
    match: [/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, hyphens, and underscores']
  },

  email: {
    type: String,
    required: [true, 'Email is required'],
    unique: true,
    lowercase: true,
    trim: true,
    match: [/^\S+@\S+\.\S+$/, 'Please provide a valid email address']
  },

  password: {
    type: String,
    required: [true, 'Password is required'],
    minlength: [8, 'Password must be at least 8 characters']
  },

  phoneNumber: {
    type: String,
    sparse: true, // Allows multiple null values but enforces uniqueness for non-null values
    match: [/^\+[1-9]\d{1,14}$/, 'Please provide a valid international phone number']
  },

  // Profile Information
  profile: {
    firstName: {
      type: String,
      trim: true,
      maxlength: [50, 'First name cannot exceed 50 characters']
    },
    lastName: {
      type: String,
      trim: true,
      maxlength: [50, 'Last name cannot exceed 50 characters']
    },
    avatar: {
      type: String, // URL to avatar image
      default: null
    },
    bio: {
      type: String,
      maxlength: [500, 'Bio cannot exceed 500 characters']
    },
    dateOfBirth: {
      type: Date
    },
    timezone: {
      type: String,
      default: 'UTC'
    }
  },

  // Two-Factor Authentication
  totpSecret: {
    type: String,
    required: true // Generated during registration
  },

  backupCodes: [{
    type: String // Hashed backup codes
  }],

  // User Preferences
  preferences: {
    otpMethod: {
      type: String,
      enum: ['email', 'sms', 'totp'],
      default: 'email'
    },
    twoFactorEnabled: {
      type: Boolean,
      default: false
    },
    theme: {
      type: String,
      enum: ['dark', 'light', 'auto'],
      default: 'dark'
    },
    language: {
      type: String,
      default: 'en'
    },
    notifications: {
      email: {
        type: Boolean,
        default: true
      },
      sms: {
        type: Boolean,
        default: false
      },
      push: {
        type: Boolean,
        default: true
      },
      marketing: {
        type: Boolean,
        default: false
      }
    }
  },

  // Security Settings
  securitySettings: {
    loginAttempts: {
      type: Number,
      default: 0
    },
    lastLoginAttempt: {
      type: Date
    },
    accountLocked: {
      type: Boolean,
      default: false
    },
    lockUntil: {
      type: Date
    },
    passwordChangedAt: {
      type: Date,
      default: Date.now
    },
    sessionTimeout: {
      type: Number, // Minutes
      default: 60
    }
  },

  // Current OTP (temporary storage)
  currentOTP: {
    hash: String,
    expiry: Date,
    method: {
      type: String,
      enum: ['email', 'sms']
    },
    attempts: {
      type: Number,
      default: 0
    }
  },

  // Account Status
  accountStatus: {
    isActive: {
      type: Boolean,
      default: true
    },
    isVerified: {
      type: Boolean,
      default: false
    },
    verificationToken: String,
    verificationTokenExpiry: Date,
    suspensionReason: String,
    suspendedAt: Date,
    suspendedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }
  },

  // Login History
  lastLogin: {
    type: Date
  },

  loginHistory: [{
    timestamp: {
      type: Date,
      default: Date.now
    },
    ip: String,
    userAgent: String,
    location: {
      country: String,
      city: String,
      coordinates: {
        latitude: Number,
        longitude: Number
      }
    },
    method: {
      type: String,
      enum: ['email', 'sms', 'totp', 'backup']
    }
  }],

  // API Usage (if applicable)
  apiUsage: {
    totalRequests: {
      type: Number,
      default: 0
    },
    lastApiCall: Date,
    rateLimit: {
      requestsPerHour: {
        type: Number,
        default: 1000
      },
      requestsThisHour: {
        type: Number,
        default: 0
      },
      resetTime: Date
    }
  },

  // Subscription/Plan Information
  subscription: {
    plan: {
      type: String,
      enum: ['free', 'basic', 'premium', 'enterprise'],
      default: 'free'
    },
    startDate: Date,
    endDate: Date,
    isActive: {
      type: Boolean,
      default: true
    },
    features: [String] // Array of enabled features
  },

  // Terms and Privacy
  agreements: {
    termsAccepted: {
      type: Boolean,
      default: false
    },
    termsAcceptedAt: Date,
    privacyAccepted: {
      type: Boolean,
      default: false
    },
    privacyAcceptedAt: Date,
    marketingConsent: {
      type: Boolean,
      default: false
    }
  },

  // Metadata
  metadata: {
    registrationIP: String,
    registrationUserAgent: String,
    referralSource: String,
    tags: [String], // For admin categorization
    notes: String // Admin notes
  }

}, {
  timestamps: true, // Automatically adds createdAt and updatedAt
  
  // Indexes for better query performance
  indexes: [
    { email: 1 },
    { username: 1 },
    { phoneNumber: 1 },
    { 'accountStatus.isActive': 1 },
    { 'securitySettings.accountLocked': 1 },
    { createdAt: -1 }
  ]
});

// Virtual for full name
userSchema.virtual('profile.fullName').get(function() {
  if (this.profile.firstName && this.profile.lastName) {
    return `${this.profile.firstName} ${this.profile.lastName}`;
  }
  return this.profile.firstName || this.profile.lastName || this.username;
});

// Virtual for account age
userSchema.virtual('accountAge').get(function() {
  const now = new Date();
  const created = this.createdAt;
  const diffTime = Math.abs(now - created);
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return diffDays;
});

// Pre-save middleware
userSchema.pre('save', function(next) {
  // Update passwordChangedAt when password is modified
  if (this.isModified('password') && !this.isNew) {
    this.securitySettings.passwordChangedAt = new Date();
  }
  
  // Ensure backup codes array doesn't exceed 20 items
  if (this.backupCodes && this.backupCodes.length > 20) {
    this.backupCodes = this.backupCodes.slice(-20);
  }
  
  // Ensure login history doesn't exceed 50 items
  if (this.loginHistory && this.loginHistory.length > 50) {
    this.loginHistory = this.loginHistory.slice(-50);
  }
  
  next();
});

// Instance methods
userSchema.methods.toJSON = function() {
  const userObject = this.toObject();
  
  // Remove sensitive fields from JSON output
  delete userObject.password;
  delete userObject.totpSecret;
  delete userObject.backupCodes;
  delete userObject.currentOTP;
  delete userObject.accountStatus.verificationToken;
  
  return userObject;
};

userSchema.methods.isAccountLocked = function() {
  return this.securitySettings.accountLocked && 
         this.securitySettings.lockUntil && 
         this.securitySettings.lockUntil > new Date();
};

userSchema.methods.getPublicProfile = function() {
  return {
    id: this._id,
    username: this.username,
    profile: {
      firstName: this.profile.firstName,
      lastName: this.profile.lastName,
      avatar: this.profile.avatar,
      bio: this.profile.bio
    },
    accountAge: this.accountAge,
    isVerified: this.accountStatus.isVerified,
    lastActive: this.lastLogin
  };
};

// Static methods
userSchema.statics.findActiveUsers = function() {
  return this.find({
    'accountStatus.isActive': true,
    'accountStatus.isVerified': true
  });
};

userSchema.statics.findByEmail = function(email) {
  return this.findOne({ 
    email: email.toLowerCase(),
    'accountStatus.isActive': true 
  });
};

userSchema.statics.findByUsername = function(username) {
  return this.findOne({ 
    username: username,
    'accountStatus.isActive': true 
  });
};

// Compound indexes for better performance
userSchema.index({ email: 1, 'accountStatus.isActive': 1 });
userSchema.index({ username: 1, 'accountStatus.isActive': 1 });
userSchema.index({ 'securitySettings.accountLocked': 1, 'securitySettings.lockUntil': 1 });
userSchema.index({ 'subscription.plan': 1, 'subscription.isActive': 1 });
userSchema.index({ createdAt: -1 });

module.exports = mongoose.model('User', userSchema);