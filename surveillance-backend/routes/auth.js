const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');
const { randomInt } = require('crypto');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { protect } = require('../middleware/auth');

// Generate JWT Token
const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE || '7d'
  });
};

// Generate OTP (6 digits)
const generateOTP = () => {
  return randomInt(100000, 1000000).toString();
};

// @route   POST /api/auth/register
// @desc    Register user and send OTP
// @access  Public
router.post('/register', async (req, res) => {
  try {
    const body = req.body || {};
    const username = typeof body.username === 'string' ? body.username.trim() : '';
    const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : '';
    const mobile = typeof body.mobile === 'string' ? body.mobile.trim() : '';
    const password = typeof body.password === 'string' ? body.password : '';
    const confirmPassword = typeof body.confirmPassword === 'string' ? body.confirmPassword : '';

    // Validation
    if (!username || !email || !mobile || !password || !confirmPassword) {
      return res.status(400).json({
        success: false,
        message: 'Please provide all required fields'
      });
    }

    if (username.length < 3 || !/^\S+@\S+\.\S+$/.test(email) || !/^\d{10}$/.test(mobile) || password.length < 6) {
      return res.status(400).json({
        success: false,
        message: 'Provide a username of at least 3 characters, a valid email, a 10-digit mobile number, and a password of at least 6 characters.'
      });
    }

    if (password !== confirmPassword) {
      return res.status(400).json({
        success: false,
        message: 'Passwords do not match'
      });
    }

    if (process.env.NODE_ENV !== 'development') {
      return res.status(503).json({
        success: false,
        message: 'Email/SMS OTP delivery is not configured. Registration is available in development mode only.'
      });
    }

    // Check if user exists
    const existingUser = await User.findOne({ 
      $or: [{ email }, { username }, { mobile }] 
    });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'User already exists with this email, username, or mobile'
      });
    }

    // Generate OTP
    const otp = generateOTP();
    const otpExpires = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    // Create user
    const user = await User.create({
      username,
      email,
      mobile,
      password,
      otp,
      otpExpires,
      isVerified: false
    });

    console.log(`📱 Development OTP for ${email}: ${otp}`);

    res.status(201).json({
      success: true,
      message: process.env.NODE_ENV === 'development'
        ? 'Registration successful. Use the development OTP returned by the API.'
        : 'Registration successful. OTP delivery is not configured; contact the administrator.',
      data: {
        userId: user._id,
        email: user.email,
        mobile: user.mobile,
        otp: process.env.NODE_ENV === 'development' ? otp : undefined // Only in dev
      }
    });

  } catch (error) {
    console.error('Register error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Registration failed'
    });
  }
});

// @route   POST /api/auth/verify-otp
// @desc    Verify OTP and activate account
// @access  Public
router.post('/verify-otp', async (req, res) => {
  try {
    const body = req.body || {};
    const userId = body.userId;
    const otp = typeof body.otp === 'string' ? body.otp.trim() : '';

    if (!mongoose.Types.ObjectId.isValid(userId) || !otp) {
      return res.status(400).json({
        success: false,
        message: 'Please provide userId and OTP'
      });
    }

    const user = await User.findById(userId).select('+otp +otpExpires');

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    if (user.isVerified) {
      return res.status(400).json({
        success: false,
        message: 'User already verified'
      });
    }

    if (!user.otp || user.otp !== otp) {
      return res.status(400).json({
        success: false,
        message: 'Invalid OTP'
      });
    }

    if (user.otpExpires < Date.now()) {
      return res.status(400).json({
        success: false,
        message: 'OTP expired'
      });
    }

    // Verify user
    user.isVerified = true;
    user.otp = undefined;
    user.otpExpires = undefined;
    await user.save();

    const token = generateToken(user._id);

    res.status(200).json({
      success: true,
      message: 'Account verified successfully',
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        mobile: user.mobile,
        isVerified: user.isVerified
      }
    });

  } catch (error) {
    console.error('Verify OTP error:', error);
    res.status(500).json({
      success: false,
      message: 'OTP verification failed'
    });
  }
});

// @route   POST /api/auth/resend-otp
// @desc    Resend OTP
// @access  Public
router.post('/resend-otp', async (req, res) => {
  try {
    const { userId } = req.body || {};
    if (!mongoose.Types.ObjectId.isValid(userId)) {
      return res.status(400).json({ success: false, message: 'Please provide userId' });
    }

    const user = await User.findById(userId);

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    if (user.isVerified) {
      return res.status(400).json({
        success: false,
        message: 'User already verified'
      });
    }

    if (process.env.NODE_ENV !== 'development') {
      return res.status(503).json({
        success: false,
        message: 'Email/SMS OTP delivery is not configured. OTP resend is available in development mode only.'
      });
    }

    const otp = generateOTP();
    const otpExpires = new Date(Date.now() + 10 * 60 * 1000);

    user.otp = otp;
    user.otpExpires = otpExpires;
    await user.save();

    console.log(`📱 Development OTP regenerated for ${user.email}: ${otp}`);

    res.status(200).json({
      success: true,
      message: process.env.NODE_ENV === 'development'
        ? 'OTP regenerated. Use the development OTP returned by the API.'
        : 'OTP regenerated, but delivery is not configured.',
      otp: process.env.NODE_ENV === 'development' ? otp : undefined
    });

  } catch (error) {
    console.error('Resend OTP error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to resend OTP'
    });
  }
});

// @route   POST /api/auth/login
// @desc    Login user
// @access  Public
router.post('/login', async (req, res) => {
  try {
    const body = req.body || {};
    const identifier = typeof body.identifier === 'string' ? body.identifier.trim() : (typeof body.username === 'string' ? body.username.trim() : '');
    const password = typeof body.password === 'string' ? body.password : '';

    if (!identifier || !password) {
      return res.status(400).json({
        success: false,
        message: 'Please provide a username or email and password'
      });
    }

    const normalizedIdentifier = identifier.toLowerCase();
    const user = await User.findOne({
      $or: [{ username: identifier }, { email: normalizedIdentifier }]
    }).select('+password');

    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    const isMatch = await user.comparePassword(password);

    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    if (!user.isVerified) {
      return res.status(403).json({
        success: false,
        message: 'Please verify your account first',
        userId: user._id
      });
    }

    const token = generateToken(user._id);

    res.status(200).json({
      success: true,
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        mobile: user.mobile,
        isVerified: user.isVerified
      }
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Login failed'
    });
  }
});

// @route   GET /api/auth/me
// @desc    Get current user
// @access  Private
router.get('/me', protect, async (req, res) => {
  try {
    res.status(200).json({
      success: true,
      user: {
        id: req.user._id,
        username: req.user.username,
        email: req.user.email,
        mobile: req.user.mobile,
        isVerified: req.user.isVerified
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to get user data'
    });
  }
});

module.exports = router;
