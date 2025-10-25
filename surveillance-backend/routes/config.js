const express = require('express');
const router = express.Router();
const Config = require('../models/Config');
const { protect } = require('../middleware/auth');

// @route   GET /api/config
// @desc    Get user configuration
// @access  Private
router.get('/', protect, async (req, res) => {
  try {
    let config = await Config.findOne({ userId: req.user._id });

    if (!config) {
      // Create default config
      config = await Config.create({
        userId: req.user._id
      });
    }

    res.status(200).json({
      success: true,
      data: config
    });

  } catch (error) {
    console.error('Get config error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get configuration'
    });
  }
});

// @route   POST /api/config
// @desc    Update user configuration
// @access  Private
router.post('/', protect, async (req, res) => {
  try {
    const updateData = {
      ...req.body,
      userId: req.user._id,
      updatedAt: Date.now()
    };

    let config = await Config.findOneAndUpdate(
      { userId: req.user._id },
      updateData,
      { new: true, upsert: true, runValidators: true }
    );

    res.status(200).json({
      success: true,
      message: 'Configuration updated successfully',
      data: config
    });

  } catch (error) {
    console.error('Update config error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update configuration'
    });
  }
});

module.exports = router;
