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
    const body = req.body || {};
    const updateData = { updatedAt: new Date() };
    const validationErrors = [];

    for (const groupName of ['emailAlerts', 'smsAlerts']) {
      const group = body[groupName];
      if (group === undefined) continue;
      if (!group || typeof group !== 'object' || Array.isArray(group)) {
        validationErrors.push(`${groupName} must be an object`);
        continue;
      }

      if (group.enabled !== undefined) {
        if (typeof group.enabled !== 'boolean') {
          validationErrors.push(`${groupName}.enabled must be boolean`);
        } else {
          updateData[`${groupName}.enabled`] = group.enabled;
        }
      }

      const thresholdName = groupName === 'emailAlerts'
        ? 'highRiskThreshold'
        : 'mediumRiskThreshold';
      if (group[thresholdName] !== undefined) {
        const threshold = group[thresholdName];
        if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
          validationErrors.push(`${groupName}.${thresholdName} must be between 0 and 1`);
        } else {
          updateData[`${groupName}.${thresholdName}`] = threshold;
        }
      }
    }

    if (validationErrors.length) {
      return res.status(400).json({ success: false, message: validationErrors.join('; ') });
    }

    let config = await Config.findOneAndUpdate(
      { userId: req.user._id },
      { $set: updateData, $setOnInsert: { userId: req.user._id } },
      { new: true, upsert: true, runValidators: true, setDefaultsOnInsert: true }
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
