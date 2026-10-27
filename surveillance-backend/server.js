const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');

// Load the root .env file while keeping the API usable for health diagnostics
// when setup is incomplete.
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const app = express();
const defaultOrigins = 'http://localhost:3000,http://127.0.0.1:4300,http://localhost:4300';
const allowedOrigins = (process.env.FRONTEND_URL || defaultOrigins)
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(cors({
  origin(origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('Origin is not allowed by the account API CORS policy.'));
  },
  credentials: true,
}));
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

const databaseReady = () => mongoose.connection.readyState === 1;
const secretReady = () => Boolean(process.env.JWT_SECRET && !process.env.JWT_SECRET.startsWith('replace-with-'));

app.get('/api/health', (_req, res) => {
  const dbReady = databaseReady();
  const jwtReady = secretReady();
  const ready = dbReady && jwtReady;
  const message = !dbReady
    ? 'MongoDB is not connected. Verify MONGODB_URI credentials, database-user access, and the Atlas network rule.'
    : !jwtReady
      ? 'Set a unique JWT_SECRET in the project-root .env file.'
      : 'Account service connected.';

  res.status(ready ? 200 : 503).json({
    status: ready ? 'ok' : 'degraded',
    database: dbReady ? 'connected' : 'disconnected',
    authentication: jwtReady ? 'configured' : 'missing_secret',
    message,
  });
});

const requireAccountServices = (_req, res, next) => {
  if (!secretReady()) {
    return res.status(503).json({ success: false, message: 'The account service needs a unique JWT_SECRET in the project-root .env file.' });
  }
  if (!databaseReady()) {
    return res.status(503).json({ success: false, message: 'MongoDB is not connected. Check MONGODB_URI credentials, database-user access, and the Atlas network rule.' });
  }
  return next();
};

app.use('/api/auth', requireAccountServices, require('./routes/auth'));
app.use('/api/config', requireAccountServices, require('./routes/config'));

app.use((err, _req, res, _next) => {
  console.error(err.stack || err.message);
  res.status(500).json({ success: false, message: 'Internal Server Error' });
});

const PORT = Number(process.env.PORT || 5001);
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/surveillance';
const server = app.listen(PORT, '127.0.0.1', () => {
  console.log(`Account API listening at http://127.0.0.1:${PORT}`);
  if (!secretReady()) console.warn('JWT_SECRET is missing; account routes will report setup status.');
});

mongoose.connection.on('connected', () => console.log('MongoDB connected.'));
mongoose.connection.on('disconnected', () => console.warn('MongoDB disconnected.'));
mongoose.connection.on('error', (error) => console.error(`MongoDB connection error: ${error.message}`));
mongoose.connect(MONGODB_URI, { serverSelectionTimeoutMS: 8000 })
  .catch((error) => console.error(`MongoDB startup connection failed: ${error.message}`));

const shutdown = async () => {
  server.close();
  await mongoose.disconnect().catch(() => {});
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
