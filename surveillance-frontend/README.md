# 🚀 Next-Gen AI Surveillance System

A full-stack **MERN** application for advanced AI-powered surveillance with real-time threat detection and monitoring.

![Tech Stack](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Express](https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoColor=white)

---

## ✨ Features

### 🎨 **Landing Page**
- Dark, futuristic theme with glassmorphism
- Animated background elements
- Feature showcase cards
- Call-to-action sections

### 🔐 **Authentication System**
- **Login:** Secure JWT-based authentication
- **Register:** Multi-step registration process
  - Step 1: User details entry
  - Step 2: OTP verification (email & SMS)
  - Step 3: Account activation
- Password hashing with bcrypt
- OTP generation and validation

### 📊 **Dashboard**
- **Live Feeds Tab:**
  - Real-time camera monitoring
  - Multiple source support (Built-in, RTSP, IP, YouTube, File)
  - AI agents progress tracking
  - Recent alerts sidebar

- **Alerts & Events Tab:**
  - Security events table
  - Search and filter capabilities
  - Timestamp tracking
  - Risk level classification

- **Analytics Tab:**
  - Hourly detection charts
  - Threat distribution visualization
  - Historical data analysis

- **Configuration Tab:**
  - Email alert thresholds
  - SMS alert settings
  - Customizable risk parameters

---

## 🛠️ Technology Stack

### Frontend:
- **React.js** - UI framework
- **React Router** - Navigation
- **Axios** - API communication
- **CSS3** - Custom styling with glassmorphism

### Backend:
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM
- **JWT** - Authentication
- **bcryptjs** - Password hashing

---

## 📁 Project Structure

```
surveillance-frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   ├── pages/
│   │   ├── LandingPage.js
│   │   ├── AuthPage.js
│   │   └── Dashboard.js
│   ├── context/
│   │   └── AuthContext.js
│   ├── utils/
│   │   └── api.js
│   ├── App.js
│   └── index.js
└── package.json

surveillance-backend/
├── models/
│   ├── User.js
│   └── Config.js
├── routes/
│   ├── auth.js
│   └── config.js
├── middleware/
│   └── auth.js
├── server.js
├── .env
└── package.json
```

---

## 🚀 Quick Start

### Prerequisites:
- Node.js (v16+)
- MongoDB (v5+)
- npm or yarn

### Option 1: Automated Setup
```powershell
# Run from d:\muli_modal directory
.\QUICK_START.bat
```

### Option 2: Manual Setup

#### 1. Start MongoDB:
```powershell
net start MongoDB
```

#### 2. Install Dependencies:
```powershell
# Backend
cd surveillance-backend
npm install

# Frontend
cd ..\surveillance-frontend
npm install
```

#### 3. Configure Environment:
Edit `surveillance-backend/.env`:
```env
PORT=5001
MONGODB_URI=mongodb://localhost:27017/surveillance
JWT_SECRET=your_super_secret_jwt_key
JWT_EXPIRE=7d
```

#### 4. Start Servers:
```powershell
# Backend (Terminal 1)
cd surveillance-backend
npm start

# Frontend (Terminal 2)
cd surveillance-frontend
npm start
```

#### 5. Access Application:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5001/api

---

## 📡 API Endpoints

### Authentication (`/api/auth`)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/register` | Create new user account | Public |
| POST | `/verify-otp` | Verify OTP code | Public |
| POST | `/resend-otp` | Resend OTP code | Public |
| POST | `/login` | Authenticate user | Public |
| GET | `/me` | Get current user | Protected |

### Configuration (`/api/config`)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/` | Get user configuration | Protected |
| POST | `/` | Update configuration | Protected |

---

## 🎨 UI Design

### Color Palette:
- **Primary Background:** `#0a192f` (Dark Navy)
- **Accent Colors:**
  - Cyan: `#00ffff`
  - Teal: `#00d4d4`
  - Green: `#00ff88`
- **Text:** `#e6f1ff` (Light Blue-White)

### Effects:
- **Glassmorphism:** Frosted glass with subtle borders
- **Glow:** Neon glow on interactive elements
- **Animations:** Smooth fade-in, slide-in, float effects
- **Transitions:** 0.3s ease on all interactive elements

### Typography:
- **Font Family:** 'Poppins', sans-serif
- **Weights:** 300 (Light), 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold)

---

## 🔐 Security Features

- ✅ JWT token-based authentication
- ✅ Password hashing with bcrypt (12 rounds)
- ✅ OTP verification (6-digit code)
- ✅ Token expiration (7 days default)
- ✅ Protected routes
- ✅ CORS enabled
- ✅ Input validation
- ✅ Error handling

---

## 📊 Database Schema

### Users Collection:
```javascript
{
  _id: ObjectId,
  username: String (unique, required),
  email: String (unique, required),
  mobile: String (required, 10 digits),
  password: String (hashed, required),
  isVerified: Boolean (default: false),
  otp: String (6 digits),
  otpExpires: Date,
  createdAt: Date (default: now)
}
```

### Config Collection:
```javascript
{
  _id: ObjectId,
  userId: ObjectId (ref: User),
  emailAlerts: {
    enabled: Boolean (default: true),
    highRiskThreshold: Number (default: 0.7)
  },
  smsAlerts: {
    enabled: Boolean (default: false),
    mediumRiskThreshold: Number (default: 0.3)
  },
  cameraSource: {
    type: String (builtin/rtsp/ip/youtube/file),
    url: String,
    username: String,
    password: String
  },
  updatedAt: Date
}
```

---

## 🧪 Testing

### Test User Registration:
1. Navigate to http://localhost:3000/login
2. Click "REGISTER" tab
3. Fill in details:
   - Username: `testuser`
   - Email: `test@example.com`
   - Mobile: `1234567890`
   - Password: `password123`
4. Check backend console for OTP
5. Enter OTP and verify
6. Login with credentials

---

## 🐛 Troubleshooting

### MongoDB Connection Error:
```powershell
# Ensure MongoDB is running
net start MongoDB

# Or use MongoDB Atlas (cloud)
```

### Port Already in Use:
```powershell
# Backend: Change PORT in .env
# Frontend: Change port in package.json
```

### Dependencies Installation Failed:
```powershell
# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

---

## 📝 Environment Variables

### Backend `.env`:
```env
PORT=5001
MONGODB_URI=mongodb://localhost:27017/surveillance
JWT_SECRET=your_super_secret_jwt_key_change_in_production
JWT_EXPIRE=7d
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
```

---

## 🚀 Deployment

### Backend (Node.js/Express):
- Deploy to: Heroku, AWS, DigitalOcean, Render
- Set environment variables
- Connect to MongoDB Atlas

### Frontend (React):
- Build: `npm run build`
- Deploy to: Vercel, Netlify, AWS S3, Firebase Hosting

---

## 📄 License

MIT License - feel free to use this project for your own purposes.

---

## 👨‍💻 Author

Built with ❤️ using MERN stack for Next-Gen AI Surveillance

---

## 🎉 Getting Started

```powershell
# Clone or navigate to project
cd d:\muli_modal

# Quick start (installs and runs everything)
.\QUICK_START.bat

# Or manually start servers
.\START_SERVERS.bat
```

**Access the application at:** http://localhost:3000

**Happy Monitoring! 🚀**
