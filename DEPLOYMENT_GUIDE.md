# ETF Trading Dashboard - Supabase & Vercel Deployment Guide

## 🗄️ Supabase Database Setup

### Step 1: Create Supabase Project
1. Go to [supabase.com](https://supabase.com) and create a new account
2. Click "New Project" and fill in:
   - **Name**: ETF Trading System
   - **Database Password**: (create a strong password)
   - **Region**: Choose closest to your location
3. Wait for project initialization (2-3 minutes)

### Step 2: Get Supabase Credentials
1. In your Supabase dashboard, go to **Settings > API**
2. Copy these values to your `.env` file:
   ```env
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
   ```

### Step 3: Create Database Tables
1. In Supabase dashboard, go to **SQL Editor**
2. Copy and paste the contents of `database_schema.sql`
3. Click **Run** to create all tables
4. Verify tables are created in **Table Editor**

### Step 4: Configure Row Level Security (Optional)
- If you want authentication, uncomment the RLS policies in the SQL file
- For now, tables are public for easier development

## 🚀 Vercel Deployment Setup

### Step 1: Prepare for Vercel
1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

### Step 2: Configure Environment Variables
1. In your Vercel dashboard, go to your project
2. Navigate to **Settings > Environment Variables**
3. Add these variables:

   **Production Variables:**
   ```
   BROKER_API_KEY=i0bd6xlyqau3ivqe
   BROKER_SECRET_KEY=grz7zb7x5e95li24lajrtbnbjprpbvl7
   BROKER_USER_ID=KK2034
   BROKER_PASSWORD=gowlikar06
   SUPABASE_URL=your-supabase-url
   SUPABASE_ANON_KEY=your-supabase-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   ```

### Step 3: Deploy to Vercel
1. In your project folder, run:
   ```bash
   cd "c:\Users\hp\Desktop\ETF\ETF"
   vercel
   ```

2. Follow the prompts:
   - **Set up and deploy**: Yes
   - **Which scope**: Your account
   - **Link to existing project**: No
   - **Project name**: etf-trading-dashboard
   - **Directory**: ./
   - **Override settings**: No

3. Vercel will deploy and provide a URL like: `https://etf-trading-dashboard.vercel.app`

## 🔧 Local Development with Supabase

### Test Supabase Connection
1. Update your `.env` file with real Supabase credentials
2. Run the dashboard:
   ```bash
   python web_dashboard.py
   ```
3. Check logs for "🗄️ Supabase connected" message

### Database Features Enabled
- ✅ **Real-time market data** saved to database
- ✅ **Trading sessions** tracked
- ✅ **Portfolio snapshots** for analytics
- ✅ **Risk metrics** monitoring
- ✅ **Trade history** preservation
- ✅ **ML predictions** logging

## 📊 Dashboard Features with Database

### Real-time Data Persistence
- Market data saved every 2 seconds
- Portfolio snapshots for performance tracking
- Risk metrics for compliance monitoring
- Complete audit trail of all trades

### Analytics Capabilities
- Historical performance analysis
- Risk management reporting
- Trading pattern insights
- ML model accuracy tracking

### API Endpoints Extended
- `/api/data` - Real-time dashboard data
- `/api/history` - Historical trading data
- `/api/analytics` - Portfolio analytics
- `/api/risk-report` - Risk management report

## 🛡️ Security Considerations

### Environment Variables
- Never commit `.env` files to Git
- Use Vercel environment variables for production
- Rotate API keys regularly

### Database Security
- Enable RLS when ready for production
- Use service role key only for admin operations
- Monitor database usage and costs

### Trading Security
- Implement position size limits
- Set daily loss limits
- Use emergency stop functionality

## 📱 Mobile & Performance

### Optimizations for Vercel
- Serverless functions auto-scale
- Global CDN for fast loading
- WebSocket fallback for real-time updates

### Mobile Responsive
- Dashboard works on all devices
- Touch-friendly controls
- Responsive grid layouts

## 🔍 Monitoring & Logs

### Supabase Monitoring
- Database performance metrics
- Query optimization insights
- Storage usage tracking

### Vercel Monitoring
- Function execution logs
- Performance analytics
- Error tracking

## 🚀 Go Live Checklist

### Before Deployment:
- [ ] Supabase project created and configured
- [ ] All environment variables set in Vercel
- [ ] Database schema deployed
- [ ] Local testing with Supabase completed
- [ ] Trading limits configured

### After Deployment:
- [ ] Verify dashboard loads at Vercel URL
- [ ] Test WebSocket connections
- [ ] Confirm market data is being saved
- [ ] Check trading session creation
- [ ] Monitor logs for errors

## 💡 Next Steps

### Enhanced Features to Add:
1. **User Authentication** - Add login/logout
2. **Multiple Strategies** - Portfolio of trading strategies
3. **Alert System** - Email/SMS notifications
4. **Advanced Analytics** - Performance reports
5. **Paper Trading** - Test strategies safely

### Scaling Considerations:
- Upgrade Supabase plan for higher limits
- Implement caching for better performance
- Add error monitoring (Sentry)
- Set up automated backups

---

**Your ETF Trading Dashboard is now ready for cloud deployment! 🚀**

Access your live dashboard at: `https://your-vercel-url.vercel.app`
