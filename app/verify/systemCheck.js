// File: backend/systemCheck.js
import mongoose from 'mongoose';
import axios from 'axios';
import dotenv from 'dotenv';
import Bot from './dbStructure/bot.js'; // Ensure path is correct

dotenv.config();

const WALLET_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4";
const PYTHON_URL = process.env.ML_SERVER_URL || "http://74.208.28.77:8000";

async function runFullDiagnostics() {
    console.log("🚀 STARTING FULL SYSTEM AUDIT...\n");

    // --- STEP 1: DB CONNECTION ---
    try {
        console.log("Step 1: Database Connectivity...");
        if (!process.env.MONGO_URI) throw new Error("MONGO_URI missing from .env");
        
        await mongoose.connect(process.env.MONGO_URI);
        console.log("✅ Database Connected.\n");
    } catch (err) {
        console.error("❌ DB CONNECTION FAILED:", err.message);
        console.log("👉 FIX: Check your password for special characters or IP whitelist in Atlas.");
        process.exit(1);
    }

    // --- STEP 2: DB WRITE PERMISSIONS ---
    try {
        console.log("Step 2: Database Write Permissions...");
        const testId = `diag_${Date.now()}`;
        const testBot = new Bot({
            userId: WALLET_ID,
            botId: testId,
            status: 'stopped',
            symbol: 'BTC-USD',
            timeframe: '1h',
            capitalAllocation: 1000,
            currentBalance: 1000
        });
        
        await testBot.save();
        console.log("✅ DB Write Successful.");
        await Bot.deleteOne({ botId: testId });
        console.log("✅ DB Delete Successful.\n");
    } catch (err) {
        console.error("❌ DB WRITE FAILED:", err.message);
        console.log("👉 FIX: This usually means your Render IP is blocked by MongoDB Atlas Network Access.");
    }

    // --- STEP 3: PYTHON ENGINE REACHABILITY ---
    try {
        console.log("Step 3: Python Engine Ping...");
        const start = Date.now();
        const res = await axios.get(`${PYTHON_URL}/docs`, { timeout: 5000 });
        console.log(`✅ Python Reachable (Response time: ${Date.now() - start}ms)\n`);
    } catch (err) {
        console.error("❌ PYTHON ENGINE UNREACHABLE:", err.message);
        console.log(`👉 FIX: Ensure main3.py is running on ${PYTHON_URL} and port 8000 is open.`);
    }

    // --- STEP 4: HANDSHAKE SIMULATION ---
    try {
        console.log("Step 4: Bot Data Handshake...");
        const res = await axios.get(`${PYTHON_URL}/api/bot/status?userId=${WALLET_ID}`);
        const data = res.data;
        
        if (data && data.candles) {
            console.log(`✅ Handshake Successful. Received ${data.candles.length} candles.`);
        } else {
            console.log("⚠️ Handshake Warning: No candle data returned. Is the bot engaged?");
        }
    } catch (err) {
        console.error("❌ HANDSHAKE FAILED:", err.message);
    }

    console.log("\n🏁 AUDIT COMPLETE.");
    await mongoose.connection.close();
    process.exit();
}

runFullDiagnostics();
