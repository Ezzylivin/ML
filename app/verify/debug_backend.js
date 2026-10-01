// File: debug_backend.js
import axios from "axios";
import mongoose from "mongoose";
import dotenv from "dotenv";
dotenv.config();

// 🟢 CONFIG
const ML_SERVER_URL = "http://74.208.28.77:8000";
const USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4";
const MONGO_URI = process.env.MONGO_URI; 

async function runDebug() {
    console.log("🕵️ STARTING BACKEND SIMULATION...");

    if (!MONGO_URI) {
        console.error("❌ Missing MONGO_URI in .env file");
        process.exit(1);
    }

    try {
        await mongoose.connect(MONGO_URI);
        console.log("✅ Connected to MongoDB");

        // 1. Send Start Command
        console.log("👉 1. Sending START to Python...");
        const startPayload = {
            userId: USER_ID,
            config: {
                symbol: "BTC-USD",
                timeframe: "1h",
                capitalAllocation: 300,
                strategies: [{ code: "stoch" }, { code: "bb_fade" }],
                comboConfig: { combinationRule: "AND" }
            }
        };
        
        await axios.post(`${ML_SERVER_URL}/api/bot/start`, startPayload);
        console.log("✅ Python Bot Started.");

        // 2. The Retry Loop (Simulated)
        console.log("👉 2. Entering Wait Loop...");
        let candles = [];

        for (let i = 1; i <= 5; i++) {
            console.log(`   ⏳ Attempt ${i}: Waiting 2 seconds...`);
            await new Promise(r => setTimeout(r, 2000));

            const res = await axios.get(`${ML_SERVER_URL}/api/bot/status`, { params: { userId: USER_ID } });
            const data = res.data;
            
            console.log(`   📡 Python Status: ${data.candles?.length || 0} candles | Balance: $${data.currentBalance}`);

            if (data.candles && data.candles.length > 0) {
                candles = data.candles;
                console.log("   🎉 DATA FOUND!");
                break;
            }
        }

        if (candles.length === 0) {
            console.error("❌ FAILED: Python never returned candles.");
        } else {
            console.log("👉 3. Attempting Database Write...");
            const collection = mongoose.connection.collection("bots");
            const result = await collection.updateOne(
                { userId: USER_ID },
                { $set: { candles: candles, status: "running" } }
            );
            console.log(`✅ MongoDB Update Result: ${result.modifiedCount} documents modified.`);
            console.log("👉 CHECK YOUR FRONTEND NOW.");
        }

    } catch (e) {
        console.error("❌ ERROR:", e.message);
    } finally {
        await mongoose.disconnect();
    }
}

runDebug();
