// File: force_sync.js
import mongoose from "mongoose";
import axios from "axios";
import dotenv from "dotenv";
import Bot from "./src/backend/dbStructure/bot.js"; // Ensure path is correct

dotenv.config();

const USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4";
const PYTHON_URL = "http://74.208.28.77:8000";

const forceUpdate = async () => {
    console.log("🛠️ FORCING DATA SYNC...");
    await mongoose.connect(process.env.MONGO_URI);
    
    // 1. Get Data from Python
    const pyRes = await axios.get(`${PYTHON_URL}/api/bot/status?userId=${USER_ID}`);
    const candles = pyRes.data.candles || [];
    
    console.log(`📡 Python has ${candles.length} candles.`);

    if (candles.length > 0) {
        // 2. Force Write to DB
        const bot = await Bot.findOne({ userId: USER_ID });
        bot.candles = candles;
        bot.equityCurve = pyRes.data.equityCurve;
        bot.markModified('candles'); // <--- CRITICAL
        await bot.save();
        console.log("✅ SUCCESS: Data forced into DB. Refresh your frontend.");
    } else {
        console.log("❌ Python returned 0 candles. Issue is on VPS side.");
    }
    process.exit();
};

forceUpdate();
